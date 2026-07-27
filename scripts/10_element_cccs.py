from pathlib import Path
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# --- CONFIGURATION ---
ADDGENE_DIR = Path().cwd().parent / "data/addgene"
ELEMENT_OVERLAPS = ADDGENE_DIR / "mammalian_plasmids_element_cre_overlaps.parquet"
ELEMENT_CITATIONS = ADDGENE_DIR / "citations_addgene_elements.parquet"

OUT_CLUSTER_RAW = ADDGENE_DIR / "element_cre_overlap_clustering.csv"
OUT_CLUSTER_RAW_HEAT = ADDGENE_DIR / "element_cre_overlap_clustering_heatmap.csv"

ID_COLUMNS = ["type", "name", "element_length"]
METRIC_COLUMNS = ["n_cre_midpoints", "cre_avg_signal", "fraction_cre_bp", "n_tss_midpoints", "tss_avg_signal"]
POPULARITY_COLUMNS = ["n_plasmids", "n_citations"]

METRIC_LABELS = ["# CRE Midpoints per Feature Instance", "Average activity of CRE base pairs", "Fraction base pairs that are CRE", "# TSS Midpoints per Feature Instance", "Average activity of TSS base pairs"]
METRIC_THRESH = [0.25, 2, 0.1, 0.25, 0.1]
N_CITATIONS_FILTER = 50
N_CLUSTERS = 7
CRE_LENGTH_THRESH = 100


def functional_profile_clustering(
    df: pl.DataFrame,
    metric_columns: list[str],
    metric_labels: list[str],
    metric_thresh: list[float],
    n_clusters: int,
    diversity_coeff: int = 50,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    # --- 1. Data Preparation & Normalization ---
    df_pd = df.to_pandas()

    # Calculate standard Z-scores
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_pd[metric_columns])

    # --- 2. Incorporate Physical Thresholds & Handle Outliers ---
    # Create a boolean mask of elements passing their physical baselines
    thresh_matrix = np.array(metric_thresh)
    activity_mask = df_pd[metric_columns].values >= thresh_matrix

    # Calculate how many metrics are physically active for each element
    n_active_metrics = activity_mask.sum(axis=1)

    # Apply logic: If below raw threshold, mute interest contribution to 0.0
    effective_z = np.where(activity_mask, scaled_data, 0.0)

    # Cap maximum Z-score contribution per column to prevent single-column outliers from dominating
    clipped_z = np.clip(effective_z, 0, 3.0)

    # Compute the robust composite interest score
    composite_interest = clipped_z.sum(axis=1)

    # --- 3. K-Means Clustering on Robust Profiles ---
    # Clustering on clipped_z ensures groups are formed by overall activation patterns
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df_pd['raw_cluster'] = kmeans.fit_predict(clipped_z)

    # Attach tracking metrics to the DataFrame
    df_pd['n_active_metrics'] = n_active_metrics
    df_pd['composite_z_score'] = composite_interest

    # --- 4. Rank Clusters by Multi-Metric Activity ---
    # Group by cluster and find the average number of active metrics and signal magnitude
    cluster_profiles = df_pd.groupby('raw_cluster').agg({
        'n_active_metrics': 'mean',
        'composite_z_score': 'mean'
    })

    # Rank clusters: Primary weight on number of active columns, secondary weight on signal strength
    cluster_rank_metric = (cluster_profiles['n_active_metrics'] * diversity_coeff) + cluster_profiles['composite_z_score']
    ranked_clusters = cluster_rank_metric.rank(ascending=False, method='min').astype(int)
    priority_mapping = ranked_clusters.to_dict()

    # Map priority ranks back to elements (Priority 1 = Best)
    df_pd['priority_group'] = df_pd['raw_cluster'].map(priority_mapping)

    # --- 5. Sort Elements for Visualization ---
    # Sort strictly by Priority Group (asc), then number of active metrics (desc), then remaining signal (desc)
    df_sorted = df_pd.sort_values(
        by=['priority_group', 'n_active_metrics', 'composite_z_score'], 
        ascending=[True, False, False]
    ).reset_index(drop=True)

    # Re-extract standard unclipped Z-scores for visualization so raw magnitudes remain visible on the plot
    heatmap_data = scaler.transform(df_sorted[metric_columns])
    df_heatmap = pl.DataFrame(heatmap_data, schema=metric_labels)

    # Clean intermediate tracking column before returning
    df_sorted = df_sorted.drop(columns=['raw_cluster'])

    return pl.from_pandas(df_sorted), df_heatmap


if __name__ == "__main__":

    # 0. Load pre-processed overlap data
    element_citations = pl.read_parquet(ELEMENT_CITATIONS)
    element_lengths = (
        pl.read_parquet(ADDGENE_DIR / "mammalian_plasmids_elements.parquet")
        .group_by(["element_type", "element_name"])
        .agg(pl.col("length").median().cast(pl.Int64))
        .rename({"length": "element_length"})
    )
    element_cre_overlap = (
        pl.read_parquet(ELEMENT_OVERLAPS)
        .join(element_citations[["element_type", "element_name", "n_plasmids", "n_citations"]], left_on=["type", "name"], right_on=["element_type", "element_name"], how="left")
        .join(element_lengths, left_on=["type", "name"], right_on=["element_type", "element_name"], how="left")
        .with_columns(pl.max_horizontal("tss_fwd_avg_signal", "tss_rev_avg_signal").alias("tss_avg_signal"))
    )

    # 1. Clustering
    df = (
        element_cre_overlap
        .filter(pl.col("element_length") >= CRE_LENGTH_THRESH)
        [ID_COLUMNS + METRIC_COLUMNS + POPULARITY_COLUMNS]
        .with_columns(pl.col(col_name).fill_null(0) for col_name in METRIC_COLUMNS)
    )
    df_clustered, df_heatmap = functional_profile_clustering(df, METRIC_COLUMNS, METRIC_LABELS, METRIC_THRESH, N_CLUSTERS)
    df_heatmap.write_csv(OUT_CLUSTER_RAW_HEAT)

    # 2. Identify Cryptic CRE candidates
    df_clustered = df_clustered.with_columns(
        is_cryptic_cre=(
            (~pl.col("type").is_in(["promoter", "enhancer"])) &
            (pl.col("element_length") >= CRE_LENGTH_THRESH) &
            ((pl.col("n_citations") >= N_CITATIONS_FILTER) | (pl.col("n_citations").is_null())) &
            (pl.col("priority_group") <= N_CLUSTERS - 1)
        )
    )
    df_clustered.write_csv(OUT_CLUSTER_RAW)
