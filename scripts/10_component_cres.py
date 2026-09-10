from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
PROJECT_DIR = Path().cwd().parent
ADDGENE_DIR = PROJECT_DIR / "data/addgene"
ALIGN_DIR = PROJECT_DIR / "data/addgene_alignments"

ELEMENT_POSITIONS = ADDGENE_DIR / "mammalian_plasmids_elements.parquet"
ELEMENT_OVERLAPS = ADDGENE_DIR / "mammalian_plasmids_element_cre_overlaps.parquet"
ELEMENT_CITATIONS = ADDGENE_DIR / "citations_addgene_elements.parquet"
PROMOTER_DISTANCE = ADDGENE_DIR / "mammalian_plasmids_element_promoter_distance.parquet"
ELEMENT_DIVERGENCE = ALIGN_DIR / "element_average_divergence.parquet"
CREST_THRESHOLDS = PROJECT_DIR.parent / "mpra-predictor/data/cre_thresholds_fdr_001.csv"

OUT_CLUSTER_RAW = ADDGENE_DIR / "element_cre_overlap_clustering.csv"
OUT_CLUSTER_RAW_HEAT = ADDGENE_DIR / "element_cre_overlap_clustering_heatmap.csv"

ID_COLUMNS = ["type", "name", "element_length"]
POPULARITY_COLUMNS = ["n_plasmids", "n_citations"]

# Carried through to the written table so the heatmap can annotate its rows with
# them, but deliberately kept out of METRIC_COLUMNS: they describe an element's
# context rather than its predicted activity, and must not shape the clusters.
ANNOTATION_COLUMNS = [
    "promoter_distance_mean", "promoter_distance_median", "is_reference_element",
    "sequence_diversity_pct", "n_sequence_variants",
]

# --- METRIC REGISTRY ---
# `cre_avg_signal` deliberately carries no fixed threshold. CREST activity scales
# differ by ~2.2x between cell lines (GM12878 1.05 vs SHSY5Y 2.35 at FDR 0.01),
# so one shared constant would be far too strict for the former and too lenient
# for the latter. It is instead derived per cell line as the strict CRE-calling
# threshold used in `07_cre_annotation.py`
PER_CELL_THRESHOLD = None

# Entries are (description, heatmap label, threshold). The heatmap label is kept
# short because it is drawn once per cell line as a rotated tick label, where the
# full description would repeat six times and swamp the figure.
CRE_METRIC_SPECS = {
    "n_cre_midpoints": ("# CRE Midpoints per Feature Instance", "# CRE midpoints", 0.25),
    "cre_avg_signal": ("Average activity of CRE base pairs", "Mean CRE activity", PER_CELL_THRESHOLD),
    "fraction_cre_bp": ("Fraction base pairs that are CRE", "Fraction CRE bp", 0.1),
}
TSS_METRIC_SPECS = {
    "n_tss_midpoints": ("# TSS Midpoints per Feature Instance", "# TSS midpoints", 0.25),
    "tss_avg_signal": ("Average activity of TSS base pairs", "Mean TSS activity", 0.1),
}

CRE_SIGNAL_THRESH_SCALE = 1.15  # strict thresholding, as in 07_cre_annotation.py
CRE_SIGNAL_THRESH = {
    row["cell"]: row["threshold"] * CRE_SIGNAL_THRESH_SCALE
    for row in pl.read_csv(CREST_THRESHOLDS).iter_rows(named=True)
}

# Share of the composite score each metric group carries. The CRE side gains a
# column per cell line while the TSS side has two columns in total, so an
# unweighted sum measures CRE coverage rather than initiation: on six cell lines
# CRE outvotes TSS 18:2, and promoter-like elements are demoted for it. Splitting
# the score by group and dividing by the group's column count makes it
# `CRE_WEIGHT * mean(CRE) + TSS_WEIGHT * mean(TSS)`, whatever the cell count.
CRE_WEIGHT = 0.5
TSS_WEIGHT = 0.5

# --- CLUSTERING ---
# Every metric above is computed for every cell line and written to OUT_CLUSTER_RAW.
# Only the columns listed here shape the clusters and the heatmap

# CLUSTERING_CELLS = ["GM12878", "Jurkat", "MRC5", "A549", "HEK293T", "K562", "SHSY5Y", "SiHa"]
CLUSTERING_CELLS = ["GM12878", "MRC5", "A549", "HEK293T", "K562", "SHSY5Y"]

# Metric-major: every cell line's `# CRE midpoints` sits together, then every
# `Mean CRE activity`, and so on. The heatmap draws columns in this order, so
# related columns stay adjacent without relying on a clustering dendrogram.
METRIC_COLUMNS = (
    [f"{metric} ({cell})" for metric in CRE_METRIC_SPECS for cell in CLUSTERING_CELLS]
    + list(TSS_METRIC_SPECS)
)

N_CLUSTERS = 8
CRE_LENGTH_THRESH = 100


def split_metric(column: str) -> tuple[str, str | None]:
    """Split a `<metric> (<cell>)` column into its base metric and cell line."""
    base, separator, cell = column.partition(" (")
    return (base, cell[:-1]) if separator else (base, None)


def metric_label(column: str) -> str:
    """Short heatmap column label for a metric column, cell-line suffix and all."""
    base, cell = split_metric(column)
    if base in TSS_METRIC_SPECS:
        return TSS_METRIC_SPECS[base][1]
    return f"{CRE_METRIC_SPECS[base][1]} [{cell}]"


def metric_threshold(column: str) -> float:
    """Physical activity threshold a metric column must clear to count as active."""
    base, cell = split_metric(column)
    if base in TSS_METRIC_SPECS:
        return TSS_METRIC_SPECS[base][2]
    threshold = CRE_METRIC_SPECS[base][2]
    if threshold is not PER_CELL_THRESHOLD:
        return threshold
    if cell is None:
        raise ValueError(f"'{column}' needs a cell line to resolve its threshold")
    return CRE_SIGNAL_THRESH[cell]


def metric_group_weights(columns: list[str]) -> np.ndarray:
    """Per-column weights giving the CRE and TSS groups a fixed share each.

    Each group's weight is spread evenly over its columns, so adding cell lines
    changes the resolution of the CRE side without changing how much it counts.
    """
    is_tss = np.array([split_metric(column)[0] in TSS_METRIC_SPECS for column in columns])
    weights = np.where(is_tss, TSS_WEIGHT, CRE_WEIGHT).astype(float)
    for group in (is_tss, ~is_tss):
        if group.any():
            weights[group] /= group.sum()
    return weights


def available_metric_columns(columns) -> list[str]:
    """Every metric the overlaps table supports, CRE metrics once per cell line."""
    prefix = "cre_avg_signal ("
    cells = [column[len(prefix):-1] for column in columns if column.startswith(prefix)]
    return [f"{metric} ({cell})" for metric in CRE_METRIC_SPECS for cell in cells] + list(TSS_METRIC_SPECS)


METRIC_LABELS = [metric_label(column) for column in METRIC_COLUMNS]
METRIC_THRESH = [metric_threshold(column) for column in METRIC_COLUMNS]
METRIC_WEIGHTS = metric_group_weights(METRIC_COLUMNS)


def functional_profile_clustering(
    df: pl.DataFrame,
    metric_columns: list[str],
    metric_labels: list[str],
    metric_thresh: list[float],
    metric_weights: np.ndarray,
    n_clusters: int,
    breadth_coeff: int = 50,
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

    # Measure how broadly an element is active, as a group-weighted share of its
    # metrics rather than a raw column count, so breadth is not decided by
    # whichever group happens to contribute more columns
    active_breadth = (activity_mask * metric_weights).sum(axis=1)

    # Apply logic: If below raw threshold, mute interest contribution to 0.0
    effective_z = np.where(activity_mask, scaled_data, 0.0)

    # Cap maximum Z-score contribution per column to prevent single-column outliers from dominating
    clipped_z = np.clip(effective_z, 0, 3.0)

    # Weight by metric group, then compute the robust composite interest score
    weighted_z = clipped_z * metric_weights
    composite_interest = weighted_z.sum(axis=1)

    # --- 3. K-Means Clustering on Robust Profiles ---
    # Clustering on weighted_z ensures groups are formed by overall activation
    # patterns, on the same group balance the composite score uses
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df_pd['raw_cluster'] = kmeans.fit_predict(weighted_z)

    # Attach tracking metrics to the DataFrame
    df_pd['active_breadth'] = active_breadth
    df_pd['composite_z_score'] = composite_interest

    # --- 4. Rank Clusters by Multi-Metric Activity ---
    # Group by cluster and find the average breadth of activity and signal magnitude
    cluster_profiles = df_pd.groupby('raw_cluster').agg({
        'active_breadth': 'mean',
        'composite_z_score': 'mean'
    })

    # Rank clusters: Primary weight on breadth of activity, secondary weight on signal strength
    cluster_rank_metric = (cluster_profiles['active_breadth'] * breadth_coeff) + cluster_profiles['composite_z_score']
    ranked_clusters = cluster_rank_metric.rank(ascending=False, method='min').astype(int)
    priority_mapping = ranked_clusters.to_dict()

    # Map priority ranks back to elements (Priority 1 = Best)
    df_pd['priority_group'] = df_pd['raw_cluster'].map(priority_mapping)

    # --- 5. Sort Elements for Visualization ---
    # Sort strictly by Priority Group (asc), then breadth of activity (desc), then remaining signal (desc)
    df_sorted = df_pd.sort_values(
        by=['priority_group', 'active_breadth', 'composite_z_score'], 
        ascending=[True, False, False]
    ).reset_index(drop=True)

    # Re-extract standard unclipped Z-scores for visualization so raw magnitudes remain visible on the plot
    heatmap_data = scaler.transform(df_sorted[metric_columns])
    df_heatmap = pl.DataFrame(heatmap_data, schema=metric_labels)

    # Clean intermediate tracking columns before returning
    df_sorted = df_sorted.drop(columns=['raw_cluster', 'active_breadth'])

    return pl.from_pandas(df_sorted), df_heatmap


if __name__ == "__main__":

    # 0. Load pre-processed overlap data
    element_citations = pl.read_parquet(ELEMENT_CITATIONS)
    element_lengths = (
        pl.read_parquet(ELEMENT_POSITIONS)
        .group_by(["element_type", "element_name"])
        .agg(pl.col("length").median().cast(pl.Int64))
        .rename({"length": "element_length"})
    )
    promoter_distance = pl.read_parquet(PROMOTER_DISTANCE).select(
        ["element_type", "element_name", "promoter_distance_mean", "promoter_distance_median", "is_reference_element"]
    )
    # Divergence from the element's MSA consensus, as a percentage. Stored in
    # percent rather than as a fraction because the values span four orders of
    # magnitude, and percent is what the figure and any caption quote.
    sequence_diversity = pl.read_parquet(ELEMENT_DIVERGENCE).select([
        "element_type",
        "element_name",
        (100.0 * (1.0 - pl.col("avg_identity"))).alias("sequence_diversity_pct"),
        pl.col("n_instances_unique").alias("n_sequence_variants"),
    ])
    element_cre_overlap = (
        pl.read_parquet(ELEMENT_OVERLAPS)
        .join(element_citations[["element_type", "element_name", "n_plasmids", "n_citations"]], left_on=["type", "name"], right_on=["element_type", "element_name"], how="left")
        .join(element_lengths, left_on=["type", "name"], right_on=["element_type", "element_name"], how="left")
        .join(promoter_distance, left_on=["type", "name"], right_on=["element_type", "element_name"], how="left")
        .join(sequence_diversity, left_on=["type", "name"], right_on=["element_type", "element_name"], how="left")
        .with_columns(pl.max_horizontal("tss_fwd_avg_signal", "tss_rev_avg_signal").alias("tss_avg_signal"))
    )

    # 1. Clustering
    # Every available metric is kept in the frame so the written table carries all
    # cell lines; METRIC_COLUMNS alone decides what the clustering actually sees.
    ALL_METRIC_COLUMNS = available_metric_columns(element_cre_overlap.columns)
    missing = [column for column in METRIC_COLUMNS if column not in ALL_METRIC_COLUMNS]
    if missing:
        raise KeyError(f"METRIC_COLUMNS absent from {ELEMENT_OVERLAPS.name}: {missing}")

    print(f"Writing {len(ALL_METRIC_COLUMNS)} metrics, clustering on {len(METRIC_COLUMNS)}:")
    for column, weight in zip(METRIC_COLUMNS, METRIC_WEIGHTS):
        print(f"  {column:44s} threshold {metric_threshold(column):.5f}   weight {weight:.4f}")

    df = (
        element_cre_overlap
        .filter(pl.col("element_length") >= CRE_LENGTH_THRESH)
        [ID_COLUMNS + ALL_METRIC_COLUMNS + POPULARITY_COLUMNS + ANNOTATION_COLUMNS]
        .with_columns(pl.col(col_name).fill_null(0) for col_name in ALL_METRIC_COLUMNS)
    )
    df_clustered, df_heatmap = functional_profile_clustering(
        df, METRIC_COLUMNS, METRIC_LABELS, METRIC_THRESH, METRIC_WEIGHTS, N_CLUSTERS
    )
    df_heatmap.write_csv(OUT_CLUSTER_RAW_HEAT)

    # 2. Identify Cryptic CRE candidates
    df_clustered = df_clustered.with_columns(
        is_cryptic_cre=(
            (~pl.col("type").is_in(["promoter", "enhancer"])) &
            (pl.col("element_length") >= CRE_LENGTH_THRESH) &
            (pl.col("priority_group") <= N_CLUSTERS - 1)
        )
    )
    df_clustered.write_csv(OUT_CLUSTER_RAW)
