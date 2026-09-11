from pathlib import Path

import numpy as np
import polars as pl
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

# Written to their own files so a gated run is not overwritten. Drop the
# `_nogate` suffixes to make this a drop-in replacement.
OUT_CLUSTER_RAW = ADDGENE_DIR / "element_cre_overlap_clustering.csv"
OUT_CLUSTER_RAW_HEAT = ADDGENE_DIR / "element_cre_overlap_clustering_heatmap.csv"

ID_COLUMNS = ["type", "name", "element_length"]
POPULARITY_COLUMNS = ["n_plasmids", "n_citations"]

# Carried through to the written table so the heatmap can annotate its rows with
# them, but deliberately kept out of the metric columns: they describe an
# element's context rather than its predicted activity, and must not shape the
# categories.
ANNOTATION_COLUMNS = [
    "promoter_distance_mean", "promoter_distance_median", "is_reference_element",
    "sequence_diversity_pct", "n_sequence_variants",
]

# --- METRIC REGISTRY ---
# Entries are (description, heatmap label). The description is documentation
# only; the heatmap label is kept short because it is drawn once per cell line
# as a rotated tick label, where the full description would repeat six times and
# swamp the figure.
#
# No metric carries a calling threshold in this variant of the script: a column
# contributes on its standardised value alone. Because each column is
# standardised separately, the ~2.2x difference in CREST activity scale between
# cell lines (GM12878 1.05 vs SHSY5Y 2.35 at FDR 0.01) is absorbed by the
# z-score rather than by a per-cell threshold. What is lost is the absolute
# floor: an element now scores on a column for being active relative to the
# cohort, whether or not it would be called a CRE there.
CRE_METRIC_SPECS = {
    "n_cre_midpoints": ("# CRE Midpoints per Feature Instance", "# CRE midpoints"),
    "fraction_cre_bp": ("Fraction base pairs that are CRE", "Fraction CRE bp"),
    "cre_avg_signal": ("Average activity of CRE base pairs", "Mean CRE activity"),
}
# Unlike the CRE metrics these are not resolved per cell line: the overlaps table
# carries a single TSS column pair, so the TSS axis has no cross-cell replication.
TSS_METRIC_SPECS = {
    "n_tss_midpoints": ("# TSS Midpoints per Feature Instance", "# TSS midpoints"),
    "tss_avg_signal": ("Average activity of TSS base pairs", "Mean TSS activity"),
}

# Share of the composite score each metric group carries. The CRE side gains a
# column per cell line while the TSS side has two columns in total, so an
# unweighted sum measures CRE coverage rather than initiation: on six cell lines
# CRE outvotes TSS 18:2, and promoter-like elements are demoted for it. Splitting
# the score by group and dividing by the group's column count makes it
# `CRE_WEIGHT * mean(CRE) + TSS_WEIGHT * mean(TSS)`, whatever the cell count.
CRE_WEIGHT = 0.5
TSS_WEIGHT = 0.5

# How much each metric counts towards its typing axis. On the CRE side the first
# two metrics grow with element length - a longer element accumulates more CRE
# midpoints - while the mean activity of the base pairs that are CRE does not.
# Weights are relative and renormalised within a group, so {1, 1, 1} is a plain
# mean and {0, 0, 1} types on mean activity alone. This shapes the categories
# only; the composite score keeps using CRE_WEIGHT / TSS_WEIGHT over all columns
# equally.
AXIS_WEIGHTS = {
    "n_cre_midpoints": 1.0,
    "fraction_cre_bp": 1.0,
    "cre_avg_signal": 1.0,
    "n_tss_midpoints": 1.0,
    "tss_avg_signal": 1.0,
}

# --- CLUSTERING ---
# Every metric is computed for every cell line the overlaps table carries and
# written to OUT_CLUSTER_RAW. Only the cell lines listed here shape the
# categories and the heatmap.

# CLUSTERING_CELLS = ["GM12878", "Jurkat", "MRC5", "A549", "HEK293T", "K562", "SHSY5Y", "SiHa"]
CLUSTERING_CELLS = ["GM12878", "MRC5", "A549", "HEK293T", "K562", "SHSY5Y"]

CRE_LENGTH_THRESH = 100

# Resolved at the bottom of the module, once the helpers below are defined. The
# heatmap's own column labels are its header row, so nothing else needs exporting.
METRIC_COLUMNS: list[str]

# --- CATEGORIES ---
# Elements are typed in the plane of the CRE and TSS halves of the composite
# score - the two quantities the composite is already built from - rather than
# clustered in the 20 metric columns. In the full column space the strongest
# splits separate `n_tss_midpoints` from `tss_avg_signal`, which carry a quarter
# of the score each; that is a split between two ways of measuring initiation,
# not between two kinds of element. Averaging each group to one axis removes it
# and leaves axes that name themselves.
#
# The cut-offs are applied directly rather than through k-means, so an element's
# category does not depend on a randomly seeded centroid search. It does still
# depend on the cohort, because the axes are built from z-scores taken over the
# elements present in the table.
CATEGORY_ACTIVE = 0.35  # below this on both axes an element is inactive
CATEGORY_STRONG = 1.0   # at or above this an element is strong on that axis

CLIP_Z = 3.0  # ceiling on any single column's contribution to an axis

# Display order, and the order `priority_group` numbers follow.
CATEGORY_ORDER = [
    "enhancer & promoter",
    "promoter-only",
    "enhancer-only",
    "weak promoter",
    "weak enhancer",
    "inactive",
]
# Categories whose CRE or TSS evidence is strong enough to call a candidate.
STRONG_CATEGORIES = ["enhancer & promoter", "promoter-only", "enhancer-only"]


# --- METRIC COLUMN HELPERS ---

def metric_columns_for(cells: list[str]) -> list[str]:
    """Metric-major column list: every cell line's `# CRE midpoints` together,
    then every `Mean CRE activity`, and so on, with the TSS pair last. The
    heatmap draws columns in this order, so related columns stay adjacent
    without relying on a clustering dendrogram."""
    return (
        [f"{metric} ({cell})" for metric in CRE_METRIC_SPECS for cell in cells]
        + list(TSS_METRIC_SPECS)
    )


def available_cells(columns) -> list[str]:
    """Cell lines the overlaps table carries, in the order it carries them."""
    prefix = "cre_avg_signal ("
    return [column[len(prefix):-1] for column in columns if column.startswith(prefix)]


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


def metric_group_mask(columns: list[str]) -> np.ndarray:
    """True where a metric column belongs to the TSS group, False for the CRE group."""
    return np.array([split_metric(column)[0] in TSS_METRIC_SPECS for column in columns])


def composite_weights(columns: list[str]) -> np.ndarray:
    """Per-column weights giving the CRE and TSS groups a fixed share each.

    Each group's weight is spread evenly over its columns, so adding cell lines
    changes the resolution of the CRE side without changing how much it counts.
    """
    is_tss = metric_group_mask(columns)
    weights = np.where(is_tss, TSS_WEIGHT, CRE_WEIGHT).astype(float)
    for group in (is_tss, ~is_tss):
        if group.any():
            weights[group] /= group.sum()
    return weights


def axis_weights(columns: list[str]) -> np.ndarray:
    """Weights over one group's columns for its typing axis, summing to one."""
    weights = np.array([AXIS_WEIGHTS[split_metric(column)[0]] for column in columns], dtype=float)
    total = weights.sum()
    if total == 0:
        raise ValueError(f"AXIS_WEIGHTS leaves every column of {columns} at zero weight")
    return weights / total


METRIC_COLUMNS = metric_columns_for(CLUSTERING_CELLS)


def category_name(cre_score: float, tss_score: float) -> str:
    """Name an element from where it sits on the CRE and TSS axes."""
    if cre_score < CATEGORY_ACTIVE and tss_score < CATEGORY_ACTIVE:
        return "inactive"
    if cre_score >= CATEGORY_STRONG and tss_score >= CATEGORY_STRONG:
        return "enhancer & promoter"
    if tss_score >= CATEGORY_STRONG:
        return "promoter-only"
    if cre_score >= CATEGORY_STRONG:
        return "enhancer-only"
    return "weak promoter" if tss_score > cre_score else "weak enhancer"


# --- TYPING ---

def functional_profile_typing(
    df: pl.DataFrame,
    metric_columns: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score, type and sort elements. Returns the annotated table and the
    matching heatmap matrix, whose rows are in the same order.

    Everything the scoring needs is derived from `metric_columns`, so the
    caller cannot pass labels and weights that disagree with it.
    """
    labels = [metric_label(column) for column in metric_columns]

    # --- 1. Data Preparation & Normalization ---
    # StandardScaler rather than a bare z-score: it leaves a zero-variance column
    # at 0 instead of propagating NaN.
    raw = df.select(metric_columns).to_numpy()
    scaled_data = StandardScaler().fit_transform(raw)

    # --- 2. Handle Outliers ---
    # Floor at zero so that below-average behaviour is absent evidence rather
    # than evidence against, and cap the maximum z-score per column so single-
    # column outliers cannot dominate.
    clipped_z = np.clip(scaled_data, 0, CLIP_Z)

    # --- 3. Type Elements on the CRE and TSS Axes ---
    # The axes say what kind of element this is, while CRE_WEIGHT / TSS_WEIGHT say
    # how much each kind counts towards the composite. Keeping the two separate
    # lets the weighting change without redrawing the categories. Within a group
    # the metrics are combined by AXIS_WEIGHTS rather than averaged flat, so the
    # length-insensitive mean activity can be given more of the say.
    #
    # With flat AXIS_WEIGHTS the composite reduces exactly to
    # `CRE_WEIGHT * cre_score + TSS_WEIGHT * tss_score`; it only carries
    # independent information once a group's weights are made uneven.
    is_tss = metric_group_mask(metric_columns)
    scores = {}
    for name, group in (("cre_score", ~is_tss), ("tss_score", is_tss)):
        group_columns = [column for column, keep in zip(metric_columns, group) if keep]
        scores[name] = clipped_z[:, group] @ axis_weights(group_columns)
    composite = (clipped_z * composite_weights(metric_columns)).sum(axis=1)

    cluster_label = [category_name(cre, tss) for cre, tss in zip(scores["cre_score"], scores["tss_score"])]

    # --- 4. Sort Elements for Visualization ---
    # By category (in CATEGORY_ORDER), then by composite score (desc). The source
    # row is the final key so that the many elements tied at a composite of zero
    # keep a reproducible order.
    typed = df.with_columns(
        cre_score=pl.Series(scores["cre_score"]),
        tss_score=pl.Series(scores["tss_score"]),
        composite_z_score=pl.Series(composite),
        cluster_label=pl.Series(cluster_label),
        priority_group=pl.Series([CATEGORY_ORDER.index(label) + 1 for label in cluster_label]),
        source_row=pl.Series(np.arange(df.height, dtype=np.uint32)),
    ).sort(
        ["priority_group", "composite_z_score", "source_row"],
        descending=[False, True, False],
    )

    # Standard unclipped, unmuted z-scores for visualization, reordered to match,
    # so raw magnitudes remain visible on the plot.
    df_heatmap = pl.DataFrame(scaled_data[typed["source_row"].to_numpy()], schema=labels)

    return typed.drop("source_row"), df_heatmap


# --- LOADING ---

def load_element_overlaps() -> pl.DataFrame:
    """Overlap metrics joined to the popularity, length and annotation columns."""
    element_citations = pl.read_parquet(ELEMENT_CITATIONS).select(
        ["element_type", "element_name", *POPULARITY_COLUMNS]
    )
    element_lengths = (
        pl.read_parquet(ELEMENT_POSITIONS)
        .group_by(["element_type", "element_name"])
        .agg(pl.col("length").median().cast(pl.Int64).alias("element_length"))
    )
    promoter_distance = pl.read_parquet(PROMOTER_DISTANCE).select(
        ["element_type", "element_name", "promoter_distance_mean",
         "promoter_distance_median", "is_reference_element"]
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

    overlaps = pl.read_parquet(ELEMENT_OVERLAPS)
    for table in (element_citations, element_lengths, promoter_distance, sequence_diversity):
        overlaps = overlaps.join(
            table, left_on=["type", "name"], right_on=["element_type", "element_name"], how="left"
        )
    return overlaps.with_columns(
        pl.max_horizontal("tss_fwd_avg_signal", "tss_rev_avg_signal").alias("tss_avg_signal")
    )


if __name__ == "__main__":

    # 0. Load pre-processed overlap data
    element_cre_overlap = load_element_overlaps()

    # 1. Typing
    # Every available metric is kept in the frame so the written table carries all
    # cell lines; metric_columns alone decides what the typing actually sees.
    all_metric_columns = metric_columns_for(available_cells(element_cre_overlap.columns))
    missing = [column for column in METRIC_COLUMNS if column not in all_metric_columns]
    if missing:
        raise KeyError(f"metric columns absent from {ELEMENT_OVERLAPS.name}: {missing}")

    print(f"Writing {len(all_metric_columns)} metrics, typing on {len(METRIC_COLUMNS)}:")
    for column, weight in zip(METRIC_COLUMNS, composite_weights(METRIC_COLUMNS)):
        print(f"  {column:44s} weight {weight:.4f}")

    df = (
        element_cre_overlap
        .filter(pl.col("element_length") >= CRE_LENGTH_THRESH)
        .select(ID_COLUMNS + all_metric_columns + POPULARITY_COLUMNS + ANNOTATION_COLUMNS)
        .with_columns(pl.col(all_metric_columns).fill_null(0))
    )
    df_typed, df_heatmap = functional_profile_typing(df, METRIC_COLUMNS)
    df_heatmap.write_csv(OUT_CLUSTER_RAW_HEAT)

    # 2. Identify Cryptic CRE candidates
    # The length filter above already applies, so it is not repeated here.
    df_typed = df_typed.with_columns(
        is_cryptic_cre=(
            (~pl.col("type").is_in(["promoter", "enhancer"])) &
            (pl.col("cluster_label").is_in(STRONG_CATEGORIES))
        )
    )
    df_typed.write_csv(OUT_CLUSTER_RAW)
