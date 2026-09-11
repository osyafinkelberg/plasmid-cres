from pathlib import Path

import numpy as np
import polars as pl

# --- CONFIGURATION ---
PROJECT_DIR = Path().cwd().parent
ADDGENE_DIR = PROJECT_DIR / "data/addgene"
ALIGN_DIR = PROJECT_DIR / "data/addgene_alignments"

ELEMENT_POSITIONS = ADDGENE_DIR / "mammalian_plasmids_elements.parquet"
ELEMENT_OVERLAPS = ADDGENE_DIR / "mammalian_plasmids_element_cre_overlaps.parquet"
ELEMENT_CITATIONS = ADDGENE_DIR / "citations_addgene_elements.parquet"
PROMOTER_DISTANCE = ADDGENE_DIR / "mammalian_plasmids_element_promoter_distance.parquet"
ELEMENT_DIVERGENCE = ALIGN_DIR / "element_average_divergence.parquet"

OUT_CLUSTER_RAW = ADDGENE_DIR / "element_cre_overlap_clustering.csv"
OUT_CLUSTER_RAW_HEAT = ADDGENE_DIR / "element_cre_overlap_clustering_heatmap.csv"

ID_COLUMNS = ["type", "name", "element_length"]
POPULARITY_COLUMNS = ["n_plasmids", "n_citations"]

# Repeated on the heatmap table so it can be joined to the clustering table by
# key rather than by row position: a heatmap file that is stale with respect to
# its clustering file is otherwise indistinguishable from a fresh one, and
# mis-annotates every row silently.
HEATMAP_KEY_COLUMNS = ["type", "name"]

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
# No metric carries a calling threshold. A column contributes on its
# standardised value alone, and because each column is standardised separately
# the ~2.2x difference in CREST activity scale between cell lines (GM12878 1.05
# vs SHSY5Y 2.35 at FDR 0.01) is absorbed by the z-score. What that trades away
# is an absolute floor: an element scores on a column for being active relative
# to the cohort, whether or not it would be called a CRE there.
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

# Metrics that are conditional on an overlap existing. `08_element_cre_overlap.py`
# aggregates these with `.drop_nans().mean()`, so they are the mean activity over
# the instances that had a CRE or TSS, and are simply undefined for an element
# that has neither anywhere. The remaining metrics count or measure coverage over
# every instance, so zero is a value they genuinely take.
#
# The distinction matters because these columns are standardised. Filling an
# undefined entry with zero puts it at a value the conditional distribution never
# takes - in SHSY5Y the mean activity of a called CRE is 2.19 with an SD of 0.62,
# so an imputed zero sits 3.5 SD below the population and nearly doubles the SD
# the column is divided by. The entries are therefore left missing and contribute
# nothing, which is the same treatment an element below the column mean already
# gets from the clip at zero.
CONDITIONAL_METRICS = {"cre_avg_signal", "tss_avg_signal"}

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

# --- SCOPE ---
# Cell lines the categories are built from. The overlaps table also carries
# Jurkat and SiHa; every cell line it carries is written to OUT_CLUSTER_RAW, but
# only these shape the axes and the heatmap.
CLUSTERING_CELLS = ["GM12878", "MRC5", "A549", "HEK293T", "K562", "SHSY5Y"]

CRE_LENGTH_THRESH = 100

CLIP_Z = 3.0  # ceiling on any single column's contribution to an axis

# Resolved at the bottom of the module, once the helpers below are defined. The
# heatmap's own column labels are its header row, so nothing else needs exporting.
METRIC_COLUMNS: list[str]

# --- CATEGORIES ---
# Elements are typed in the plane of the CRE and TSS halves of the composite
# score - the two quantities the composite is already built from - rather than
# clustered in the 20 metric columns. There is no cluster structure there to
# find: over the active elements k-means silhouette peaks at 0.39 for k=2 and
# falls monotonically from there, so any choice of k would impose an arbitrary
# partition and reintroduce a seed dependence that direct cut-offs avoid.
#
# The two cut-offs divide each axis into none / weak / strong, and the nine
# resulting cells are the categories. Naming all nine rather than collapsing them
# into six removes the one arbitrary step the earlier scheme had: a
# `tss_score > cre_score` tie-break that split the elements weak on both axes
# between "weak promoter" and "weak enhancer" by comparing an 18-column mean
# against a 2-column mean. A category now follows only from which band each of an
# element's two scores falls in. It does still depend on the cohort, because the
# axes are built from z-scores taken over the elements present in the table.
CATEGORY_ACTIVE = 0.35  # below this an axis carries no evidence
CATEGORY_STRONG = 1.0   # at or above this an axis carries strong evidence

# Keyed by (CRE band, TSS band), in display order, which is the order
# `priority_group` numbers follow: strongest single band first, then total
# evidence, with TSS-leaning cells ahead of their CRE-leaning mirrors.
CATEGORY_NAMES = {
    ("strong", "strong"): "enhancer & promoter",
    ("weak", "strong"): "promoter, weak enhancer",
    ("strong", "weak"): "enhancer, weak promoter",
    ("none", "strong"): "promoter-only",
    ("strong", "none"): "enhancer-only",
    ("weak", "weak"): "weak enhancer & promoter",
    ("none", "weak"): "weak promoter",
    ("weak", "none"): "weak enhancer",
    ("none", "none"): "inactive",
}
CATEGORY_ORDER = list(CATEGORY_NAMES.values())
# Categories with strong evidence on at least one axis: enough to call a candidate.
STRONG_CATEGORIES = [name for bands, name in CATEGORY_NAMES.items() if "strong" in bands]

# Row order within a category, as (column, descending). `composite_z_score` puts
# the strongest elements at the top of each band. `cre_block_size_bp` instead
# lays the architecture gradient out down the figure - a few long CRE blocks at
# one end, many short ones at the other - which is the structure that otherwise
# reads as an unresolved split inside the enhancer-like categories.
SORT_WITHIN_CATEGORY = ("cre_block_size_bp", True)  # ("composite_z_score", True)


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


def is_conditional(column: str) -> bool:
    """True where a column is only defined for elements that have an overlap."""
    return split_metric(column)[0] in CONDITIONAL_METRICS


def columns_of(metric: str, metric_columns: list[str]) -> list[str]:
    """Every column of `metric_columns` that measures the given base metric."""
    return [column for column in metric_columns if split_metric(column)[0] == metric]


def standardise(values: np.ndarray) -> np.ndarray:
    """Column z-scores computed over the observed entries only.

    Missing entries stay missing, so a column that is defined for only some
    elements is centred on the elements that have a value. A column with no
    spread is left at zero rather than yielding NaN, and a column with nothing
    observed at all is an error rather than a silently empty one.
    """
    observed = np.count_nonzero(~np.isnan(values), axis=0)
    if (observed == 0).any():
        raise ValueError("a metric column has no observed values to standardise on")
    centre = np.nanmean(values, axis=0)
    spread = np.nanstd(values, axis=0)
    return (values - centre) / np.where(spread > 0, spread, 1.0)


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


def cre_block_size(metric_columns: list[str]) -> pl.Expr:
    """Mean length in base pairs of one CRE block inside the element.

    Total CRE base pairs per instance over total CREs per instance, pooled across
    the cell lines being typed. This describes an element's CRE architecture - a
    few long blocks, or many short ones - which is a different question from how
    much CRE there is, and is why it is reported rather than typed on. The raw
    contrast it derives from, mean z(# midpoints) against mean z(fraction bp),
    separates the enhancer-like categories into two visible halves but tracks
    element length at rho 0.88, so promoting it to an axis would put length into
    the taxonomy. This ratio removes that dependence, tracking length at +0.08
    while still ordering the categories sensibly (median 295 bp for
    `enhancer & promoter`, 188 bp for `enhancer-only`, 167 bp for `weak enhancer`).

    Capped at the element's own length, because the two terms are counted over
    different spans: coverage is clipped to the element, while a CRE is counted
    only where its midpoint falls inside. An element sitting wholly within a
    larger CRE therefore has coverage but few or no midpoints, and the bare ratio
    runs away - it exceeded the element's own length for 85 of 341 elements
    before capping, and reached 67 kb. At the cap the value means "no smaller
    than this element", which is the most the table can say.

    Undefined where no cell line calls a CRE anywhere in the element.
    """
    total_cres = pl.sum_horizontal(columns_of("n_cre_midpoints", metric_columns))
    total_bp = pl.sum_horizontal(columns_of("fraction_cre_bp", metric_columns)) * pl.col("element_length")
    block_size = pl.min_horizontal(total_bp / total_cres, pl.col("element_length"))
    return pl.when(total_cres > 0).then(block_size).otherwise(None)


def axis_band(score: float) -> str:
    """Which of the three evidence bands a single axis score falls in."""
    if score >= CATEGORY_STRONG:
        return "strong"
    return "weak" if score >= CATEGORY_ACTIVE else "none"


def category_name(cre_score: float, tss_score: float) -> str:
    """Name an element from the band each of its two axis scores falls in."""
    return CATEGORY_NAMES[(axis_band(cre_score), axis_band(tss_score))]


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
    # Missing entries in the conditional columns stay missing here, so that each
    # column is centred and scaled on the elements that actually have a value.
    raw = df.select(pl.col(metric_columns).cast(pl.Float64)).to_numpy()
    scaled_data = standardise(raw)

    # --- 2. Handle Outliers ---
    # Floor at zero so that below-average behaviour is absent evidence rather
    # than evidence against, and cap the maximum z-score per column so single-
    # column outliers cannot dominate. An element with no value on a conditional
    # column contributes nothing, which is the floor applied to absent evidence.
    clipped_z = np.nan_to_num(np.clip(scaled_data, 0, CLIP_Z), nan=0.0)

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
    # By category (in CATEGORY_ORDER), then by SORT_WITHIN_CATEGORY. The source
    # row is the final key so that the many elements tied at a composite of zero
    # keep a reproducible order.
    sort_column, sort_descending = SORT_WITHIN_CATEGORY
    typed = df.with_columns(
        cre_score=pl.Series(scores["cre_score"]),
        tss_score=pl.Series(scores["tss_score"]),
        composite_z_score=pl.Series(composite),
        cre_block_size_bp=cre_block_size(metric_columns),
        cluster_label=pl.Series(cluster_label),
        priority_group=pl.Series([CATEGORY_ORDER.index(label) + 1 for label in cluster_label]),
        source_row=pl.Series(np.arange(df.height, dtype=np.uint32)),
    ).sort(
        ["priority_group", sort_column, "source_row"],
        descending=[False, sort_descending, False],
        nulls_last=True,
    )
    order = typed["source_row"].to_numpy()

    # Standard unclipped z-scores for visualization, reordered to match, so raw
    # magnitudes remain visible on the plot. Conditional columns stay empty where
    # the element has no CRE or TSS, so the figure can draw "none called" as a
    # blank cell (matplotlib `cmap.set_bad`) instead of as a strong negative.
    typed = typed.drop("source_row")
    df_heatmap = pl.concat(
        [typed.select(HEATMAP_KEY_COLUMNS),
         pl.DataFrame(scaled_data[order], schema=labels)],
        how="horizontal",
    )

    return typed, df_heatmap


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
    # cell lines; METRIC_COLUMNS alone decides what the typing actually sees.
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
        # Only the count and coverage metrics: a null there means the element was
        # absent from a join and zero is the right reading. The conditional
        # columns keep their nulls, which is what `standardise` expects, and the
        # written table then distinguishes "no CRE called" from "CRE of zero
        # activity" instead of writing both as 0.
        .with_columns(pl.col([c for c in all_metric_columns if not is_conditional(c)]).fill_null(0))
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
