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

# Repeated on the heatmap table so the two files join by key, not row position.
HEATMAP_KEY_COLUMNS = ["type", "name"]

# Context, not predicted activity: written out, never scored.
ANNOTATION_COLUMNS = [
    "promoter_distance_mean", "promoter_distance_median", "is_reference_element",
    "sequence_diversity_pct", "n_sequence_variants",
]

# Inputs to `tss_directionality`; written out, never scored.
TSS_STRAND_COLUMNS = ["fraction_tss_fwd_bp", "fraction_tss_rev_bp"]

# --- METRIC REGISTRY ---
# (description, heatmap label). No metric carries a calling threshold: columns are
# standardised one at a time, which absorbs the ~2.2x difference in CREST activity
# scale between cell lines but leaves scoring relative to the cohort, not absolute.
CRE_METRIC_SPECS = {
    "n_cre_midpoints": ("# CRE Midpoints per Feature Instance", "# CRE midpoints"),
    "fraction_cre_bp": ("Fraction base pairs that are CRE", "Fraction CRE bp"),
    "cre_avg_signal": ("Average activity of CRE base pairs", "Mean CRE activity"),
}
# Not resolved per cell line. `08` resolves them by strand and the loader collapses
# each pair: counts by sum, activity by the stronger strand. Deliberate - 113 of the
# 191 elements with any TSS initiate in one direction only, so averaging the strands
# would halve every unidirectional promoter. Direction is kept as
# `tss_directionality`. Note this strand collapse is a separate maximum from the
# within-instance peak `08` already took over the element's TSS base pairs.
TSS_METRIC_SPECS = {
    "n_tss_midpoints": ("# TSS Midpoints per Feature Instance", "# TSS midpoints"),
    "tss_max_signal": ("Peak activity of TSS base pairs", "Peak TSS activity"),
}

# Peak activity within an instance, averaged over the instances that had an overlap
# (`08` uses `.drop_nans().mean()`), so undefined rather than zero where nothing was
# called. Left missing: an imputed zero is a value the conditional distribution never
# takes and would distort the mean and SD these columns are standardised by.
CONDITIONAL_METRICS = {"cre_avg_signal", "tss_max_signal"}

# Fixed share per group, spread evenly over the group's columns, so the composite
# is `CRE_WEIGHT * mean(CRE) + TSS_WEIGHT * mean(TSS)` whatever the cell count.
CRE_WEIGHT = 0.5
TSS_WEIGHT = 0.5

# Relative weights within a group, renormalised: {1, 1, 1} is a plain mean,
# {0, 0, 1} types on mean activity alone. Shapes the categories only.
AXIS_WEIGHTS = {
    "n_cre_midpoints": 1.0,
    "fraction_cre_bp": 1.0,
    "cre_avg_signal": 1.0,
    "n_tss_midpoints": 1.0,
    "tss_max_signal": 1.0,
}

# --- SCOPE ---
# Every cell line in the overlaps table is written out; only these are typed on.
CLUSTERING_CELLS = ["GM12878", "MRC5", "A549", "HEK293T", "K562", "SHSY5Y"]

CRE_LENGTH_THRESH = 100

CLIP_Z = 3.0  # ceiling on any single column's contribution to an axis

METRIC_COLUMNS: list[str]  # resolved below, once the helpers exist

# --- CATEGORIES ---
# Typed by direct cut-offs in the (CRE, TSS) plane rather than clustered: there is
# no cluster structure to find (k-means silhouette peaks at 0.39 for k=2 and falls
# from there), and cut-offs avoid the seed dependence. The two cut-offs band each
# axis none / weak / strong; the nine cells are the categories. Naming all nine
# removes the tie-break the earlier six-label scheme needed. Still cohort-dependent,
# since the axes are z-scores over the elements in the table.
CATEGORY_ACTIVE = 0.35  # below this an axis carries no evidence
CATEGORY_STRONG = 1.0   # at or above this an axis carries strong evidence

# Keyed by (CRE band, TSS band). This dict's order is the display order and the
# order `priority_group` follows - reorder these lines to rearrange the figure.
# STRONG_CATEGORIES is a membership test, so it is unaffected.
CATEGORY_NAMES = {
    ("strong", "strong"): "enhancer & promoter",
    ("weak", "strong"): "promoter, weak enhancer",
    ("none", "strong"): "promoter-only",
    ("strong", "weak"): "enhancer, weak promoter",
    ("weak", "weak"): "weak enhancer & promoter",
    ("strong", "none"): "enhancer-only",
    ("none", "weak"): "weak promoter",
    ("weak", "none"): "weak enhancer",
    ("none", "none"): "inactive",
}
CATEGORY_ORDER = list(CATEGORY_NAMES.values())
STRONG_CATEGORIES = [name for bands, name in CATEGORY_NAMES.items() if "strong" in bands]

# Row order within a category, as (column, descending).
SORT_WITHIN_CATEGORY = ("cre_block_size_bp", True)  # ("composite_z_score", True)


# --- METRIC COLUMN HELPERS ---

def metric_columns_for(cells: list[str]) -> list[str]:
    """Metric-major column list, so the heatmap keeps related columns adjacent."""
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
    """Short heatmap column label, cell-line suffix and all."""
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
    """Column z-scores over the observed entries only; missing stays missing.

    A zero-variance column is left at zero rather than yielding NaN; a column with
    nothing observed is an error rather than a silently empty one.
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
    """Per-column weights giving the CRE and TSS groups a fixed share each."""
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


# --- DESCRIPTORS ---
# Reported alongside the scores, never typed on: each describes what kind of
# element this is, while the categories are about how much.

def cre_block_size(metric_columns: list[str]) -> pl.Expr:
    """Mean length in bp of one CRE block inside the element, pooled over cell lines.

    CRE architecture - a few long blocks, or many short ones. Reported rather than
    typed on because the raw midpoints-against-coverage contrast it derives from
    tracks element length at rho 0.88; this ratio tracks it at +0.08.

    Capped at the element's own length: coverage is clipped to the element while a
    CRE is counted only where its midpoint falls inside, so an element sitting
    inside a larger CRE has coverage but no midpoint and the bare ratio runs away.
    Undefined where no cell line calls a CRE anywhere in the element.
    """
    total_cres = pl.sum_horizontal(columns_of("n_cre_midpoints", metric_columns))
    total_bp = pl.sum_horizontal(columns_of("fraction_cre_bp", metric_columns)) * pl.col("element_length")
    block_size = pl.min_horizontal(total_bp / total_cres, pl.col("element_length"))
    return pl.when(total_cres > 0).then(block_size).otherwise(None)


def tss_directionality() -> pl.Expr:
    """Strand bias of initiation: +1 sense, -1 antisense, 0 bidirectional.

    `(fwd - rev) / (fwd + rev)` over TSS coverage, the quantity the strand collapse
    in the loader discards. Coverage rather than counts or activity because all
    three agree strongly and coverage is defined for the most elements.
    Preferred over a `max / min` ratio, which is the same magnitude rescaled but is
    infinite for the 97 of 191 elements initiating on one strand only and drops the
    sign - and the sign is what separates promoters (median +0.99) from CDS (-0.51).
    Undefined where neither strand has any TSS coverage.
    """
    fwd, rev = (pl.col(column) for column in TSS_STRAND_COLUMNS)
    total = fwd + rev
    return pl.when(total > 0).then((fwd - rev) / total).otherwise(None)


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
    """Score, type and sort elements.

    Returns the annotated table and the row-aligned heatmap matrix. Everything the
    scoring needs is derived from `metric_columns`, so the caller cannot pass
    labels and weights that disagree with it.
    """
    labels = [metric_label(column) for column in metric_columns]

    # --- 1. Data Preparation & Normalization ---
    raw = df.select(pl.col(metric_columns).cast(pl.Float64)).to_numpy()
    scaled_data = standardise(raw)

    # --- 2. Handle Outliers ---
    # Floor at zero so below-average is absent evidence rather than evidence
    # against, and cap so no single column dominates. A missing conditional value
    # contributes nothing, which is the same floor.
    clipped_z = np.nan_to_num(np.clip(scaled_data, 0, CLIP_Z), nan=0.0)

    # --- 3. Type Elements on the CRE and TSS Axes ---
    # The axes say what kind of element this is; CRE_WEIGHT / TSS_WEIGHT say how
    # much each kind counts. Under flat AXIS_WEIGHTS the composite reduces exactly
    # to `CRE_WEIGHT * cre_score + TSS_WEIGHT * tss_score`.
    is_tss = metric_group_mask(metric_columns)
    scores = {}
    for name, group in (("cre_score", ~is_tss), ("tss_score", is_tss)):
        group_columns = [column for column, keep in zip(metric_columns, group) if keep]
        scores[name] = clipped_z[:, group] @ axis_weights(group_columns)
    composite = (clipped_z * composite_weights(metric_columns)).sum(axis=1)

    cluster_label = [category_name(cre, tss) for cre, tss in zip(scores["cre_score"], scores["tss_score"])]

    # --- 4. Sort Elements for Visualization ---
    # Source row is the final key so elements tied at zero keep a reproducible order.
    sort_column, sort_descending = SORT_WITHIN_CATEGORY
    typed = df.with_columns(
        cre_score=pl.Series(scores["cre_score"]),
        tss_score=pl.Series(scores["tss_score"]),
        composite_z_score=pl.Series(composite),
        cre_block_size_bp=cre_block_size(metric_columns),
        tss_directionality=tss_directionality(),
        cluster_label=pl.Series(cluster_label),
        priority_group=pl.Series([CATEGORY_ORDER.index(label) + 1 for label in cluster_label]),
        source_row=pl.Series(np.arange(df.height, dtype=np.uint32)),
    ).sort(
        ["priority_group", sort_column, "source_row"],
        descending=[False, sort_descending, False],
        nulls_last=True,
    )
    order = typed["source_row"].to_numpy()

    # Unclipped z-scores, so raw magnitudes stay visible. Conditional columns are
    # empty where nothing was called: draw those blank (matplotlib `cmap.set_bad`)
    # rather than as a strong negative.
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
    # Divergence from the element's MSA consensus, in percent: the values span four
    # orders of magnitude and percent is what the figure quotes.
    sequence_diversity = pl.read_parquet(ELEMENT_DIVERGENCE).select([
        "element_type",
        "element_name",
        (100.0 * (1.0 - pl.col("avg_identity"))).alias("sequence_diversity_pct"),
        pl.col("n_instances_unique").alias("n_sequence_variants"),
    ])

    overlaps = pl.read_parquet(ELEMENT_OVERLAPS)
    absent = [column for column in TSS_STRAND_COLUMNS if column not in overlaps.columns]
    if absent:
        raise KeyError(
            f"{ELEMENT_OVERLAPS.name} lacks the strand-resolved TSS coverage columns "
            f"{absent} that `tss_directionality` is built from"
        )

    for table in (element_citations, element_lengths, promoter_distance, sequence_diversity):
        overlaps = overlaps.join(
            table, left_on=["type", "name"], right_on=["element_type", "element_name"], how="left"
        )
    return overlaps.with_columns(
        pl.max_horizontal("tss_fwd_max_signal", "tss_rev_max_signal").alias("tss_max_signal")
    )


if __name__ == "__main__":

    # 0. Load pre-processed overlap data
    element_cre_overlap = load_element_overlaps()

    # 1. Typing
    # All metrics are kept so the written table carries every cell line;
    # METRIC_COLUMNS alone decides what the typing sees.
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
        .select(ID_COLUMNS + all_metric_columns + POPULARITY_COLUMNS
                + ANNOTATION_COLUMNS + TSS_STRAND_COLUMNS)
        # Count and coverage metrics only: zero is the right reading for a null
        # there. Conditional columns keep their nulls, so the written table
        # distinguishes "nothing called" from "called, zero activity".
        .with_columns(pl.col([c for c in all_metric_columns if not is_conditional(c)]).fill_null(0))
    )
    df_typed, df_heatmap = functional_profile_typing(df, METRIC_COLUMNS)
    df_heatmap.write_csv(OUT_CLUSTER_RAW_HEAT)

    # 2. Identify Cryptic CRE candidates
    df_typed = df_typed.with_columns(
        is_cryptic_cre=(
            (~pl.col("type").is_in(["promoter", "enhancer"])) &
            (pl.col("cluster_label").is_in(STRONG_CATEGORIES))
        )
    )
    df_typed.write_csv(OUT_CLUSTER_RAW)
