import itertools

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from adjustText import adjust_text
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist

# `08_element_cre_overlap.py` emits CRE metrics once per CREST cell line, as
# `<metric> (<cell>)`. This is the cell line assumed when a caller does not name
# one; TSS metrics are cell-agnostic and keep their plain names.
DEFAULT_CRE_CELL = "HEK293T"

# Figures are read from slides and printed panels, not zoomed in a notebook, so
# every text element is sized up front. Tune a figure's type here rather than by
# hunting down scattered `fontsize=` arguments.
FONT_SIZES = {
    "title": 22,
    "axis_label": 18,
    "tick": 17,
    "legend": 16,
    "legend_title": 17,
    "row_label": 13,     # per-element ids, one per heatmap row
    "group_label": 15,   # priority-block labels, one per block
    "cbar_tick": 14,
    "cbar_label": 19,   # colourbar titles, read at figure-panel size
    "column_group": 18,  # metric name spanning a block of cell-line columns
    "strip_label": 16,   # names under the row-annotation strips
}

# `functional_profiling_plot` sizing. Height follows the row count and width the
# metric count, so the same call renders a 60-row and a 600-row view legibly.
LABEL_ROWS_THRESHOLD = 120    # above this, rows get block labels instead of ids
PER_ROW_HEIGHT = 0.25         # inches per labelled row - room for its text
DENSE_ROW_HEIGHT = 0.024      # inches per unlabelled row - a visible band
HEIGHT_OVERHEAD = 4.0         # inches of non-heatmap chrome
MAX_FIG_HEIGHT = 32
PER_METRIC_WIDTH = 1.1        # inches per heatmap column
WIDTH_OVERHEAD = 5.5          # inches for row colors, legend strip and margins
MAX_FIG_WIDTH = 24

# Optional per-row annotation strips for `functional_profiling_plot`. These are
# control metrics, not data: what matters is spotting elements whose CRE metrics
# may be untrustworthy, not reading a value off a ramp. So each maps its quantity
# through an explicit ramp to one shared 0-1 concern score and they share a
# single colormap and colourbar - a dark mark means the same thing in any strip,
# and the safe majority stays pale.
CONCERN_CMAP = LinearSegmentedColormap.from_list(
    "control_concern", ["#f0f0f0", "#fee391", "#fe9929", "#993404"]
)

LENGTH_SHORT_FLAG = 200    # bp; at or below, a short element is fully flagged
LENGTH_SHORT_CLEAR = 650   # bp; at or above, shortness is no longer a concern
LENGTH_LONG_CLEAR = 2000   # bp; at or below, length is no longer a concern
LENGTH_LONG_FLAG = 4000    # bp; at or above, a long element is fully flagged
PROMOTER_CLEAR = 300       # bp; a promoter beyond this is unlikely to explain the signal
DIVERSITY_CLEAR = 5.0      # % divergence tolerated before instances disagree meaningfully
DIVERSITY_FLAG = 25.0      # % divergence at which one value cannot represent the group


def _concern_ramp(values: np.ndarray, clear: float, flag: float) -> np.ndarray:
    """Clipped linear 0-1 ramp: 0 at `clear`, 1 at `flag`, either direction."""
    return np.clip((values - clear) / (flag - clear), 0.0, 1.0)


def short_length_concern(values: np.ndarray) -> np.ndarray:
    """A short element's predicted CREs may really belong to the flanking plasmid context."""
    return _concern_ramp(values, LENGTH_SHORT_CLEAR, LENGTH_SHORT_FLAG)


def long_length_concern(values: np.ndarray) -> np.ndarray:
    """A very long element inflates per-element counts, such as the number of CRE midpoints."""
    return _concern_ramp(values, LENGTH_LONG_CLEAR, LENGTH_LONG_FLAG)


def promoter_distance_concern(values: np.ndarray) -> np.ndarray:
    """Only proximity is a risk: a promoter close by may explain the activity.

    A large distance carries no risk, and neither does a missing value - it means
    no annotated promoter shares the plasmid at all, so nothing can be confounding.
    """
    concern = _concern_ramp(values, PROMOTER_CLEAR, 0.0)
    return np.where(np.isnan(values), 0.0, concern)


def sequence_diversity_concern(values: np.ndarray) -> np.ndarray:
    """Only high divergence is a risk: one value cannot represent instances that differ."""
    return _concern_ramp(values, DIVERSITY_CLEAR, DIVERSITY_FLAG)


# Keyed by annotation name rather than by column, because being too short and
# being too long are separate concerns with different causes and so get a strip
# each: on one shared colour scale a single two-sided ramp could not say which
# end a dark mark came from.
ROW_ANNOTATIONS = {
    "length_short": {
        "column": "element_length",
        "strip": "Short",
        "rule": f"length < {LENGTH_SHORT_CLEAR} bp",
        "concern": short_length_concern,
    },
    "length_long": {
        "column": "element_length",
        "strip": "Long",
        "rule": f"length > {LENGTH_LONG_CLEAR // 1000} kb",
        "concern": long_length_concern,
    },
    "promoter_distance": {
        "column": "promoter_distance_mean",
        "strip": "Promoter dist.",
        "rule": f"< {PROMOTER_CLEAR} bp",
        "concern": promoter_distance_concern,
    },
    "promoter_distance_median": {
        "column": "promoter_distance_median",
        "strip": "Promoter dist.",
        "rule": f"< {PROMOTER_CLEAR} bp",
        "concern": promoter_distance_concern,
    },
    "sequence_diversity": {
        "column": "sequence_diversity_pct",
        "strip": "Diversity",
        "rule": f"> {DIVERSITY_CLEAR:g}% divergence",
        "concern": sequence_diversity_concern,
    },
}
ANNOTATION_BAR_HEIGHT = 1.7   # inches, so the concern bar keeps one physical size
ANNOTATION_BAR_GAP = 1.0      # inches between it and the z-score bar above

ELEMENT_TYPE_PRIORITIES = {
    "CDS": 29, "promoter": 28, "rep_origin": 27, "oriT": 26,
    "RBS": 25, "terminator": 24, "polyA_signal": 23, "enhancer": 22, "regulatory": 21,
    "sig_peptide": 20, "tRNA": 19, "ncRNA": 18, "misc_RNA": 17, "mobile_element": 16,
    "LTR": 15, "repeat_region": 14,
    "exon": 13, "intron": 12, "gene": 11,
    "3'UTR": 10, "5'UTR": 9, "primer_bind": 8, "protein_bind": 7, "misc_signal": 6, "misc_recomb": 5,
    "misc_feature": 4, "gap": 3, "putative_orf": 2, "putative_noncoding": 1, "backbone_spacer": 0,
}

GENOMIC_COLORS = {
    "putative_orf": "#E0E0E0",  # Light Grey
    "putative_noncoding": "#E0E0E0",  # Light Grey
    "backbone_spacer": "#E0E0E0",  # Light Grey
    "backbone": "#E0E0E0",      # Light Grey

    "CDS": "#27ae60",           # Emerald Green
    "exon": "#2ecc71",          # Lighter Green
    "intron": "#55efc4",        # Muted Grey-Green
    "promoter": "#e74c3c",      # Red
    "enhancer": "#e67e22",      # Orange
    "regulatory": "#d35400",    # Darker Orange
    "rep_origin": "#f1c40f",    # Sunflower Yellow
    "oriT": "#f39c12",          # Orange-Yellow
    "terminator": "#3498db",    # Bright Blue
    "polyA_signal": "#2980b9",  # Deep Blue
    "3'UTR": "#9b59b6",         # Amethyst Purple
    "5'UTR": "#8e44ad",         # Wisteria Purple
    "RBS": "#fd79a8",           # Pink
    "primer_bind": "#eda6a8",
    "protein_bind": "#636e72",  # Grey
    "mobile_element": "#CF8263",# Strong Red
    "repeat_region": "#b2bec3", # Silver
    "gap": "#ffffff",           # White (Actual missing data/gap)
    "LTR": "#D627F5",
    "misc_feature": "#2A27F5",  # Turquoise
    "misc_RNA": "#6563CF",      # Darker Teal
    "ncRNA": "#16a085",
    "tRNA": "#16a085",
    "misc_recomb": "#7f8c8d",
    "misc_signal": "#2A27F5",
    "gene": "#55efc4",
    "sig_peptide": "#fab1a0"
}


def plot_regulatory_correlation(
    df: pl.DataFrame,
    genomic_colors: dict = GENOMIC_COLORS,
    cre_cell: str = DEFAULT_CRE_CELL,
    x_col: str | None = None,
    y_col: str = "n_tss_midpoints",
    x_label: str = "# CRE Midpoints per Feature Instance",
    y_label: str = "# TSS Midpoints per Feature Instance",
    x_thresh: float = 0.25,
    y_thresh: float = 0.25,
    density_radius: float = 0.05,  # radius in normalized space (0.05 = 5% of axis size)
    max_neighbors: int = 5,  # skip label if more than this many points are in the radius
    figsize: tuple = (11, 9)
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a publication-quality scatter correlation between CRE and TSS metrics.
    Filters elements passing a threshold for at least one metric and handles label layout dynamically.
    Skips labels in high-density regions to reduce visual clutter.

    `x_col` defaults to the CRE midpoint count for `cre_cell`; pass it explicitly
    to plot any other metric, including another cell line's.
    """
    if x_col is None:
        x_col = f"n_cre_midpoints ({cre_cell})"
    # set publication theme
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    # filter data: pass threshold for at least one metric
    df_filtered = df.filter((pl.col(x_col) >= x_thresh) | (pl.col(y_col) >= y_thresh))

    # convert to pandas for seamless iteration and plotting with matplotlib
    pdf = df_filtered.to_pandas()

    # map feature_count to discrete marker sizes using log-spaced bins
    counts = pdf["n_citations"].to_numpy()
    bin_edges = np.array([11, 101, 1001, 10001])
    sizes_map = np.array([35, 85, 170, 310, 520])  # Scaled marker sizes (s)
    pdf["marker_size"] = sizes_map[np.digitize(counts, bin_edges)]

    # Figure
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", zorder=0)

    # plotting unique types to preserve clean legend mapping
    unique_types = sorted(pdf["type"].unique())

    for t_name in unique_types:
        sub_df = pdf[pdf["type"] == t_name]
        color = genomic_colors.get(t_name, "#94a3b8")  # fallback to a neutral slate gray if missing
        ax.scatter(
            sub_df[x_col],
            sub_df[y_col],
            label=t_name,
            color=color,
            s=sub_df["marker_size"],  # dynamically sized using the binned mapping
            alpha=0.85,
            zorder=3
        )

    # draw metric thresholds
    ax.axvline(x=x_thresh, color="#64748b", linestyle="--", lw=1.2, alpha=0.8, zorder=2)
    ax.axhline(y=y_thresh, color="#64748b", linestyle="--", lw=1.2, alpha=0.8, zorder=2)
    # ax.text(x_thresh * 1.1, ax.get_ylim()[1] * 0.95, f'Threshold ({y_thresh})', 
    #         color="#64748b", fontsize=9, fontstyle='italic')
    # ax.text(ax.get_xlim()[1] * 0.95, y_thresh * 1.1, f'Threshold ({x_thresh})', 
    #         color="#64748b", fontsize=9, fontstyle='italic')

    # build dynamic text labels (eliminate overlaps in dense regions)
    x_vals = pdf[x_col].to_numpy()
    y_vals = pdf[y_col].to_numpy()

    # normalize coordinates to a [0, 1] range so the distance threshold behaves 
    # identically across different axis scales/ranges.
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_denom = (x_max - x_min) if x_max != x_min else 1.0
    y_denom = (y_max - y_min) if y_max != y_min else 1.0
    x_norm = (x_vals - x_min) / x_denom
    y_norm = (y_vals - y_min) / y_denom

    texts = []
    for i, row in pdf.iterrows():
        # calculate euclidean distances to all other points in normalized space
        dists = np.sqrt((x_norm - x_norm[i])**2 + (y_norm - y_norm[i])**2)
        # count how many neighbors fall within your density radius (subtract 1 for the point itself)
        local_density = np.sum(dists < density_radius) - 1
        # only label if the point is in a low-to-moderate density region
        if local_density <= max_neighbors:
            texts.append(
                ax.text(
                    row[x_col],
                    row[y_col],
                    row["name"],
                    fontsize=8.5,
                    fontweight="medium",
                    color="#0f172a"
                )
            )

    # force layout adjust_text engine
    adjust_text(
        texts,
        ax=ax,
        arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": 0.6, "alpha": 0.7},
        expand_points=(1.6, 1.6),
        force_points=(0.2, 0.4),
        zorder=4
    )

    # set clean axis padding
    ax.set_xlim(left=-max(pdf[x_col])*0.03)
    ax.set_ylim(bottom=-max(pdf[y_col])*0.03)
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold", labelpad=10)

    # clean Despine
    sns.despine(ax=ax, offset=5, trim=False)

    # primary legend: genomic feature type
    leg1 = ax.legend(
        title="Genomic Feature Type", 
        title_fontproperties={'weight': 'bold', 'size': 10},
        bbox_to_anchor=(.93, 1), 
        loc='upper left', 
        frameon=True,
        facecolor='#f8fafc',
        edgecolor='#e2e8f0'
    )

    # explicitly set a uniform dot size (e.g., 75) for all entries in the type legend
    for handle in leg1.legend_handles:
        handle.set_sizes([75])

    # secondary legend: citation / abundance size bins; # NOTE: should match the above bins
    size_labels = ["≤ 10", "11 - 100", "101 - 1k", "1k - 10k", "> 10k"]
    size_handles = [
        ax.scatter([], [], s=sz, color="#64748b", alpha=0.6, linestyle='None') 
        for sz in sizes_map
    ]

    _ = ax.legend(
        size_handles, size_labels,
        title="Number of Citations",
        title_fontproperties={'weight': 'bold', 'size': 10},
        bbox_to_anchor=(.93, 0.45),
        loc='upper left', 
        frameon=True,
        facecolor='#f8fafc',
        edgecolor='#e2e8f0'
    )

    # re-insert the first legend onto the axes canvas (matplotlib overrides leg1 otherwise)
    ax.add_artist(leg1)
    plt.tight_layout()
    return fig, ax


def _split_column_label(label: str) -> tuple[str, str]:
    """Split a heatmap column label into its metric and its cell line.

    `"# CRE midpoints [K562]"` becomes `("# CRE midpoints", "K562")`. A label
    with no bracketed suffix is its own group and has no member - in practice
    the cell-agnostic TSS metrics, which stand alone.
    """
    metric, separator, cell = label.partition(" [")
    return (metric, cell[:-1]) if separator else (label, "")


def _column_groups(labels: list[str]) -> tuple[list[tuple[str, str]], list[list]]:
    """Parsed labels, plus contiguous runs of columns sharing one metric.

    Runs are contiguous because the caller supplies columns in metric-major
    order; nothing is reordered here.
    """
    parsed = [_split_column_label(label) for label in labels]
    runs = []
    for i, (metric, cell) in enumerate(parsed):
        if runs and runs[-1][0] == metric:
            runs[-1][1].append(cell)
            runs[-1][3] = i + 1
        else:
            runs.append([metric, [cell], i, i + 1])
    return parsed, runs


def _draw_column_group_tier(ax: plt.Axes, runs: list[list], fontsize: int) -> None:
    """Draw a second x-axis tier: a bracket and metric name under each run.

    The tier is placed just below whatever vertical space the tick labels
    actually occupy, which is only knowable after a draw, so one is forced here.
    Offsets are in points converted to axes fractions, so the tier sits the same
    distance below the labels whether the figure is 9 or 32 inches tall.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    labels_bottom = min(t.get_window_extent(renderer).y0 for t in ax.get_xticklabels())
    y_labels = ax.transAxes.inverted().transform((0, labels_bottom))[1]
    per_point = (fig.dpi / 72) / ax.get_window_extent(renderer).height
    line_y = y_labels - 10 * per_point
    text_y = line_y - 7 * per_point

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for metric, cells, start, stop in runs:
        if not any(cells):
            continue  # a lone metric already names itself in the tick label
        ax.plot(
            [start + 0.12, stop - 0.12], [line_y, line_y], transform=trans,
            color="#333333", lw=1.6, clip_on=False, solid_capstyle="butt",
            scalex=False, scaley=False,
        )
        ax.text(
            (start + stop) / 2, text_y, metric, transform=trans,
            ha="center", va="top", fontsize=fontsize, fontweight="bold", clip_on=False,
        )


def _annotation_colors(values: np.ndarray, spec: dict) -> list:
    """Row colours for one control metric, on the shared concern scale."""
    concern = spec["concern"](np.asarray(values, dtype=float))
    return [CONCERN_CMAP(level) for level in concern]


def _concern_colorbar(fig, rect: list, specs: list[dict], fontsize: dict) -> None:
    """One colourbar for every control strip, with the flagging rules beneath it.

    A single shared bar is the point of the design: the strips do not carry
    independent units, so three separate bars would invite reading a value where
    only the level matters. The rules are spelled out underneath so the figure
    still says what earned a mark.
    """
    cax = fig.add_axes(rect)
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CONCERN_CMAP), cax=cax)
    bar.set_ticks([0.0, 0.5, 1.0])
    bar.set_ticklabels(["none", "some", "high"])
    bar.set_label("Control-metric concern", fontsize=fontsize["cbar_label"])
    cax.tick_params(labelsize=fontsize["cbar_tick"])

    # Placed well below the bar: a vertical colourbar puts its title on the right,
    # centred, and that title is longer than the bar itself, so anything closer
    # than this collides with it.
    rules = "\n".join(f"{spec['strip']}:  {spec['rule']}" for spec in specs)
    cax.text(
        0.0, -0.42, rules, transform=cax.transAxes,
        ha="left", va="top", fontsize=fontsize["cbar_tick"], linespacing=1.6,
    )


def functional_profiling_plot(
    df_clustered: pl.DataFrame,
    heatmap_df: pl.DataFrame,
    title: str | None = None,
    show_ylabels: bool | None = None,
    annotations: list[str] | None = None,
) -> sns.matrix.ClusterGrid:
    """Ordered heatmap of the functional-profile clustering.

    The figure sizes itself from the data, so a 60-row cryptic-CRE view and a
    600-row full view are both legible from the same call. Neither axis is
    clustered here: rows keep the priority ranking they were sorted into and
    columns keep their metric-major order, which already places the same metric
    for different cell lines side by side.

    Parameters
    ----------
    title : str, optional
        Names the subset being shown; appended to the standing heading.
    show_ylabels : bool, optional
        Force per-element row labels on or off. By default they appear only when
        there are few enough rows to read them; otherwise each priority block
        gets a single label, which is the only structure legible at that density.
    annotations : list of str, optional
        Names of `ROW_ANNOTATIONS` entries to draw as extra row-colour strips
        beside the element-type strip. All share one concern scale and one
        colourbar. Omit for the plain heatmap.
    """
    df_clustered, heatmap_df = df_clustered.to_pandas(), heatmap_df.to_pandas()
    n_rows = len(df_clustered)

    # --- 1. Render the Ordered Heatmap ---
    sns.set_theme(style="white", context="paper", font_scale=1.4)

    # Map row colors. Extra annotation strips sit between the type strip and the
    # heatmap, so the reader meets them on the way in from the element labels.
    annotations = annotations or []
    missing = [name for name in annotations if ROW_ANNOTATIONS[name]["column"] not in df_clustered.columns]
    if missing:
        raise KeyError(f"annotation columns absent from the clustering table: {missing}")

    row_colors = pd.DataFrame(
        {"Type": df_clustered['type'].map(lambda x: GENOMIC_COLORS.get(x, '#cccccc'))},
        index=df_clustered.index,
    )
    for name in annotations:
        spec = ROW_ANNOTATIONS[name]
        row_colors[spec["strip"]] = pd.Series(
            _annotation_colors(df_clustered[spec["column"]].to_numpy(), spec),
            index=df_clustered.index, dtype=object,
        )

    # Height follows the row count in both regimes. Labelled rows need room for
    # their text; unlabelled rows only need to stay a visible band, but they do
    # need height - a fixed figure crushes several hundred rows into nothing.
    if show_ylabels is None:
        show_ylabels = n_rows <= LABEL_ROWS_THRESHOLD
    row_height = PER_ROW_HEIGHT if show_ylabels else DENSE_ROW_HEIGHT
    fig_height = min(MAX_FIG_HEIGHT, max(9, n_rows * row_height + HEIGHT_OVERHEAD))
    fig_width = min(MAX_FIG_WIDTH, max(11, heatmap_df.shape[1] * PER_METRIC_WIDTH + WIDTH_OVERHEAD))

    # Clustering is disabled on both axes, so no dendrogram is drawn. The column
    # strip is kept just deep enough to carry the heading; the row strip holds
    # the element-type legend.
    cg = sns.clustermap(
        heatmap_df,
        row_cluster=False,       # KEEP rows sorted by our Priority ranking
        col_cluster=False,       # KEEP metric-major column order from the caller
        row_colors=row_colors,
        cmap="RdBu_r",
        vmin=-2.5,
        vmax=7.5,
        center=0,
        figsize=(fig_width, fig_height),
        dendrogram_ratio=(0.20, 0.06),
        cbar_kws={'label': 'Relative Values\n(Z-Score)'},
        colors_ratio=0.03,  # seaborn already scales this by the number of strips
    )
    cg.ax_row_colors.set_xticklabels(
        cg.ax_row_colors.get_xticklabels(), fontsize=FONT_SIZES["strip_label"], rotation=45, ha='right',
    )

    # Draw separators between Priority Groups
    ax_heat = cg.ax_heatmap
    groups = df_clustered['priority_group'].to_numpy()
    bounds = [0] + [i for i in range(1, n_rows) if groups[i] != groups[i - 1]] + [n_rows]
    for boundary in bounds[1:-1]:
        ax_heat.axhline(y=boundary, color='black', linewidth=1.8)

    # Formatting
    # Columns arrive grouped by metric, so the metric name is factored out into a
    # second tier below and each tick only has to name its cell line.
    parsed_columns, column_runs = _column_groups(list(heatmap_df.columns))
    ax_heat.set_xticklabels(
        [cell or metric for metric, cell in parsed_columns],
        rotation=45, ha='right', fontsize=FONT_SIZES["tick"],
    )

    # --- Y-tick labels: element IDs when they fit, priority blocks otherwise ---
    if show_ylabels:
        element_ids = [
            f"{row['type']}, {row['name']}"
            for _, row in df_clustered.iterrows()
        ]
        ax_heat.yaxis.set_ticks(np.arange(0.5, len(element_ids), 1))
        ax_heat.set_yticklabels(
            element_ids,
            rotation=0,
            fontsize=FONT_SIZES["row_label"],
            fontweight='bold',
            va='center',
        )
    else:
        ax_heat.yaxis.set_ticks([(a + b) / 2 for a, b in itertools.pairwise(bounds)])
        ax_heat.set_yticklabels(
            [f"P{groups[a]} - n={b - a}" for a, b in itertools.pairwise(bounds)],
            rotation=0,
            fontsize=FONT_SIZES["group_label"],
            fontweight='bold',
            va='center',
        )

    ax_heat.set_ylabel(f"N = {n_rows}", fontsize=FONT_SIZES["axis_label"], fontweight='bold')

    heading = "Functional Profiling of Plasmid Elements"
    cg.ax_col_dendrogram.set_title(
        f"{heading} - {title}" if title else heading,
        fontsize=FONT_SIZES["title"], fontweight='bold', pad=20,
    )

    # Add Element Type Legend
    legend_patches = [
        mpatches.Patch(color=color, label=el_type)
        for el_type, color in GENOMIC_COLORS.items() if el_type in df_clustered['type'].unique()
    ]
    cg.ax_row_dendrogram.legend(
        handles=legend_patches, title="Element Type",
        title_fontproperties={'weight': 'bold', 'size': FONT_SIZES["legend_title"]},
        fontsize=FONT_SIZES["legend"],
        loc="lower left", bbox_to_anchor=(-0.4, -0.2), frameon=False
    )
    cg.ax_cbar.set_position([0.02, 0.8, 0.03, 0.15])
    cg.ax_cbar.set_ylabel('Relative Values\n(Z-Score)', fontsize=FONT_SIZES["cbar_label"])
    cg.ax_cbar.tick_params(labelsize=FONT_SIZES["cbar_tick"])

    # One shared concern colourbar below the z-score bar, sized in inches
    # converted to figure fractions so it keeps one physical size across the
    # 9-32 inch height range the figure spans.
    if annotations:
        bar_height = ANNOTATION_BAR_HEIGHT / fig_height
        bar_gap = ANNOTATION_BAR_GAP / fig_height
        _concern_colorbar(
            cg.figure, [0.02, 0.80 - bar_gap - bar_height, 0.03, bar_height],
            [ROW_ANNOTATIONS[name] for name in annotations], FONT_SIZES,
        )

    # Last, so the forced draw inside it sees final tick-label extents.
    _draw_column_group_tier(ax_heat, column_runs, FONT_SIZES["column_group"])
    return cg


def combined_prediction_pileups(
    cre_matrix, fwd_matrix, rev_matrix, element_type, element_name, flank_size,
    percentile_bands=((5, 95, 0.15), (25, 75, 0.30)),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plots aligned CREST and stranded Puffin tracks for a specific element.

    Spread across instances is drawn as nested percentile ribbons rather than one
    translucent polyline per instance. Every instance contributes, where the
    per-instance version had to cap at 500 traces and so silently showed under
    1% of the data for the most common elements. It also keeps the vector PDF at
    a few hundred kilobytes instead of ~16 MB, since a ribbon is two polygons
    rather than 1500 polylines.

    `percentile_bands` are (low, high, alpha) triples, drawn in the order given,
    so list the widest first and let the narrower ones darken on top of it.
    """
    total_len = cre_matrix.shape[1]
    element_size = total_len - 2 * flank_size
    x_axis = np.arange(total_len) - flank_size

    # Every level any band needs, evaluated in a single pass per matrix. NaNs are
    # present in the CREST pileups, so the nan-aware forms are required here and
    # for the trend lines below.
    levels = sorted({level for low, high, _ in percentile_bands for level in (low, high)})

    def percentile_curves(matrix: np.ndarray) -> dict[int, np.ndarray]:
        return dict(zip(levels, np.nanpercentile(matrix, levels, axis=0)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]})
    plt.subplots_adjust(hspace=0.05)  # tight vertical spacing for alignment

    # --- TOP SUBPLOT: CREST (Unstranded/Aggregate) ---
    cre_curves = percentile_curves(cre_matrix)
    for low, high, alpha in percentile_bands:
        ax1.fill_between(x_axis, cre_curves[low], cre_curves[high], color='gray', alpha=alpha, lw=0)
    ax1.plot(x_axis, np.nanmean(cre_matrix, axis=0), color='crimson', lw=2, label='CREST Mean')
    ax1.set_ylabel("CREST (HEK293T)", fontweight='bold')
    ax1.set_ylim([-1, 8.5])
    ax1.legend(loc='upper right', frameon=False)

    # --- BOTTOM SUBPLOT: PUFFIN (Strand-Aware Mirror Plot) ---
    fwd_curves = percentile_curves(fwd_matrix)
    rev_curves = percentile_curves(rev_matrix)

    # Spread per strand, the reverse strand mirrored below the baseline
    for low, high, alpha in percentile_bands:
        ax2.fill_between(x_axis, fwd_curves[low], fwd_curves[high], color='royalblue', alpha=alpha, lw=0)
        ax2.fill_between(x_axis, -rev_curves[high], -rev_curves[low], color='forestgreen', alpha=alpha, lw=0)

    # Trend lines
    ax2.plot(x_axis, np.nanmean(fwd_matrix, axis=0), color='navy', lw=2, label="Feature Strand (5'→3')")
    ax2.plot(x_axis, -np.nanmean(rev_matrix, axis=0), color='darkgreen', lw=2, label='Opposite Strand')

    # Baseline for mirror plot
    ax2.axhline(0, color='black', lw=1, alpha=0.5)

    ax2.set_ylabel("Puffin CAGE (± Strand)", fontweight='bold')
    ax2.set_ylim([-0.5, 0.5])
    ax2.legend(loc='upper right', frameon=False)

    # --- GLOBAL FORMATTING ---
    for ax in [ax1, ax2]:
        # Highlight normalized core
        ax.axvspan(0, element_size, color='yellow', alpha=0.1, zorder=0)
        ax.axvline(0, color='black', alpha=0.2, ls='--')
        ax.axvline(element_size, color='black', alpha=0.2, ls='--')
        # Cleanup spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.1)

    ax2.set_xlabel("Distance from Element Start (bp)", fontweight='bold')

    # band_text = ", ".join(f"{low}-{high}%" for low, high, _ in percentile_bands)
    fig.suptitle(
        f"Aligned Pileup Profile: {element_type} - {element_name}\n"
        f"(n={len(cre_matrix)} instances)",  # "; shaded bands: {band_text})",
        fontsize=16, fontweight='bold', y=0.95,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, (ax1, ax2)


def element_centered_architecture_heatmap(
    type_matrix: np.ndarray,
    flank_size: int, 
    element_size: int,
    cre_matrix: np.ndarray = None,
    cre_label: str = "CRE Prediction",
    target_label: str = "Element of Interest",
    max_rows: int = 10000
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a publication-quality heatmap centered on instances of a specific feature.
    If a cre_matrix is provided, it uses a dual-layer alpha modulation approach:
    CRE-containing regions are displayed in full vivid color, while non-CRE regions
    are elegantly dimmed to provide structural context without covering it up.
    """
    # 1. Create color tracking bounds
    cmap_list = [GENOMIC_COLORS.get("Backbone", "#E0E0E0")]
    element_types = sorted(ELEMENT_TYPE_PRIORITIES.keys())
    for t in element_types:
        cmap_list.append(GENOMIC_COLORS.get(t, "#95afc0"))
    custom_cmap = ListedColormap(cmap_list)

    # 2. Downsample and Cluster Rows (Hamming distance)
    matrix_to_cluster = type_matrix[:max_rows] if type_matrix.shape[0] > max_rows else type_matrix
    if len(matrix_to_cluster) > 1:
        # Avoid clustering failure if all elements are perfectly identical
        dist_matrix = pdist(matrix_to_cluster, metric='hamming')
        if np.any(dist_matrix):
            row_linkage = linkage(dist_matrix, method='average')
            row_order = leaves_list(row_linkage)
            max_dist = row_linkage[:, 2].max()
            cluster_labels = fcluster(row_linkage, t=0.5 * max_dist, criterion='distance')
            ordered_clusters = cluster_labels[row_order].reshape(-1, 1)
        else:
            row_order = np.arange(len(matrix_to_cluster))
            ordered_clusters = np.ones((len(matrix_to_cluster), 1))
    else:
        row_order = np.arange(len(matrix_to_cluster))
        ordered_clusters = np.ones((len(matrix_to_cluster), 1))

    # 3. Setup Layout GridSpec
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 40], wspace=0.01)
    ax_clusters = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    # Plot Cluster Side Bar
    cluster_cmap = plt.get_cmap('tab20')
    ax_clusters.imshow(ordered_clusters, aspect='auto', interpolation='nearest', 
                       cmap=cluster_cmap, rasterized=True)
    ax_clusters.set_xticks([])
    ax_clusters.set_yticks([])
    for spine in ax_clusters.spines.values():
        spine.set_visible(False)

    # Determine background dimming based on whether we are overlaying annotations
    bg_alpha = 0.15 if cre_matrix is not None else 1.0

    # Plot Base Heatmap Layer (Muted/Dimmed background context)
    ax.imshow(
        matrix_to_cluster[row_order, :], 
        aspect='auto', 
        interpolation='none', 
        cmap=custom_cmap, 
        vmin=0, 
        vmax=len(element_types),
        alpha=bg_alpha,
        rasterized=True
    )

    # 4. ELEGANT OVERLAY: Masked Alpha Modulation
    if cre_matrix is not None:
        cre_to_cluster = cre_matrix[:max_rows] if cre_matrix.shape[0] > max_rows else cre_matrix
        cre_ordered = cre_to_cluster[row_order, :]

        # Create a masked array where positions WITHOUT a CRE prediction (0) are hidden
        masked_architecture = np.ma.masked_where(cre_ordered == 0, matrix_to_cluster[row_order, :])

        # Overlay the illuminated layer at full opacity (alpha=1.0)
        ax.imshow(
            masked_architecture,
            aspect='auto', 
            interpolation='none', 
            cmap=custom_cmap, 
            vmin=0, 
            vmax=len(element_types),
            alpha=1.0,
            rasterized=True,
            zorder=3
        )

    # 5. Boundary Indicator Lines & Shading
    start_idx = flank_size
    end_idx = flank_size + element_size
    ax.axvline(x=start_idx, color="#1e293b", linestyle="--", lw=1.5, alpha=0.85, zorder=4)
    ax.axvline(x=end_idx, color="#1e293b", linestyle="--", lw=1.5, alpha=0.85, zorder=4)
    ax.axvspan(start_idx, end_idx, color="#0f172a", alpha=0.03, zorder=2)

    # 6. X-Axis Labeling (Modified to show exact positions at borders instead of center)
    tick_positions = [0, start_idx, end_idx, type_matrix.shape[1] - 1]
    tick_labels = [f"-{flank_size} bp", "Start (0 bp)", f"End ({element_size} bp)", f"+{flank_size} bp"]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11, fontweight='medium')
    ax.set_xlabel(f"Localized Distance Coordinates Relative to {target_label}", fontweight='bold', fontsize=14, labelpad=12)
    ax.set_yticks([])

    ax.set_title(f"{target_label}, {cre_label} (n={len(matrix_to_cluster)})", fontweight='bold', fontsize=16, pad=25)
    
    # 7. Unified Legend Generation
    legend_elements = [Line2D([0], [0], color=cmap_list[0], lw=8, label='Backbone')]
    for i, t in enumerate(element_types):
        legend_elements.append(Line2D([0], [0], color=cmap_list[i+1], lw=8, label=t))

    # Append an elegant intensity guide key to the legend to guide interpretation
    if cre_matrix is not None:
        legend_elements.append(Line2D([0], [0], color='none', label='')) # Blank spacer
        legend_elements.append(
            Line2D([0], [0], color='#475569', lw=8, alpha=1.0, label=f'Solid Color: {cre_label}')
        )
        legend_elements.append(
            Line2D([0], [0], color='#475569', lw=8, alpha=0.35, label=f'Muted Color: no {cre_label}')
        )

    ax.legend(
        handles=legend_elements, 
        title="Map Features & Intensity", 
        title_fontproperties={'weight': 'bold', 'size': 11},
        bbox_to_anchor=(1.01, 1), 
        loc='upper left', 
        fontsize=10, 
        frameon=False
    )

    sns.despine(left=True, bottom=False)
    return fig, ax
