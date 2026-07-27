import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from adjustText import adjust_text
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster


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
    x_col: str = "n_cre_midpoints",
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
    """
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
        arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.6, alpha=0.7),
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


def functional_profiling_plot(df_clustered: pl.DataFrame, heatmap_df: pl.DataFrame) -> sns.matrix.ClusterGrid:
    df_clustered, heatmap_df = df_clustered.to_pandas(), heatmap_df.to_pandas()

    # --- 1. Render the Ordered Heatmap ---
    sns.set_theme(style="white", context="paper", font_scale=1.1)

    # Map row colors
    row_colors = df_clustered['type'].map(lambda x: GENOMIC_COLORS.get(x, '#cccccc'))
    row_colors.name = "Type"

    # Use clustermap but DISABLE row clustering so our strict sorting is preserved
    cg = sns.clustermap(
        heatmap_df,
        row_cluster=False,       # KEEP rows sorted by our Priority ranking
        col_cluster=True,        # Let metrics group if they behave similarly
        row_colors=row_colors,
        cmap="RdBu_r",
        vmin=-2.5,
        vmax=7.5,
        center=0,
        figsize=(11, 9),
        cbar_kws={'label': 'Relative Values\n(Z-Score)'},
        colors_ratio=0.03
    )

    # Draw separators between Priority Groups
    ax_heat = cg.ax_heatmap
    current_priority = df_clustered['priority_group'].iloc[0]

    for i, priority in enumerate(df_clustered['priority_group']):
        if priority != current_priority:
            # Draw a solid line when the priority group changes
            ax_heat.axhline(y=i, color='black', linewidth=1.5, linestyle='-')
            current_priority = priority

    # Formatting
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax_heat.set_yticklabels([])
    ax_heat.set_ylabel(f"N = {len(df_clustered)}", fontsize=12, fontweight='bold')
    cg.ax_col_dendrogram.set_title("Functional Profiling of Plasmid Elements", fontsize=14, fontweight='bold', pad=20)

    # Add Element Type Legend
    legend_patches = [
        mpatches.Patch(color=color, label=el_type) 
        for el_type, color in GENOMIC_COLORS.items() if el_type in df_clustered['type'].unique()
    ]
    cg.ax_row_dendrogram.legend(
        handles=legend_patches, title="Element Type", title_fontproperties={'weight': 'bold'},
        loc="lower left", bbox_to_anchor=(-0.4, -0.2), frameon=False
    )
    cg.ax_cbar.set_position([0.02, 0.8, 0.03, 0.15])
    return cg


def combined_prediction_pileups(
    cre_matrix, fwd_matrix, rev_matrix, element_type, element_name, flank_size
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plots aligned CREST and stranded Puffin tracks for a specific element.
    """
    total_len = cre_matrix.shape[1]
    element_size = total_len - 2 * flank_size
    x_axis = np.arange(total_len) - flank_size

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]})
    plt.subplots_adjust(hspace=0.05)  # tight vertical spacing for alignment

    # --- TOP SUBPLOT: CREST (Unstranded/Aggregate) ---
    c_mean = np.nanmean(cre_matrix, axis=0)
    for i in range(min(len(cre_matrix), 500)): # Plot subset of tracks for clarity
        ax1.plot(x_axis, cre_matrix[i], color='gray', alpha=0.03, lw=0.5)
    ax1.plot(x_axis, c_mean, color='crimson', lw=2, label='CREST Mean')
    ax1.set_ylabel("CREST Signal", fontweight='bold')
    ax1.set_ylim([-1, 8.5])
    ax1.legend(loc='upper right', frameon=False)

    # --- BOTTOM SUBPLOT: PUFFIN (Strand-Aware Mirror Plot) ---
    pf_mean = np.mean(fwd_matrix, axis=0)
    pr_mean = np.mean(rev_matrix, axis=0)

    # Individual tracks (Strand-aware)
    for i in range(min(len(fwd_matrix), 500)):
        ax2.plot(x_axis, fwd_matrix[i], color='royalblue', alpha=0.02, lw=0.5)
        ax2.plot(x_axis, -rev_matrix[i], color='forestgreen', alpha=0.02, lw=0.5)

    # Trend lines
    ax2.plot(x_axis, pf_mean, color='navy', lw=2, label='Feature Strand (5\'→3\')')
    ax2.plot(x_axis, -pr_mean, color='darkgreen', lw=2, label='Opposite Strand')
    
    # Baseline for mirror plot
    ax2.axhline(0, color='black', lw=1, alpha=0.5)
    
    ax2.set_ylabel("Puffin Signal (± Strand)", fontweight='bold')
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

    fig.suptitle(f"Aligned Pileup Profile: {element_type} - {element_name}\n(n={len(cre_matrix)} instances)", fontsize=16, fontweight='bold', y=0.95)
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
