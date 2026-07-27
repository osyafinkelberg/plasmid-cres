import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import logomaker


def rolling_absolute_contribution_scores(scores: np.ndarray, window: int = 10) -> np.ndarray:
    negative_sums = -scores.clip(max=0).sum(axis=0)
    positive_sums = scores.clip(min=0).sum(axis=0)
    total_abs_contrib = np.maximum(negative_sums, positive_sums)
    cumsum = np.cumsum(total_abs_contrib)
    half_window_left = (window + 1) // 2
    half_window_right = window // 2
    total_abs_contrib[half_window_left: -half_window_right] = (
        cumsum[window:] - cumsum[:-window]
    )
    return total_abs_contrib


def contribution_scores_plot(
    scores: np.ndarray, 
    tacs_window: int = 5,
    per_pos_threshold: float = 0.15,
    y_min: float = -1,
    y_max: float = 2.5,
    cre_label: str = "CRE",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Renders importance score logo distributions layered with TACS and 
    secondary regulatory density tracks, ensuring perfectly aligned zero-baselines.
    """
    score_df = pl.DataFrame(scores.astype(np.float64).T, schema=["A", "C", "G", "T"]).to_pandas()
    length = score_df.shape[0]
    
    fig, ax = plt.subplots(figsize=(18 * length / 200, 8), dpi=300)
    
    # 1. Base Sequence Logo canvas mapping
    logomaker.Logo(score_df, ax=ax, center_values=False)
    
    # 2. Superimpose structural TACS sequence track
    tacs = rolling_absolute_contribution_scores(scores, window=tacs_window)
    ax.plot(np.arange(length), tacs, color='cornflowerblue', linewidth=3.0, label='TACS', zorder=4)
    ax.axhline(per_pos_threshold * tacs_window, linestyle='--', color='red', linewidth=1.2, alpha=0.7, zorder=3)
    
    # Define primary axis bounds
    y1_min, y1_max = y_min, y_max
    ax.set_ylim([y1_min, y1_max])
    ax.set_xlim([0, length])
    
    # Add a clean, shared baseline at zero to anchor both datasets visually
    ax.axhline(0, color='#94a3b8', linestyle='-', linewidth=1.0, alpha=0.5, zorder=2)

    ax.tick_params(axis='x', bottom=True, labelbottom=True)
    ax.grid(False)
    plt.subplots_adjust(wspace=0, hspace=0.1)
    
    return fig, ax


def apply_element_annotations(
    fig: plt.Figure, 
    ax: plt.Axes, 
    slice_start: int, 
    slice_end: int, 
    flank_size: int, 
    element_size: int,
    element_label: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Dynamically applies biological background shading and relative coordinates 
    to any sliced window segment of the plasmid visualization.
    """
    ori_start = flank_size
    ori_end = flank_size + element_size
    window_len = slice_end - slice_start
    
    # Translate absolute boundaries into the current slice's coordinate system
    local_ori_start = ori_start - slice_start
    local_ori_end = ori_end - slice_start
    
    # 1. Render adaptive domain shading based on what falls within the window view
    # Left Flank Region
    if local_ori_start > 0:
        ax.axvspan(0, min(local_ori_start, window_len), color='#f8fafc', alpha=0.6, zorder=0)
    
    # Ori Element Region
    ori_view_start = max(0, local_ori_start)
    ori_view_end = min(local_ori_end, window_len)
    if ori_view_start < ori_view_end:
        ax.axvspan(ori_view_start, ori_view_end, color='#f0f9ff', alpha=0.6, zorder=0)
        
    # Right Flank Region
    if local_ori_end < window_len:
        ax.axvspan(max(0, local_ori_end), window_len, color='#f8fafc', alpha=0.6, zorder=0)
        
    # Draw boundary markers if they intersect the active viewport slice
    if 0 <= local_ori_start <= window_len:
        ax.axvline(x=local_ori_start, color='#0284c7', linestyle='--', linewidth=1.2, alpha=0.5, zorder=1)
    if 0 <= local_ori_end <= window_len:
        ax.axvline(x=local_ori_end, color='#0284c7', linestyle='--', linewidth=1.2, alpha=0.5, zorder=1)

    # 2. Build aligned coordinate ticks
    tick_positions = []
    tick_labels = []
    
    # Inject primary milestone labels if they are visible
    if 0 <= local_ori_start <= window_len:
        tick_positions.append(local_ori_start)
        tick_labels.append(f"0 bp\n({element_label} Start)")
    if 0 <= local_ori_end <= window_len:
        tick_positions.append(local_ori_end)
        tick_labels.append(f"{element_size} bp\n({element_label} End)")
        
    # Fill remaining space with uniform intervals (every 100 bp)
    round_start_abs = ((slice_start + 99) // 100) * 100
    for abs_pos in range(round_start_abs, slice_end + 1, 100):
        local_pos = abs_pos - slice_start
        # Prevent layout clashing with primary biological milestones
        if 0 <= local_pos <= window_len and not any(abs(local_pos - t) < 15 for t in tick_positions):
            tick_positions.append(local_pos)
            rel_coord = abs_pos - ori_start
            tick_labels.append(f"+{rel_coord} bp" if rel_coord > 0 else f"{rel_coord} bp")
            
    # Sort coordinates concurrently to prevent visual overlapping artifacts
    sorted_order = np.argsort(tick_positions)
    ax.set_xticks([tick_positions[i] for i in sorted_order])
    ax.set_xticklabels([tick_labels[i] for i in sorted_order], fontsize=13, rotation=90, color='#334155')
    
    return fig, ax
