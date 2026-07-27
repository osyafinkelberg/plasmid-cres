import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from matplotlib.patches import FancyArrow

from .helpers import extract_feature_name
from .statplots import GENOMIC_COLORS


def get_polar_angle(position, seq_length):
    # top-start and clockwise rotation
    return 2 * np.pi * (position / seq_length)


def plot_backbone(gb_file: Path) -> plt.Axes:
    record = SeqIO.read(gb_file, "genbank")
    comments = record.annotations.get("comment", "")
    match = re.search(r"Sequence Label:\s*(.*)", comments)
    assert match
    seq_label = match.group(1).strip()

    seq_len = len(record.seq)

    _, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_offset(np.pi / 2)  # 0 at the top
    ax.axis('off')  # hide standard polar grid

    feature_radius = 1.0
    ax.plot(np.linspace(0, 2*np.pi, 1000), [feature_radius]*1000, color='gray', lw=3)

    legend_patches = []

    # default widths for major features (genes, CDS, origins)
    major_width = 0.12
    major_head_width = 0.22

    # default widths for minor features (promoters, signals)
    minor_width = 0.04
    minor_head_width = 0.07

    max_head_bp = 150

    for feature in record.features:
        if feature.type in ['source']: 
            continue  # skip plotting the whole plasmid source
        if "primer" in feature.type:
            continue

        strand = feature.location.strand
        name = extract_feature_name(feature)
        parts = feature.location.parts
        cw_parts = list(reversed(parts)) if strand == -1 else parts

        if len(cw_parts) > 1 and cw_parts[0].start > cw_parts[-1].start:
            start = int(cw_parts[0].start)
            end = int(cw_parts[-1].end)
        else:
            start = int(feature.location.start)
            end = int(feature.location.end)

        color = GENOMIC_COLORS.get(feature.type)
        if not color:
            if strand == -1:
                color = feature.qualifiers.get('ApEinfo_revcolor', ['#cccccc'])[0]
            else:
                color = feature.qualifiers.get('ApEinfo_fwdcolor', ['#cccccc'])[0]

        # visual hierarchy: determine if feature is major or minor
        is_major = feature.type in ["CDS", "gene", "rep_origin", "promoter", "enhancer", "LTR"]
        w = major_width if is_major else minor_width
        hw = major_head_width if is_major else minor_head_width

        actual_end = end if end > start else end + seq_len
        feature_len = actual_end - start

        head_bp_calc = int(feature_len * 0.8) if is_major else int(feature_len * 0.4)
        head_bp = min(head_bp_calc, max_head_bp)

        if strand == -1:
            p_outer = np.linspace(start + head_bp, actual_end, 50)
            p_inner = np.linspace(actual_end, start + head_bp, 50)
            pos = np.concatenate([p_outer, p_inner, [start + head_bp, start, start + head_bp]])
            rs = np.concatenate([
                np.full(50, feature_radius + w/2),
                np.full(50, feature_radius - w/2),
                [feature_radius - hw/2, feature_radius, feature_radius + hw/2]
            ])
        else:
            p_outer = np.linspace(start, actual_end - head_bp, 50)
            p_inner = np.linspace(actual_end - head_bp, start, 50)
            pos = np.concatenate([
                p_outer,
                [actual_end - head_bp, actual_end, actual_end - head_bp],
                p_inner
            ])
            rs = np.concatenate([
                np.full(50, feature_radius + w/2),
                [feature_radius + hw/2, feature_radius, feature_radius - hw/2],
                np.full(50, feature_radius - w/2)
            ])

        thetas = get_polar_angle(pos, seq_len)
        ax.fill(thetas, rs, color=color, edgecolor='black', lw=0.8, alpha=0.9, zorder=3) 

        if feature.type in ["CDS", "gene", "rep_origin", "enhancer"]:
            center_pos_bp = start + feature_len / 2
            center_theta = get_polar_angle(center_pos_bp, seq_len)
            text_r = feature_radius + w/2 + 0.08
            rot_deg = -np.degrees(center_theta)
            if rot_deg < -90 or rot_deg > 90:
                rot_deg += 180

            ax.text(
                center_theta, text_r, name, 
                ha='center', va='center', 
                rotation=rot_deg, 
                fontsize=9, fontweight='bold', zorder=4
            )

        if any(p.get_label() == name for p in legend_patches):
            continue
        legend_patches.append(mpatches.Patch(color=color, label=name))

    backbone_legend = ax.legend(
        handles=legend_patches, 
        loc='upper left', 
        bbox_to_anchor=(0, 1.05), 
        title="Plasmid Elements", 
        frameon=False,
        prop={'size': 9}
    )
    ax.add_artist(backbone_legend) 
    ax.text(0, 0, f"{seq_label}\n{seq_len} bp", ha='center', va='center', fontsize=12, fontweight='bold')

    return ax


def plot_track(
    ax: plt.Axes, track: np.ndarray, track_color: str,
    track_base_radius: float, track_label: str, k_norm: float | None = None,
    direction: str | None = None
) -> None:

    # Map the rotations to actual plasmid coordinates relative to minP
    # We use modulo seq_len so that coordinates wrapping past the plasmid ends loop correctly
    track_angles = get_polar_angle(np.arange(len(track)), len(track))

    # normalize track
    if k_norm is None:
        pred_normalized = (track - np.min(track)) / (np.max(track) - np.min(track))
    else:
        pred_normalized = track / k_norm
    track_radii = track_base_radius + (pred_normalized * 0.3)

    sort_idx = np.argsort(track_angles)
    ax.plot(track_angles[sort_idx], track_radii[sort_idx], color=track_color, lw=1.5, label=track_label)
    ax.fill_between(track_angles[sort_idx], track_base_radius, track_radii[sort_idx], color=track_color, alpha=0.3)
    ax.plot(np.linspace(0, 2*np.pi, 1000), [track_base_radius]*1000, color='black', lw=0.8, ls='--')
    if direction in ['cw', 'ccw']:
        arrow_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for ang in arrow_angles:
            ang_end = ang + 0.1 if direction == 'cw' else ang - 0.1
            ax.annotate(
                '', xy=(ang_end, track_base_radius), xytext=(ang, track_base_radius),
                arrowprops={"arrowstyle": "simple", "color": "grey", "lw": 2}
            )


def plot_track_heatmap(
    ax: plt.Axes, track: np.ndarray, cmap: str,
    track_base_radius: float, track_width: float, 
    track_label: str, k_norm: float | None = None,
    direction: str | None = None
) -> None:
    seq_len = len(track)
    
    # pcolormesh requires bin edges, so we need seq_len + 1 elements
    theta_edges = get_polar_angle(np.arange(seq_len + 1), seq_len)
    r_edges = [track_base_radius, track_base_radius + track_width]
    
    # Create the 2D meshgrid for polar plotting
    Theta, R = np.meshgrid(theta_edges, r_edges)
    
    # Values array must be 2D (1 row, seq_len columns)
    Z = track.reshape(1, -1)
    
    # Set up normalization
    if k_norm is None:
        vmin, vmax = np.min(track), np.max(track)
        if vmin == vmax: 
            vmax = vmin + 1e-6 # prevent division by zero in norm
    else:
        vmin, vmax = 0, k_norm
        
    # Draw the circular heatmap
    ax.pcolormesh(Theta, R, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    
    # Extract a representative color from the colormap to use in the legend
    cmap_obj = plt.get_cmap(cmap)
    legend_color = cmap_obj(0.7) 
    
    # Add a proxy artist (invisible line) so the track shows up in ax.legend()
    ax.plot([], [], color=legend_color, lw=5, label=track_label)
    
    # Add direction arrows if specified
    if direction in ['cw', 'ccw']:
        arrow_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        arrow_r = track_base_radius + (track_width / 2)
        for ang in arrow_angles:
            ang_end = ang + 0.1 if direction == 'cw' else ang - 0.1
            ax.annotate(
                '', xy=(ang_end, arrow_r), xytext=(ang, arrow_r),
                arrowprops={"arrowstyle": "simple", "color": "black", "lw": 1, "alpha": 0.5}
            )


def plot_peak_annotation(
    ax: plt.Axes,
    peaks,
    seq_len: int,
    track_base_radius: float,
    track_width: float,
    color: str = "#333333",
    alpha: float = 0.80,
    line_width: float = 2.5,
    tick_size: float = 0.04,
) -> None:
    """
    Overlay lightweight peak annotations on a circular heatmap track.

    For each annotated peak interval, renders:
      - A thin arc along the track's outer rim spanning the peak region.
      - Small radial bracket ticks at the peak's start and end boundaries.

    Everything is drawn at / just beyond the outer edge of the target ring so
    the underlying heatmap remains fully visible.  Annotations sit inside the
    ~0.05 radial gap between consecutive track rings.

    Parameters
    ----------
    ax : plt.Axes
        Polar axes returned by ``plot_backbone``.
    peaks : list
        Peak intervals as ``[[start_bp, end_bp], ...]``.
        Also accepts the raw nested output of a polars column's ``.to_list()``
        on a single-plasmid-filtered DataFrame — i.e. ``[[[s, e], ...]]`` —
        and unwraps the extra level automatically.
    seq_len : int
        Plasmid length in bp; must match the track that was plotted.
    track_base_radius : float
        Inner radius of the heatmap ring (same value passed to
        ``plot_track_heatmap``).
    track_width : float
        Radial thickness of the heatmap ring (same value passed to
        ``plot_track_heatmap``).
    color : str
        Annotation colour.  Near-black by default: legible over any colormap
        without clashing with the track palette.
    alpha : float
        Opacity of annotation elements (0–1).
    line_width : float
        Stroke width (pt) for arcs and tick marks.
    tick_size : float
        Radial span of boundary ticks.  ~35 % descends into the track,
        ~65 % extends outward into the gap.

    Examples
    --------
    # Puffin forward track
    examplots.plot_peak_annotation(
        ax, plasmid_fwd_tss, seq_len=len(puffin_fwd_preds),
        track_base_radius=1.3, track_width=track_width,
    )

    # CREST tracks
    for i, cell in enumerate(CELL_TYPES):
        r = crest_base_r + i * (track_width + 0.05)
        examplots.plot_peak_annotation(
            ax, plasmid_crest_peaks[cell], seq_len=len(plasmid_crest_preds[cell]),
            track_base_radius=r, track_width=track_width,
        )
    """
    if not peaks or peaks == [[]]:
        return

    # ── unwrap polars .to_list() nesting: [[[s,e],...]] → [[s,e],...] ─────────
    first = peaks[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        peaks = first

    if not peaks:
        return

    outer_r  = track_base_radius + track_width   # outer rim of the ring
    tick_lo  = outer_r - tick_size * 0.35        # tick foot, inside track
    tick_hi  = outer_r + tick_size * 0.65        # tick head, in the gap

    for interval in peaks:
        s_bp = int(interval[0])
        e_bp = int(interval[1])

        # handle wrap-around: peak crosses the sequence origin
        actual_end = e_bp if e_bp >= s_bp else e_bp + seq_len
        span_bp    = actual_end - s_bp

        # ── arc spanning the peak interval ────────────────────────────────────
        n_pts  = max(40, span_bp // 4)
        bp_pos = np.linspace(s_bp, actual_end, n_pts)
        thetas = get_polar_angle(bp_pos, seq_len)

        ax.plot(
            thetas, np.full(n_pts, outer_r),
            color=color, alpha=alpha, lw=line_width,
            solid_capstyle="round", zorder=6,
        )

        # ── radial bracket ticks at both boundaries ───────────────────────────
        # use original e_bp (not actual_end) so the tick lands on the correct
        # angular position even when the peak wraps around the origin
        for boundary_bp in (s_bp, e_bp):
            t = float(get_polar_angle(np.array([boundary_bp]), seq_len)[0])
            ax.plot(
                [t, t], [tick_lo, tick_hi],
                color=color, alpha=min(alpha + 0.15, 1.0),
                lw=line_width, solid_capstyle="butt", zorder=6,
            )


def plot_track_linear(
    track: np.ndarray,
    start_bp: int,
    end_bp: int,
    peaks=None,
    label: str = "",
    color: str = "#2a6ebb",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot a slice of a prediction track as a simple filled line plot.

    Parameters
    ----------
    track     : full-plasmid 1-D array of signal values.
    start_bp  : start of the interval to display (bp, inclusive).
    end_bp    : end   of the interval to display (bp, exclusive).
    peaks     : peak intervals [[s, e], ...] (same format accepted by
                plot_peak_annotation, polars nesting unwrapped automatically).
    label     : subplot title / y-label string.
    color     : fill / line colour.
    ax        : existing Axes to draw into; creates a new figure if None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.2))

    x = np.arange(start_bp, end_bp)
    y = track[start_bp:end_bp]

    ax.fill_between(x, y, alpha=0.35, color=color, linewidth=0)
    ax.plot(x, y, color=color, linewidth=0.9)

    # ── peak shading ──────────────────────────────────────────────────────────
    if peaks:
        # unwrap polars nesting
        if isinstance(peaks[0], list) and peaks[0] and isinstance(peaks[0][0], list):
            peaks = peaks[0]
        for s, e in peaks:
            if e < start_bp or s > end_bp:
                continue
            ax.axvspan(max(s, start_bp), min(e, end_bp),
                       color=color, alpha=0.18, linewidth=0, zorder=0)
            # boundary ticks
            for b in (s, e):
                if start_bp <= b <= end_bp:
                    ax.axvline(b, color=color, alpha=0.55,
                               linewidth=1.0, linestyle="--")

    ax.set_xlim(start_bp, end_bp - 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Position (bp)")
    ax.set_ylabel("Signal")
    ax.set_title(label, fontsize=10, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    return ax


def plot_linear_features(
    gb_file: Path,
    start_bp: int,
    end_bp: int,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Draw a simple linear feature map for all GenBank elements overlapping
    [start_bp, end_bp].  Designed to sit as the bottom subplot below stacked
    ``plot_track_linear`` panels that share the same x-axis.

    Features are drawn as directional arrows (strand-aware) and stacked into
    non-overlapping lanes automatically.  Partially visible features are drawn
    at full length and clipped by the axes limits, so a flat edge indicates
    the feature continues beyond the view.

    Parameters
    ----------
    gb_file  : path to the GenBank (.gbk / .gb) file.
    start_bp : left boundary of the region to display (bp).
    end_bp   : right boundary of the region to display (bp).
    ax       : existing Axes to draw into; a new figure is created if None.
    """
    record    = SeqIO.read(gb_file, "genbank")
    seq_len   = len(record.seq)
    view_span = end_bp - start_bp

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))

    # ── collect features that overlap the view ────────────────────────────────
    features_in_view = []
    for feature in record.features:
        if feature.type in ["source"] or "primer" in feature.type:
            continue

        strand  = feature.location.strand
        parts   = feature.location.parts
        f_start = min(int(p.start) for p in parts)
        f_end   = max(int(p.end)   for p in parts)

        if f_end < f_start:          # wrap-around feature (crosses origin)
            f_end += seq_len

        if f_end <= start_bp or f_start >= end_bp:
            continue                 # completely outside the view

        name  = extract_feature_name(feature)
        color = GENOMIC_COLORS.get(feature.type)
        if not color:
            qual_key = "ApEinfo_revcolor" if strand == -1 else "ApEinfo_fwdcolor"
            color = feature.qualifiers.get(qual_key, ["#cccccc"])[0]

        features_in_view.append({
            "start":  f_start,
            "end":    f_end,
            "strand": strand,
            "name":   name,
            "color":  color,
            "type":   feature.type,
        })

    features_in_view.sort(key=lambda f: f["start"])

    # ── greedy lane assignment ────────────────────────────────────────────────
    padding     = max(30, view_span // 80)   # minimum gap between features in the same lane
    lane_rights = []                         # rightmost visible end reached in each lane
    lane_idxs   = []

    for feat in features_in_view:
        vis_start = max(feat["start"], start_bp)
        vis_end   = min(feat["end"],   end_bp)
        placed    = False
        for i, right in enumerate(lane_rights):
            if vis_start >= right + padding:
                lane_rights[i] = vis_end
                lane_idxs.append(i)
                placed = True
                break
        if not placed:
            lane_rights.append(vis_end)
            lane_idxs.append(len(lane_rights) - 1)

    n_lanes  = max(len(lane_rights), 1)
    lane_h   = 0.50    # shaft height
    head_h   = 0.82    # arrowhead height (slightly wider than shaft)
    lane_sep = 1.60    # vertical distance between lane centres

    # ── draw each feature ────────────────────────────────────────────────────
    for feat, lane_idx in zip(features_in_view, lane_idxs):
        y    = lane_idx * lane_sep
        s, e = feat["start"], feat["end"]
        span = e - s

        is_major = feat["type"] in [
            "CDS", "gene", "rep_origin", "promoter", "enhancer", "LTR"
        ]
        # arrowhead: proportional to feature span, capped at 4 % of view
        head_len = min(span * 0.22, view_span * 0.04) if is_major else 0

        arrow_kw = dict(  # noqa: C408
            width=lane_h,
            head_width=head_h if is_major else lane_h,
            head_length=head_len,
            length_includes_head=True,
            color=feat["color"],
            edgecolor="#222222",
            linewidth=0.5,
            alpha=0.88,
            zorder=3,
            clip_on=True,
        )

        if feat["strand"] == -1:
            ax.add_patch(FancyArrow(e, y, -span, 0, **arrow_kw))
        else:
            ax.add_patch(FancyArrow(s, y,  span, 0, **arrow_kw))

        # ── label centred on the visible portion ──────────────────────────────
        vis_centre    = (max(s, start_bp) + min(e, end_bp)) / 2
        visible_chars = (min(e, end_bp) - max(s, start_bp)) / view_span * 80
        label = (
            feat["name"]
            if len(feat["name"]) <= visible_chars
            else feat["name"][: max(int(visible_chars) - 1, 2)] + "…"
        )
        ax.text(
            vis_centre, y, label,
            ha="center", va="center",
            fontsize=7, fontweight="bold",
            color="black", zorder=4, clip_on=True,
        )

    # ── axes chrome ───────────────────────────────────────────────────────────
    ax.set_xlim(start_bp, end_bp)
    ax.set_ylim(-lane_sep * 0.8, (n_lanes - 1) * lane_sep + lane_sep * 0.8)
    ax.set_xlabel("Position (bp)")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Plasmid Features", fontsize=9, fontweight="bold", loc="left")

    return ax
