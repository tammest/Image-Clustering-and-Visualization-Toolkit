import os, re
import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import transforms as mtransforms  # for arrows below hist axes

# ================================
# Paths & data
# ================================
ROOT_PATH = "data/ome_tiffs"
FEATURES_CSV = "data/features_with_clusters.csv"
OUT_DIR = "outputs/marker_colocalization_triptychs"
features     = pd.read_csv(FEATURES_CSV)

# ================================
# Channels (0-based)
# ================================
CH_DAPI = 0
CH_F480 = 3   # GREEN in overlay
CH_CD31 = 4   # RED in overlay

# Pretty labels (titles) vs. safe labels (filenames)
CH_NAMES_PRETTY = {CH_CD31: "CD31", CH_F480: "F4/80", CH_DAPI: "DAPI"}
CH_NAMES_SAFE   = {CH_CD31: "CD31", CH_F480: "F4_80", CH_DAPI: "DAPI"}  # for filenames

# ================================
# Publication style
# ================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",  # change to "Arial" if available
    "figure.dpi": 120,
    "axes.linewidth": 0.8,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 9,
})

# Map colors
COL_NEITHER = "#CFCFCF"   # gray
COL_ONE     = "#1f77b4"   # blue (one channel high)
COL_BOTH    = "#d62728"   # red  (both high)

# Histogram colors (lines/titles)
COL_HIST_CD31 = "#FF0000"
COL_HIST_F480 = "#00AA00"

# Tile rendering knobs
IMG_SIZE       = 128
DAPI_FAINT     = 1.22
CLIP_PCT_TILE  = 99.0

# Thresholds (map)
Q_CD31 = 0.75
Q_F480 = 0.75
USE_QUANTILES = True

# Layout tuning
MAP_LEGEND_X = 0.02
MAP_LEGEND_Y = -0.20
WSPACE = 0.30
HSPACE = 0.52
BOTTOM_PAD = 0.32

# ================================
# Small utilities
# ================================
def safe_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def short_name(s: str, maxlen: int = 36) -> str:
    s2 = s.replace("_", " ")
    return (s2[:maxlen-1] + "…") if len(s2) > maxlen else s2

def _normalize_clip(arr: np.ndarray, clip_hi_pct: float = 99.0) -> np.ndarray:
    if arr.size == 0: return arr
    pos = arr[arr > 0]
    if pos.size == 0: return np.zeros_like(arr)
    clip_hi = np.percentile(pos, clip_hi_pct)
    if clip_hi <= 0: clip_hi = float(pos.max())
    norm = np.clip(arr, 0, clip_hi)
    vmin, vmax = float(norm.min()), float(norm.max())
    if vmax > vmin:
        norm = (norm - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(norm)
    return norm

def load_sample_zarr(root_path: str, sample_name: str):
    zpath = os.path.join(root_path, sample_name, "data.zarr")
    if not os.path.exists(zpath):
        raise FileNotFoundError(f"Zarr not found: {zpath}")
    return zarr.open(zpath, mode="r")

# ================================
# Map
# ================================
def plot_threeclass_map(ax, sample_df: pd.DataFrame, sample_name: str,
                        chA: int, chB: int,
                        qA: float, qB: float,
                        use_quantile: bool = True):
    s_norm = sample_name.strip().lower()
    df = sample_df.copy()
    df['sample_name'] = df['sample_name'].str.strip().str.lower()
    sample = df[df['sample_name'] == s_norm].copy()
    if sample.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return None

    w = pd.to_numeric(sample['w'], errors='coerce').to_numpy()
    h = pd.to_numeric(sample['h'], errors='coerce').to_numpy()
    A = pd.to_numeric(sample[f'channel_{chA}'], errors='coerce').fillna(0).to_numpy(float)
    B = pd.to_numeric(sample[f'channel_{chB}'], errors='coerce').fillna(0).to_numpy(float)

    if use_quantile:
        tA = np.quantile(A, np.clip(qA, 0, 1))
        tB = np.quantile(B, np.clip(qB, 0, 1))
    else:
        tA, tB = float(qA), float(qB)

    aboveA = A > tA
    aboveB = B > tB
    both    = aboveA & aboveB
    one     = aboveA ^ aboveB
    neither = ~(aboveA | aboveB)

    total = len(A)
    n_neither = int(neither.sum())
    n_one     = int(one.sum())
    n_both    = int(both.sum())

    def _layer(mask, color, size, z):
        if not np.any(mask): return
        ax.scatter(w[mask], h[mask], s=size, c=color, alpha=0.9, edgecolors='none', zorder=z)

    _layer(neither, COL_NEITHER, 9, 1)
    _layer(one,     COL_ONE,    12, 2)
    _layer(both,    COL_BOTH,   15, 3)

    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_xlabel("w")
    ax.set_ylabel("h")
    ax.set_title(f"Three-class map\n{sample_name}", pad=6)
    for s in ("top","right"): ax.spines[s].set_visible(False)

    handles = [
        Patch(facecolor=COL_NEITHER, edgecolor='none', label=f"neither (n={n_neither}, {n_neither/total:.1%})"),
        Patch(facecolor=COL_ONE,     edgecolor='none', label=f"one (n={n_one}, {n_one/total:.1%})"),
        Patch(facecolor=COL_BOTH,    edgecolor='none', label=f"both (n={n_both}, {n_both/total:.1%})"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              bbox_to_anchor=(MAP_LEGEND_X, MAP_LEGEND_Y), ncol=1)

    return dict(w=w, h=h, A=A, B=B, tA=tA, tB=tB, aboveA=aboveA, aboveB=aboveB)

# ================================
# Tiles
# ================================
def render_tile_triptych(arr, h, w, size, chA, chB, chDapi, clip_pct=99.0, dapi_gain=0.22):
    C, Htot, Wtot = arr.shape
    y1 = min(h + size, Htot); x1 = min(w + size, Wtot)
    A = _normalize_clip(arr[chA][h:y1, w:x1], clip_pct)
    B = _normalize_clip(arr[chB][h:y1, w:x1], clip_pct)
    D = _normalize_clip(arr[chDapi][h:y1, w:x1], 99.0)
    overlay = np.zeros((A.shape[0], A.shape[1], 3), dtype=float)
    overlay[..., 0] = A
    overlay[..., 1] = B
    overlay[..., 2] += D * dapi_gain
    overlay = np.clip(overlay, 0, 1)
    return A, B, overlay

def pick_tile(info):
    both_mask = info["aboveA"] & info["aboveB"]
    if np.any(both_mask):
        idx = np.flatnonzero(both_mask)[0]
    else:
        zA = (info["A"] - info["A"].mean()) / (info["A"].std() + 1e-8)
        zB = (info["B"] - info["B"].mean()) / (info["B"].std() + 1e-8)
        idx = int(np.argmax(zA + zB))
    return int(info["h"][idx]), int(info["w"][idx])

# ================================
# Per-sample histogram (now supports fixed x-range)
# ================================
def plot_channel_hist(ax, values, cutoff, title, color_line, x_label,
                      percentile_text=None, x_range=None):
    """
    If x_range=(lo, hi) is provided, the histogram uses that fixed range
    (and sets the same xlim). Otherwise it uses the sample's [1,99] pct.
    Returns the max bin count (for syncing y-lims elsewhere).
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]

    if x_range is None:
        lo, hi = np.percentile(v, [1, 99]) if v.size else (0, 1)
    else:
        lo, hi = x_range

    counts, bins, _ = ax.hist(v, bins=50, range=(lo, hi),
                              color="#E6E6E6", edgecolor="#BBBBBB")
    ax.set_xlim(lo, hi)

    ax.axvline(cutoff, color=color_line, linewidth=1.8, linestyle="--")
    ttl = title if percentile_text is None else f"{title}\n{percentile_text}"
    ax.set_title(ttl, pad=10, color=color_line)
    ax.set_xlabel(x_label)
    ax.set_ylabel("count")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.ticklabel_format(style="plain", axis="y")

    # arrows below
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    xmin, xmax = ax.get_xlim()
    y_arrow, y_text = -0.34, -0.40
    ax.annotate("", xy=(cutoff, y_arrow), xytext=(xmin, y_arrow),
                xycoords=trans, arrowprops=dict(arrowstyle="<-", color="#555", lw=1.2),
                annotation_clip=False)
    ax.text((xmin + cutoff)/2.0, y_text, "negative",
            transform=trans, ha="center", va="center", color="#555", fontsize=8.5)
    ax.annotate("", xy=(xmax, y_arrow), xytext=(cutoff, y_arrow),
                xycoords=trans, arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
                annotation_clip=False)
    ax.text((cutoff + xmax)/2.0, y_text, "positive",
            transform=trans, ha="center", va="center", color="#555", fontsize=8.5)
    return counts.max() if len(counts) else 0.0

# ================================
# Comparison helpers
# ================================
def _collect_sample_values(sample_name: str, ch: int):
    s_norm = sample_name.strip().lower()
    df = features.copy()
    df['sample_name'] = df['sample_name'].str.strip().str.lower()
    sub = df[df['sample_name'] == s_norm]
    return pd.to_numeric(sub[f'channel_{ch}'], errors='coerce').fillna(0).to_numpy(float)

def _pooled_values(sample_names, ch):
    vals = [ _collect_sample_values(s, ch) for s in sample_names ]
    vals = [v for v in vals if v.size]
    return np.concatenate(vals) if vals else np.array([], float)

def plot_compare_combined(fig, parent_spec, sample_names, chA, chB, qA, qB, use_quantile,
                          anchor_maxA=0.0, anchor_maxB=0.0,
                          x_rangeA=None, x_rangeB=None):
    """
    Two stacked histograms (CD31 | F4/80) using pooled values.
    Uses fixed x-ranges if provided, so they match the per-sample histograms.
    """
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=parent_spec, hspace=0.45)

    vA = _pooled_values(sample_names, chA)
    vB = _pooled_values(sample_names, chB)

    for i, (v, col, ch_label, q, anchor, xr) in enumerate([
        (vA, "#cc0000", "CD31", qA, anchor_maxA, x_rangeA),
        (vB, "#0a8a0a", "F4/80", qB, anchor_maxB, x_rangeB),
    ]):
        ax = fig.add_subplot(sub[i, 0])
        if v.size:
            if xr is None:
                lo, hi = np.percentile(v, [1, 99])
            else:
                lo, hi = xr
            counts, bins, _ = ax.hist(v, bins=50, range=(lo, hi),
                                      color="#EFEFEF", edgecolor=col, linewidth=0.8)
            ax.set_xlim(lo, hi)
            cut = np.quantile(v, np.clip(q, 0, 1)) if use_quantile else float(q)
            ax.axvline(cut, color=col, linestyle="--", linewidth=1.2)
            ymax = max(counts.max(), anchor) * 1.05 if counts.size else max(1.0, anchor)
            ax.set_ylim(0, ymax)
        ax.set_xlabel(ch_label)
        ax.set_ylabel("count")
        ax.set_title(f"{ch_label} across samples (pooled)", fontsize=11, color=col, pad=6)
        ax.ticklabel_format(style="plain", axis="y")
        for s in ("top","right"): ax.spines[s].set_visible(False)

# (Optional) facets mode kept available
def plot_compare_facets(fig, parent_spec, sample_names, chA, chB, qA, qB, use_quantile, anchor_maxA=0.0, anchor_maxB=0.0):
    n = len(sample_names)
    sub = GridSpecFromSubplotSpec(n, 2, subplot_spec=parent_spec, wspace=0.25, hspace=0.40)

    # Build common binning from pooled values per channel
    vA_all = _pooled_values(sample_names, chA)
    vB_all = _pooled_values(sample_names, chB)
    binsA = np.linspace(*np.percentile(vA_all, [1, 99]), 40) if vA_all.size else np.linspace(0,1,40)
    binsB = np.linspace(*np.percentile(vB_all, [1, 99]), 40) if vB_all.size else np.linspace(0,1,40)

    # y-lims (count-based), aligned with anchors
    gmaxA, gmaxB = anchor_maxA, anchor_maxB
    if vA_all.size: gmaxA = max(gmaxA, np.histogram(vA_all, bins=binsA)[0].max())
    if vB_all.size: gmaxB = max(gmaxB, np.histogram(vB_all, bins=binsB)[0].max())
    ylimA = (0, gmaxA * 1.05 if gmaxA else 1)
    ylimB = (0, gmaxB * 1.05 if gmaxB else 1)

    for i, sname in enumerate(sample_names):
        axA = fig.add_subplot(sub[i, 0])
        vA = _collect_sample_values(sname, chA)
        if vA.size:
            cutA = np.quantile(vA, np.clip(qA, 0, 1)) if use_quantile else float(qA)
            axA.hist(vA, bins=binsA, color="#f4caca", edgecolor="#cc0000", linewidth=0.7)
            axA.axvline(cutA, color="#cc0000", linestyle="--", linewidth=1.0)
        axA.set_ylim(*ylimA)
        axA.set_ylabel("count" if i == 0 else "")
        axA.set_xlabel("CD31" if i == n-1 else "")
        axA.set_title(short_name(sname, 34), fontsize=9, pad=3)
        axA.ticklabel_format(style="plain", axis="y")
        for s in ("top","right"): axA.spines[s].set_visible(False)

        axB = fig.add_subplot(sub[i, 1])
        vB = _collect_sample_values(sname, chB)
        if vB.size:
            cutB = np.quantile(vB, np.clip(qB, 0, 1)) if use_quantile else float(qB)
            axB.hist(vB, bins=binsB, color="#cfeccd", edgecolor="#0a8a0a", linewidth=0.7)
            axB.axvline(cutB, color="#0a8a0a", linestyle="--", linewidth=1.0)
        axB.set_ylim(*ylimB)
        axB.set_ylabel("")
        axB.set_xlabel("F4/80" if i == n-1 else "")
        axB.set_title("", fontsize=9)
        axB.ticklabel_format(style="plain", axis="y")
        for s in ("top","right"): axB.spines[s].set_visible(False)

# ================================
# Main figure per sample
# ================================
def plot_map_tiles_hist(sample_name: str,
                        chA=CH_CD31, chB=CH_F480, chDapi=CH_DAPI,
                        qA=Q_CD31, qB=Q_F480, use_quantile=USE_QUANTILES,
                        img_size=IMG_SIZE,
                        out_dir="pub_threeclass_triptych_clean",
                        tile_h: int = None,
                        tile_w: int = None,
                        compare_samples=None,
                        compare_mode=None):
    """
    compare_samples: list of samples to include in the comparison block
    compare_mode: None | "combined" | "facets"
    """
    os.makedirs(out_dir, exist_ok=True)
    do_compare = bool(compare_samples) and compare_mode in ("combined", "facets")
    compare_list = []
    if do_compare:
        compare_list = [sample_name] + [s for s in compare_samples if s != sample_name]

    # ----- layout: add a small spacer col before the right block to push it right -----
    extra_cols  = 0
    width_right = []
    if do_compare:
        extra_cols  = 3  # [spacer | compare-left | compare-right(or filler)]
        width_right = [0.32, 1.15, 1.15]  # spacer + two cells; works for combined/facets

    width_ratios = [1.2, 1, 1, 1] + width_right
    fig_w = 14.2 if not do_compare else 18.0  # a bit wider to avoid any overlap

    fig = plt.figure(figsize=(fig_w, 7.2))
    cols = 4 + extra_cols
    gs = GridSpec(2, cols,
                  width_ratios=width_ratios,
                  height_ratios=[1, 0.9],
                  wspace=WSPACE, hspace=HSPACE)

    # Map
    ax_map = fig.add_subplot(gs[:, 0])
    info = plot_threeclass_map(ax_map, features, sample_name, chA, chB, qA, qB, use_quantile)
    if info is None:
        plt.close(fig); return

    # --- compute pooled x-ranges for syncing, if comparing ---
    x_rangeA = x_rangeB = None
    if do_compare:
        pooledA = _pooled_values(compare_list, chA)
        pooledB = _pooled_values(compare_list, chB)
        if pooledA.size:
            x_rangeA = tuple(np.percentile(pooledA, [1, 99]))
        if pooledB.size:
            x_rangeB = tuple(np.percentile(pooledB, [1, 99]))

    # Tile: manual or auto
    if tile_h is not None and tile_w is not None:
        th, tw = int(tile_h), int(tile_w)
        picked_label = "manual"
    else:
        th, tw = pick_tile(info)
        picked_label = "auto"

    arr = load_sample_zarr(ROOT_PATH, sample_name)
    A_gray, B_gray, overlay = render_tile_triptych(arr, th, tw, img_size, chA, chB, chDapi,
                                                   clip_pct=CLIP_PCT_TILE, dapi_gain=DAPI_FAINT)

    # Top row tiles
    axA  = fig.add_subplot(gs[0, 1]); axA.imshow(A_gray, cmap="gray"); axA.set_axis_off()
    axB  = fig.add_subplot(gs[0, 2]); axB.imshow(B_gray, cmap="gray"); axB.set_axis_off()
    axAB = fig.add_subplot(gs[0, 3]); axAB.imshow(overlay);            axAB.set_axis_off()
    axA.set_title(f"{CH_NAMES_PRETTY[chA]} (grayscale)", color="#CC0000", fontsize=11)
    axB.set_title(f"{CH_NAMES_PRETTY[chB]} (grayscale)", color="#008800", fontsize=11)
    axAB.set_title(f"Overlay ({CH_NAMES_PRETTY[chA]}=red, {CH_NAMES_PRETTY[chB]}=green)\n"
                   f"Tile (h={th}, w={tw}) [{picked_label}]", fontsize=10)

    # Bottom row per-sample hist (use synced x-ranges if available)
    axH1 = fig.add_subplot(gs[1, 1])
    axH2 = fig.add_subplot(gs[1, 2])
    pctA_text = f"{int(qA*100)}th percentile" if use_quantile else None
    pctB_text = f"{int(qB*100)}th percentile" if use_quantile else None

    maxA = plot_channel_hist(
        axH1, info["A"], info["tA"],
        f"{CH_NAMES_PRETTY[chA]} feature distribution",
        COL_HIST_CD31, x_label=f"tile feature ({CH_NAMES_PRETTY[chA]})",
        percentile_text=pctA_text, x_range=x_rangeA
    )
    maxB = plot_channel_hist(
        axH2, info["B"], info["tB"],
        f"{CH_NAMES_PRETTY[chB]} feature distribution",
        COL_HIST_F480, x_label=f"tile feature ({CH_NAMES_PRETTY[chB]})",
        percentile_text=pctB_text, x_range=x_rangeB
    )

    # Comparison block on the right
    if do_compare:
        # leave [spacer] empty (gs[:, 4])
        right_parent = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:, 5:], wspace=0.60)

        if compare_mode == "combined":
            # pooled histograms (with same x-ranges as per-sample)
            plot_compare_combined(
                fig, right_parent[0, 0], compare_list, chA, chB, qA, qB, use_quantile,
                anchor_maxA=maxA, anchor_maxB=maxB,
                x_rangeA=x_rangeA, x_rangeB=x_rangeB
            )
            ax_gap = fig.add_subplot(right_parent[0, 1]); ax_gap.axis("off")
        else:
            # facets mode (kept available)
            plot_compare_facets(fig, right_parent[0, 0], compare_list, chA, chB, qA, qB,
                                use_quantile, anchor_maxA=maxA, anchor_maxB=maxB)
            ax_gap = fig.add_subplot(right_parent[0, 1]); ax_gap.axis("off")
    else:
        ax_blank = fig.add_subplot(gs[1, 3]); ax_blank.axis("off")

    fig.subplots_adjust(bottom=BOTTOM_PAD)

    base = f"{safe_name(sample_name)}__{CH_NAMES_SAFE[chA]}_{CH_NAMES_SAFE[chB]}_triptych"
    if do_compare:
        base += "_combined" if compare_mode == "combined" else "_facets"
    out_png = os.path.join(out_dir, base + ".png")
    out_pdf = os.path.join(out_dir, base + ".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[OK] Saved: {out_png}\n[OK] Saved: {out_pdf}")
    plt.close(fig)

# ================================
# Example usage
# ================================
SAMPLES = [
    '20230721_Day 5 dbdb 4_dbdb Day 5 6_Scan6_Stitched.ome',
    '20230721_Day 5 WT 3_WT Day 5 4_Scan3_Stitched.ome',
    '20230721_Day 5 WT 3_WT Day 5 4_Scan3_Stitched_B.ome',
    '20230721_dbdb POD10 Untr 43 R_dbdb Untr POD10 41L_Scan3_Stitched.ome',
    '20230721_dbdb POD10 Untr 43 R_dbdb Untr POD10 41L_Scan3_Stitched_B.ome',
    '20230721_dbdb POD10 Untr B2R_dbdb POD10 Untr B1R_Scan3_Stitched.ome',
    '20230721_dbdb POD10 Untr B2R_dbdb POD10 Untr B1R_Scan3_Stitched_B.ome',
    '20230721_dbdb Untr POD5 39R2_dbdb POD5 9_Scan4_Stitched.ome',
    '20230721_dbdb Untr POD5 39R2_dbdb POD5 9_Scan4_Stitched_B.ome',
    '20230721_POD 10 WT 1_C57 10d 1_Scan2_Stitched.ome',
    '20230721_WT POD 5 6_WT POD 5 7_Scan3_Stitched.ome',
    '20230721_WT POD 5 6_WT POD 5 7_Scan3_Stitched_B.ome',
    '20230721_WT POD10 47_WT POD 10 48_Scan2_Stitched.ome',
    '20230721_WT POD10 47_WT POD 10 48_Scan2_Stitched_B.ome',
]

COMPARE = [
    '20230721_Day 5 dbdb 4_dbdb Day 5 6_Scan6_Stitched.ome',
    '20230721_Day 5 WT 3_WT Day 5 4_Scan3_Stitched.ome',
    '20230721_Day 5 WT 3_WT Day 5 4_Scan3_Stitched_B.ome',
    '20230721_dbdb POD10 Untr 43 R_dbdb Untr POD10 41L_Scan3_Stitched.ome',
    '20230721_dbdb POD10 Untr 43 R_dbdb Untr POD10 41L_Scan3_Stitched_B.ome',
    '20230721_dbdb POD10 Untr B2R_dbdb POD10 Untr B1R_Scan3_Stitched.ome',
    '20230721_dbdb POD10 Untr B2R_dbdb POD10 Untr B1R_Scan3_Stitched_B.ome',
    '20230721_dbdb Untr POD5 39R2_dbdb POD5 9_Scan4_Stitched.ome',
    '20230721_dbdb Untr POD5 39R2_dbdb POD5 9_Scan4_Stitched_B.ome',
    '20230721_POD 10 WT 1_C57 10d 1_Scan2_Stitched.ome',
    '20230721_WT POD 5 6_WT POD 5 7_Scan3_Stitched.ome',
    '20230721_WT POD 5 6_WT POD 5 7_Scan3_Stitched_B.ome',
    '20230721_WT POD10 47_WT POD 10 48_Scan2_Stitched.ome',
    '20230721_WT POD10 47_WT POD 10 48_Scan2_Stitched_B.ome',
]

for s in SAMPLES:
    plot_map_tiles_hist(
        s,
        chA=CH_CD31, chB=CH_F480, chDapi=CH_DAPI,
        qA=0.95, qB=0.15, use_quantile=True,
        img_size=128,
        out_dir="pub_threeclass_triptych_clean",
        compare_samples=COMPARE,
        compare_mode="combined"   # "combined" (pooled), "facets", or None
    )
