#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

# =========================
# Config
# =========================
ROOT_PATH = 'data/ome_tiffs'
TILE_SIZE = 128

# Per-tile counts:
COUNT_MODE       = "sum_all"     # "sum_all" or "single"
COUNT_CHANNEL    = 0             # used only if COUNT_MODE == "single"
AF_CHANNEL_INDEX = 6             # 0-based index of autofluorescence channel to exclude

# Plotting behavior
PLOT_MODE        = "sorted"      # "sorted" or "unsorted"
NORMALIZE_Y      = True          # normalize each sample to its max=1
Y_TRANSFORM      = "none"        # 'none' | 'sqrt' | 'logit'  (applied after normalization)
SMOOTH_WINDOW    = 0             # moving-median window (in points); 0 disables
SHOW_INDIVIDUALS = True          # faint lines for each sample
POINT_SUBSAMPLE  = 0             # 0=plot lines; >0 -> plot markers every Nth point (e.g., 50 => 2% points)
JITTER_X         = 0.0           # small jitter for markers (e.g., 0.15); only used when POINT_SUBSAMPLE>0

# Cohort ribbon (trend) options
SHOW_COHORT_RIBBONS = True
PCT_GRID_POINTS     = 500        # resample each sample to this many percentiles
RIBBON_ALPHA_FILL   = 0.18
RIBBON_ALPHA_EDGE   = 0.9

# Output
OUT_BASENAME = f"tile_counts_{PLOT_MODE}_{'norm' if NORMALIZE_Y else 'raw'}_{Y_TRANSFORM}"
OUT_PNG = f"{OUT_BASENAME}.png"
OUT_PDF = f"{OUT_BASENAME}.pdf"
OUT_SVG = f"{OUT_BASENAME}.svg"

# Your samples
SAMPLE_NAMES = [
    'sample_1.ome',
    'sample_2.ome',
    'sample_3.ome',
    'sample_4.ome',
    'sample_5.ome',
    'sample_6.ome',
    'sample_7.ome',
    'sample_8.ome',
    'sample_9.ome',
    'sample_10.ome',
    'sample_11.ome',
    'sample_12.ome',
    'sample_13.ome',
    'sample_14.ome',
    'sample_15.ome',
    'sample_16.ome'
]

# =========================
# Cohorts & palette
# =========================
cohort_order = ['Day 5 WT', 'Day 5 dbdb', 'Day 10 WT', 'Day 10 dbdb']
CHORD_COLORS = {
    'Day 5 WT':   '#c8cbc9',
    'Day 5 dbdb': '#FEA0A0',
    'Day 10 WT':  '#76817a',
    'Day 10 dbdb':'#e24a4a',
}

# =========================
# Helpers
# =========================
day5_pat  = re.compile(r'\b(day\s*5|pod\s*5|pod5|5d|day5)\b', re.I)
day10_pat = re.compile(r'\b(day\s*10|pod\s*10|pod10|10d|day10)\b', re.I)

def extract_day_group(name: str) -> str:
    if day5_pat.search(name):  return 'Day 5'
    if day10_pat.search(name): return 'Day 10'
    return 'Unknown'

def assign_genotype(name: str) -> str:
    n = name.lower()
    if 'dbdb' in n: return 'dbdb'
    if 'wt'   in n: return 'WT'
    return 'Unknown'

def find_positions_csv(sample_name: str, root: str, tile_size: int) -> Optional[str]:
    """Find positions_<tile_size>.csv/.sv under sample/tiles/."""
    tdir = os.path.join(root, sample_name, "tiles")
    patterns = [f"positions_{tile_size}.csv", f"positions_{tile_size}.sv", f"positions_{tile_size}.*"]
    for pat in patterns:
        for path in glob.glob(os.path.join(tdir, pat)):
            if os.path.isfile(path):
                return path
    return None

def compute_tile_counts(sample_name: str, root: str, tile_size: int) -> np.ndarray:
    """
    Per-tile counts via integral image (summed-area table).
    Count = # of non-zero pixels within each tile.
    - "sum_all": union across all channels except AF
    - "single":  one channel (COUNT_CHANNEL)
    """
    zarr_path = os.path.join(root, sample_name, "data.zarr")
    pos_path = find_positions_csv(sample_name, root, tile_size)
    if not pos_path:
        raise FileNotFoundError(f"[{sample_name}] positions_{tile_size}.csv/.sv not found under tiles/")
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"[{sample_name}] Zarr not found: {zarr_path}")

    data = zarr.open(zarr_path, mode='r')  # expect C x H x W

    pos_df = pd.read_csv(pos_path)
    if not {'w','h'}.issubset(pos_df.columns):
        pos_df = pd.read_csv(pos_path, index_col=0)
    x0 = pos_df['w'].astype(int).to_numpy()
    y0 = pos_df['h'].astype(int).to_numpy()
    x1 = x0 + tile_size
    y1 = y0 + tile_size

    C, H, W = data.shape
    if COUNT_MODE == "sum_all":
        use_channels = [c for c in range(C) if c != AF_CHANNEL_INDEX]
        tissue_mask = np.zeros((H, W), dtype=bool)
        for ch in use_channels:
            arr = data[ch][:]      # sequential read per channel
            tissue_mask |= (arr > 0)
    else:
        arr = data[COUNT_CHANNEL][:]
        tissue_mask = (arr > 0)

    # Integral image S (H+1 x W+1)
    S = np.zeros((H + 1, W + 1), dtype=np.int64)
    S[1:, 1:] = tissue_mask.astype(np.uint8)
    S = S.cumsum(axis=0).cumsum(axis=1)

    # Vectorized rectangle sums for all tiles
    counts = (S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]).astype(np.int64)

    out_csv = os.path.join(root, sample_name, f"tiles/tile_counts_{tile_size}.csv")
    pd.DataFrame({'count': counts}).to_csv(out_csv, index=False)
    return counts

def get_or_make_counts(sample_name: str, root: str, tile_size: int) -> np.ndarray:
    """Load cached counts if present; otherwise compute & save."""
    out_csv = os.path.join(root, sample_name, f"tiles/tile_counts_{tile_size}.csv")
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        if 'count' in df.columns and np.issubdtype(df['count'].dtype, np.number):
            return df['count'].to_numpy()
    return compute_tile_counts(sample_name, root, tile_size)

def color_for(cohort: str) -> str:
    return CHORD_COLORS.get(str(cohort), '#666666')

def apply_y_transform(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'none':
        return y
    if mode == 'sqrt':
        return np.sqrt(np.clip(y, 0, None))
    if mode == 'logit':
        eps = 1e-4
        yy = np.clip(y, eps, 1 - eps)
        return np.log(yy / (1 - yy))
    raise ValueError(f"Unknown Y_TRANSFORM: {mode}")

def moving_median(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    w = int(window)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode='edge')
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = np.median(yp[i:i+w])
    return out

def percentile_curve(yvals: np.ndarray, n_grid: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a sorted curve to a 0..1 percentile grid with interpolation."""
    n = len(yvals)
    if n == 0:
        q = np.linspace(0, 1, n_grid)
        return q, np.zeros_like(q)
    # positions at centers
    p = (np.arange(n) + 0.5) / n
    q = np.linspace(0, 1, n_grid)
    yy = np.interp(q, p, yvals)
    return q, yy

# =========================
# Build metadata
# =========================
rows = []
for s in SAMPLES_NAMES:
    rows.append({
        'sample_name': s,
        'day_group': extract_day_group(s),
        'genotype': assign_genotype(s)
    })
meta = pd.DataFrame(rows)
meta['cohort'] = meta['day_group'].astype(str) + ' ' + meta['genotype'].astype(str)
meta = meta[meta['cohort'].isin(cohort_order)].copy()
meta['cohort'] = pd.Categorical(meta['cohort'], cohort_order, ordered=True)

missing = set(meta['cohort'].astype(str).unique()) - set(CHORD_COLORS.keys())
if missing:
    print("[WARN] Missing color for cohorts:", missing, "\nEdit CHORD_COLORS keys to match exactly.")

# =========================
# Prepare data
# =========================
all_samples: List[Dict] = []
for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Loading counts"):
    cohort = str(row['cohort'])
    sname  = row['sample_name']
    try:
        counts = get_or_make_counts(sname, ROOT_PATH, TILE_SIZE)
    except Exception as e:
        print(f"[WARN] {e}")
        continue

    yvals = np.sort(counts)[::-1] if PLOT_MODE == "sorted" else counts.astype(float)

    if NORMALIZE_Y:
        maxv = yvals.max() if yvals.size else 1.0
        if maxv == 0: maxv = 1.0
        yvals = yvals / float(maxv)

    if SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
        yvals = moving_median(yvals, SMOOTH_WINDOW)

    yvals = apply_y_transform(yvals, Y_TRANSFORM)

    all_samples.append({
        'sample': sname,
        'cohort': cohort,
        'y': yvals
    })

# =========================
# Compute cohort ribbons on percentile grid
# =========================
ribbons = {}
if SHOW_COHORT_RIBBONS:
    for cohort in cohort_order:
        ys = [d['y'] for d in all_samples if d['cohort'] == cohort]
        if not ys:
            continue
        # Resample each sample to common percentiles
        grid = np.linspace(0, 1, PCT_GRID_POINTS)
        Y = []
        for y in ys:
            _, yy = percentile_curve(y, n_grid=PCT_GRID_POINTS)
            Y.append(yy)
        Y = np.vstack(Y)
        mu = np.nanmean(Y, axis=0)
        sd = np.nanstd(Y, axis=0)
        # 95% CI assuming approx normal over samples (n small is okay for a ribbon)
        n = Y.shape[0]
        se = sd / max(n**0.5, 1.0)
        lo = mu - 1.96 * se
        hi = mu + 1.96 * se
        ribbons[cohort] = {'grid': grid, 'mu': mu, 'lo': lo, 'hi': hi}

# =========================
# Plot
# =========================
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(10.5, 6.5))

# 1) Individual samples (faint)
if SHOW_INDIVIDUALS:
    for d in all_samples:
        y = d['y']
        x = np.arange(1, len(y) + 1)
        if POINT_SUBSAMPLE and POINT_SUBSAMPLE > 0:
            keep = np.arange(0, len(y), POINT_SUBSAMPLE)
            xs = x[keep].astype(float)
            ys = y[keep]
            if JITTER_X and JITTER_X > 0:
                xs = xs + np.random.uniform(-JITTER_X, JITTER_X, size=xs.shape)
            ax.scatter(xs, ys, s=8, alpha=0.25, lw=0, color=color_for(d['cohort']))
        else:
            ax.plot(x, y, lw=1.0, alpha=0.25, color=color_for(d['cohort']))

# 2) Cohort ribbons (prominent)
for cohort in cohort_order:
    if cohort not in ribbons:
        continue
    col = color_for(cohort)
    grid = ribbons[cohort]['grid']      # 0..1 percentiles
    mu   = ribbons[cohort]['mu']
    lo   = ribbons[cohort]['lo']
    hi   = ribbons[cohort]['hi']
    # Map percentile grid to x in "rank space"
    # Use a representative length: median N for this cohort
    Ns = [len(d['y']) for d in all_samples if d['cohort'] == cohort]
    Nmed = int(np.median(Ns)) if Ns else 100
    xg = 1 + grid * (Nmed - 1)
    ax.fill_between(xg, lo, hi, color=col, alpha=RIBBON_ALPHA_FILL, linewidth=0)
    ax.plot(xg, mu, color=col, lw=2.2, alpha=RIBBON_ALPHA_EDGE)

# Labels / title
x_label_core = 'Tiles (ranked by count)' if PLOT_MODE == "sorted" else 'Tile index'
y_label = {
    'none': 'Normalized counts' if NORMALIZE_Y else 'Counts',
    'sqrt': '√(normalized counts)' if NORMALIZE_Y else '√(counts)',
    'logit': 'logit(normalized counts)'  # only meaningful when normalized
}[Y_TRANSFORM]
ax.set_xlabel(x_label_core)
ax.set_ylabel(y_label)

title_bits = []
title_bits.append('Per-tile counts')
if PLOT_MODE == "sorted": title_bits.append('(ranked)')
if NORMALIZE_Y: title_bits.append('normalized=TRUE')
if Y_TRANSFORM != 'none': title_bits.append(f'y-transform={Y_TRANSFORM}')
if SMOOTH_WINDOW and SMOOTH_WINDOW > 1: title_bits.append(f'smooth={SMOOTH_WINDOW}')
#ax.set_title(' — '.join(title_bits))
ax.set_title("Comparative Tile-Level Pixel Count Profiles Across Cohorts", fontsize=13, fontweight='bold')


# Nice ticks & grid
ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
ax.grid(True, axis='y', alpha=0.15, linewidth=0.8)

# Legend in enforced order
handles = [plt.Line2D([0],[0], color=color_for(c), lw=3) for c in cohort_order]
ax.legend(handles, cohort_order, title='Cohort', frameon=False, loc='upper right')

plt.tight_layout()
for fn in (OUT_PNG, OUT_PDF, OUT_SVG):
    plt.savefig(fn, dpi=300 if fn.endswith('.png') else None, bbox_inches='tight')
plt.show()

print(f"[✓] Plotted {len(all_samples)} samples")
print(f"[✓] Saved: {OUT_PNG}")
print(f"[✓] Saved: {OUT_PDF}")
print(f"[✓] Saved: {OUT_SVG}")

