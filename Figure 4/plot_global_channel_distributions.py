# ============================================================
# Global tile-based intensity distributions + normalization
# Self-contained (no external utils)
# - Robust channel parsing (0/1-based; "Channel N" strings OK)
# - Detects channel axis (C,H,W or H,W,C)
# - Auto-detects tile coord orientation: h,w -> (x,y) or (y,x)
# - Renames CSV "Channel N" -> biological names (DAPI, aSMA, ...)
# - Contiguous, edge-aligned bars on log(Frequency) (no gaps)
# - FULL-RANGE x-axes (no percentile clipping)
# Outputs (QC_PATH/normalization/bars):
#   - global_raw_bars.pdf        (marker titles, RAW)
#   - channel_hist_log.pdf       ("Channel 1..7" titles, RAW, same axes/colors)
#   - global_zscore_bars.pdf     (marker titles, Z-score)
# Also writes normalization_stats.csv
# ============================================================

import os, re
import numpy as np
import pandas as pd
import zarr
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------
# Config
# --------------------------
ROOT_PATH = 'data/ome_tiffs'
QC_PATH   = 'outputs/qc'
TILE_SIZE = 128
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

# Canonical order + colors (used for marker-titled plots)
ORDERED_MARKERS = ['DAPI', 'aSMA', 'VIM', 'F4/80', 'CD31', 'KI-67', 'AF']
MARKER_COLORS = {
    'DAPI':  '#0000FF',
    'aSMA':  '#00FFFF',
    'VIM':   '#FF00FF',
    'F4/80': '#00FF00',
    'CD31':  '#FF0000',
    'KI-67': '#FFFF00',
    'AF':    '#FFFFFF', #white might be better to change to grey!! 
}
def _fallback_color(i): return plt.get_cmap('tab20').colors[i % 20]

# Map CSV 'marker' values like "Channel 1" -> biological names
CHANNEL_TO_MARKER = {
    "Channel 1": "DAPI",
    "Channel 2": "aSMA",
    "Channel 3": "VIM",
    "Channel 4": "F4/80",
    "Channel 5": "CD31",
    "Channel 6": "KI-67",
    "Channel 7": "AF",
    # digits-only fallback if ever needed
    "1": "DAPI", "2": "aSMA", "3": "VIM", "4": "F4/80", "5": "CD31", "6": "KI-67", "7": "AF",
}

# Axes range control (None => full range)
RAW_CLIP_PERCENTILES = None
ZSCORE_CLIP_PERCENTILES = None   # set to (-5, 5) if you want fixed z-score axis later

# --------------------------
# Matplotlib style
# --------------------------
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'figure.dpi': 300,
    'image.resample': False,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.5,
    'savefig.dpi': 300,
    'savefig.transparent': False,
    'savefig.bbox': 'tight',
    'legend.frameon': False,
})

# --------------------------
# IO helpers
# --------------------------
def _find_first_3d_array(znode):
    if hasattr(znode, 'ndim') and znode.ndim == 3: return znode
    if hasattr(znode, 'arrays'):
        for _, arr in znode.arrays():
            if arr.ndim == 3: return arr
    if hasattr(znode, 'groups'):
        for _, grp in znode.groups():
            arr = _find_first_3d_array(grp)
            if arr is not None: return arr
    return None

def load_zarr_w_channel(root_path, sample_name):
    zpath = os.path.join(root_path, sample_name, 'data.zarr')
    cpath = os.path.join(root_path, sample_name, 'channels.csv')
    if not os.path.exists(zpath): raise FileNotFoundError(zpath)
    if not os.path.exists(cpath): raise FileNotFoundError(cpath)
    znode = zarr.open(zpath, mode='r')
    arr3d = _find_first_3d_array(znode) or znode
    if getattr(arr3d, 'ndim', None) != 3:
        raise ValueError(f"Could not find 3D array in {zpath}")
    channels = pd.read_csv(cpath)
    if not {'marker','channel'}.issubset(channels.columns):
        raise KeyError(f"{cpath} must have columns ['marker','channel']")
    return arr3d, channels

def load_tile_info(root_path, sample_name, tile_size):
    tiles_dir = os.path.join(root_path, sample_name, 'tiles')
    want = os.path.join(tiles_dir, f'positions_{tile_size}.csv')
    if os.path.exists(want):
        df = pd.read_csv(want)
    else:
        if not os.path.isdir(tiles_dir):
            raise FileNotFoundError(f"No tiles dir: {tiles_dir}")
        cand = [f for f in os.listdir(tiles_dir) if f.startswith('positions_') and f.endswith('.csv')]
        if not cand: raise FileNotFoundError(f"No positions_*.csv in {tiles_dir}")
        pick = sorted(cand)[0]
        print(f"[auto] Using {pick} (requested positions_{tile_size}.csv not found)")
        df = pd.read_csv(os.path.join(tiles_dir, pick))
    if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'])
    if not {'h','w'}.issubset(df.columns):
        raise KeyError(f"positions csv must have 'h' and 'w', got {df.columns.tolist()}")
    return df

# --------------------------
# Channel / tiles logic
# --------------------------
def detect_channel_axis(arr3d):
    if arr3d.shape[0] <= 64 and arr3d.shape[1] >= TILE_SIZE and arr3d.shape[2] >= TILE_SIZE:
        return 'first'  # C,H,W
    if arr3d.shape[-1] <= 64 and arr3d.shape[0] >= TILE_SIZE and arr3d.shape[1] >= TILE_SIZE:
        return 'last'   # H, W, C
    return 'first'

def parse_channel_index(ch_val, n_channels):
    """Accept 0/1-based ints or 'Channel N' strings → normalize to 0-based."""
    if isinstance(ch_val, (int, np.integer, float, np.floating)):
        idx = int(ch_val)
    else:
        m = re.search(r'\d+', str(ch_val))
        if not m: raise ValueError(f"Unparsable channel: {ch_val}")
        idx = int(m.group())
    if idx >= n_channels: idx -= 1
    if not (0 <= idx < n_channels) and 1 <= idx <= n_channels: idx -= 1
    if not (0 <= idx < n_channels):
        raise IndexError(f"Channel index {idx} out of range (n_channels={n_channels})")
    return idx

def get_tile_region(channel_img, root_path, sample_name, tile_size):
    df = load_tile_info(root_path, sample_name, tile_size)
    H, W = channel_img.shape[:2]
    def count_valid(xcol, ycol):
        xs = df[xcol].astype(int).values; ys = df[ycol].astype(int).values
        okx = (xs >= 0) & (xs + tile_size <= W)
        oky = (ys >= 0) & (ys + tile_size <= H)
        return int(np.sum(okx & oky))
    n_xy = count_valid('h','w')   # h->x, w->y
    n_yx = count_valid('w','h')   # h->y, w->x
    if n_xy == 0 and n_yx == 0:
        print(f"[tiles] {sample_name}: no valid tiles with TILE_SIZE={tile_size} (img {H}x{W})")
        return np.array([], dtype=float)
    use_h_as_x = (n_xy >= n_yx)
    if use_h_as_x:
        xs = df['h'].astype(int).values; ys = df['w'].astype(int).values
        chosen = f"h->x, w->y ({n_xy}/{len(df)} valid)"
    else:
        xs = df['w'].astype(int).values; ys = df['h'].astype(int).values
        chosen = f"h->y, w->x ({n_yx}/{len(df)} valid)"
    print(f"[tiles] {sample_name}: using {chosen}")
    ok = (xs >= 0) & (ys >= 0) & (xs + tile_size <= W) & (ys + tile_size <= H)
    xs = xs[ok]; ys = ys[ok]
    if xs.size == 0:
        print(f"[tiles] {sample_name}: all candidate tiles out of bounds after filtering")
        return np.array([], dtype=float)
    parts = [channel_img[y:y+tile_size, x:x+tile_size].ravel() for x,y in zip(xs,ys)]
    return np.concatenate(parts).ravel() if parts else np.array([], dtype=float)

def get_channel_data(root_path, sample_name, sample_max, tile_size=TILE_SIZE, verbose=True):
    data3d, channels = load_zarr_w_channel(root_path, sample_name)
    axis = detect_channel_axis(data3d)
    C = data3d.shape[0] if axis == 'first' else data3d.shape[-1]
    if verbose:
        print(f"[zarr] {sample_name}: shape={tuple(data3d.shape)} axis={axis} C={C}")

    marker_channel = dict(zip(channels['marker'], channels['channel']))
    out = {}
    for csv_marker, ch_val in marker_channel.items():
        marker_pretty = CHANNEL_TO_MARKER.get(str(csv_marker), str(csv_marker))
        idx = parse_channel_index(ch_val, C)
        plane = data3d[idx, :, :] if axis == 'first' else data3d[:, :, idx]
        vec = get_tile_region(plane, root_path, sample_name, tile_size)

        if vec.size == 0 and verbose:
            print(f"   ! WARNING: {marker_pretty} has 0 tiles in {sample_name}")

        if vec.size > sample_max:
            vec = vec[np.random.choice(vec.size, size=sample_max, replace=False)]
        out[marker_pretty] = vec.astype(float, copy=False)

        if verbose:
            print(f"   - {marker_pretty} (csv='{csv_marker}', ch={ch_val}→idx={idx}): +{vec.size} values")
    return out

# --------------------------
# Pool + cache
# --------------------------
def gen_global_dist_data(root_path, sample_list, qc_save_path,
                         sample_max=200000, tile_size=TILE_SIZE,
                         force_rebuild=False, verbose=True):
    os.makedirs(qc_save_path, exist_ok=True)
    save_file = os.path.join(qc_save_path, f'global_hist_tiled_{tile_size}.npz')

    def _nonempty(d): return isinstance(d, dict) and any(np.asarray(v).size>0 for v in d.values())

    if (not force_rebuild) and os.path.exists(save_file):
        if verbose: print(f"[cache] Loading {save_file}")
        data = np.load(save_file, allow_pickle=True)
        cached = {k: data[k] for k in data.files}
        if _nonempty(cached):
            if verbose: print(f"[cache] OK: {len(cached)} markers")
            return cached
        print("[cache] Empty cache → rebuilding")

    merged = {}
    total = max(1, len(sample_list))
    for i, s in enumerate(sample_list, 1):
        try:
            print(f"[{i}/{total}] {s}")
            sub_max = max(1, sample_max // total)
            d = get_channel_data(root_path, s, sub_max, tile_size, verbose=True)
            for mk, arr in d.items():
                merged.setdefault(mk, []).extend(arr)
            preview = ", ".join(f"{mk}:{len(merged.get(mk,[]))}" for mk in ORDERED_MARKERS if mk in merged)
            if preview: print(f"    pooled → {preview}")
        except Exception as e:
            print(f"[WARN] Skipping {s}: {e}")

    merged = {k: np.asarray(v) for k,v in merged.items()}
    if not _nonempty(merged):
        raise RuntimeError("Global pool is empty. Check channels.csv mapping & tile CSVs.")
    np.savez(save_file, **merged)
    print(f"[cache] Wrote {save_file} ({len(merged)} markers)")
    return merged

# --------------------------
# Normalization (z-score)
# --------------------------
def normalize_channels(data_dict):
    out, stats = {}, []
    for k, v in data_dict.items():
        a = np.asarray(v, float); a = a[np.isfinite(a)]
        mu = float(a.mean()) if a.size else 0.0
        sd = float(a.std())  if a.size else 1.0
        out[k] = (a - mu)/sd if sd>0 else (a - mu)
        stats.append({'marker': k, 'mean': mu, 'std': sd})
    return out, pd.DataFrame(stats)

# --------------------------
# Shared axes + plotting
# --------------------------
def compute_shared_bins_and_ylim(data_dict, bins=200, clip_percentiles=None):
    """
    Shared histogram bins (x) and ymax for log10(count+1).
    If clip_percentiles is None -> use full data range [min, max] across all channels.
    """
    vecs = []
    for v in data_dict.values():
        a = np.asarray(v, float); a = a[np.isfinite(a)]
        if a.size: vecs.append(a)
    if not vecs:
        return np.linspace(0, 1, bins+1), 1.0

    allv = np.concatenate(vecs)
    if clip_percentiles is None:
        lo = float(allv.min())
        hi = float(allv.max())
    else:
        lo = float(np.percentile(allv, clip_percentiles[0]))
        hi = float(np.percentile(allv, clip_percentiles[1]))
    if lo == hi:
        pad = 1.0 if lo == 0 else abs(lo)*0.05
        lo, hi = lo - pad, hi + pad
    edges = np.linspace(lo, hi, bins+1)

    ymax = 0.0
    for vals in data_dict.values():
        vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
        if vals.size:
            counts, _ = np.histogram(vals, bins=edges)
            logc = np.log10(counts + 1.0)
            if np.isfinite(logc).any():
                ymax = max(ymax, float(np.nanmax(logc)))
    return edges, (ymax*1.05 if ymax > 0 else 1.0)

def plot_hist_bars_pdfstyle(
    data_dict, save_path, bins=200, title='Global',
    panel_titles='marker', xlabel='Marker Intensity',
    include_all=True, keys_override=None, titles_override=None,
    shared_axes_override=None
):
    """
    Small-multiples histograms with shared axes (contiguous, edge-aligned bars).
    """
    # Choose order
    if keys_override is not None:
        keys = [k for k in keys_override]
    else:
        present = list(data_dict.keys())
        extras  = [m for m in sorted(present) if m not in ORDERED_MARKERS]
        keys = (ORDERED_MARKERS + extras) if include_all else present

    data_view = {k: data_dict.get(k, np.array([])) for k in keys}

    # Shared axes
    if shared_axes_override is not None:
        edges, ymax = shared_axes_override
    else:
        edges, ymax = compute_shared_bins_and_ylim(data_view, bins=bins)

    # Layout
    n = len(keys)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(6.5, max(3.0, n*1.0)), sharex=True)
    if n == 1: axes = np.array([axes])

    for i, marker in enumerate(keys):
        ax = axes[i]
        vals = np.asarray(data_view.get(marker, np.array([])), float)
        vals = vals[np.isfinite(vals)]
        counts,_ = np.histogram(vals, bins=edges) if vals.size else (np.zeros(len(edges)-1, int), edges)
        logc = np.log10(counts + 1.0)

        # contiguous, edge-aligned bars (no gaps)
        ax.bar(edges[:-1], logc, width=np.diff(edges), align='edge',
               color=MARKER_COLORS.get(marker, _fallback_color(i)),
               alpha=0.85, edgecolor='none')

        title_txt = titles_override.get(marker) if titles_override else (f"Channel {i+1}" if panel_titles=='channel' else marker)
        ax.set_title(title_txt, loc='left', fontsize=9, pad=2)

        if counts.sum()==0:
            ax.text(0.02, 0.8, 'No data', transform=ax.transAxes, fontsize=8, alpha=0.8)

        ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, ymax)
        ax.set_ylabel('log(Frequency)'); ax.set_xlabel(xlabel)
        ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
        ax.grid(False); ax.tick_params(axis='both', length=2, width=0.5)

    fig.suptitle(title, y=0.995, fontsize=10)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout(rect=[0,0,1,0.985]); fig.savefig(save_path, bbox_inches='tight'); plt.close(fig)
    print(f"[✓] Saved → {save_path}")

# --------------------------
# Driver
# --------------------------
def global_bars(root_path, sample_list, qc_path, bins=200, sample_max=400000,
                tile_size=TILE_SIZE, force_rebuild=True):
    norm_dir = os.path.join(qc_path, 'normalization')
    bars_dir = os.path.join(norm_dir, 'bars')
    os.makedirs(bars_dir, exist_ok=True)

    pooled_raw = gen_global_dist_data(root_path, sample_list, norm_dir,
                                      sample_max=sample_max, tile_size=tile_size,
                                      force_rebuild=force_rebuild, verbose=True)

    # pooled summary (watch AF here)
    print("\n[pooled counts]")
    for k in sorted(pooled_raw.keys()):
        print(f"  {k}: {len(pooled_raw[k])}")

    pooled_z, stats_df = normalize_channels(dict(pooled_raw))
    stats_df.to_csv(os.path.join(norm_dir, 'normalization_stats.csv'), index=False)
    print("[✓] normalization_stats.csv written\n")

    # Build channel-order keys + titles (Channel 1..7) while keeping marker colors
    channel_numbers = list(range(1, 8))  # 1..7
    channel_keys = []
    titles_override = {}
    for n in channel_numbers:
        mk = CHANNEL_TO_MARKER.get(f"Channel {n}", CHANNEL_TO_MARKER.get(str(n), f"Channel{n}"))
        if mk not in channel_keys: channel_keys.append(mk)
        titles_override[mk] = f"Channel {n}"

    # RAW: full-range axes shared across both RAW figures
    raw_edges, raw_ymax = compute_shared_bins_and_ylim(pooled_raw, bins=bins, clip_percentiles=RAW_CLIP_PERCENTILES)
    marker_keys = [m for m in ORDERED_MARKERS if m in pooled_raw] + [m for m in sorted(pooled_raw.keys()) if m not in ORDERED_MARKERS]

    plot_hist_bars_pdfstyle(
        pooled_raw, os.path.join(bars_dir, 'global_raw_bars.pdf'),
        bins=bins, title='Global (raw)', panel_titles='marker', xlabel='Marker Intensity',
        include_all=False, keys_override=marker_keys, shared_axes_override=(raw_edges, raw_ymax)
    )
    plot_hist_bars_pdfstyle(
        pooled_raw, os.path.join(bars_dir, 'channel_hist_log.pdf'),
        bins=bins, title='Global (raw)', panel_titles='channel', xlabel='Marker Intensity',
        include_all=True, keys_override=channel_keys, titles_override=titles_override,
        shared_axes_override=(raw_edges, raw_ymax)
    )

    # Z-SCORE: full-range (set ZSCORE_CLIP_PERCENTILES to (-5,5) if you want fixed)
    z_edges, z_ymax = compute_shared_bins_and_ylim(pooled_z, bins=bins, clip_percentiles=ZSCORE_CLIP_PERCENTILES)
    plot_hist_bars_pdfstyle(
        pooled_z, os.path.join(bars_dir, 'global_zscore_bars.pdf'),
        bins=bins, title='Global (z-score)', panel_titles='marker', xlabel='Intensity (z-score)',
        include_all=False, keys_override=marker_keys, shared_axes_override=(z_edges, z_ymax)
    )

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # Set force_rebuild=True to regenerate cached distributions.
    # To start from scratch, delete:
    # outputs/qc/normalization/global_hist_tiled_128.npz

    global_bars(
        ROOT_PATH,
        SAMPLE_NAMES,
        QC_PATH,
        bins=200,
        sample_max=400000,
        tile_size=TILE_SIZE,
        force_rebuild=True
    )

    print("[✓] Done.")
