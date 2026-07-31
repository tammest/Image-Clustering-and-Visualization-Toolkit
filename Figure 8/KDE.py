#this code is to those KDE points that we are making for each phenotype so for POD 5 and POD 10 with respective treatment - we also calculate the best grid points (optimization) 


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize

# =========================
# FILES
# =========================
CSV_IN = "rabbanilab/wsi_analysis/input_csv_files/n_of_3_with_rotated_coords_and_wound_info.csv"
OUTDIR = Path("kde_mode_output_granulation_only")
OUTDIR.mkdir(exist_ok=True, parents=True)
OUTCSV = OUTDIR / "kde_modes_table.csv"

# =========================
# COLUMNS
# =========================
xcol, ycol = "UMAP1", "UMAP2"
SUPERCLUSTER_COL = "superclusters"
GROUP_COL = "sample_group"

# only keep granulation tissue supercluster
GRAN_MATCH = r"granulation\s*tissue"

# =========================
# GROUP MAPPING
# =========================
GROUP_MAP = {
    "Wild-type POD 5":         "WT_D5",
    "Wild-type POD 10":        "WT_D10",
    "Diabetic POD 5":          "dbdb_D5",
    "Diabetic POD 10":         "dbdb_D10",
    "Diabetic + Exo POD 5":    "dbdb_Exo_D5",
    "Diabetic + Exo POD 10":   "dbdb_Exo_D10",
}

GROUPS = [
    "WT_D5",
    "WT_D10",
    "dbdb_D5",
    "dbdb_D10",
    "dbdb_Exo_D5",
    "dbdb_Exo_D10",
]

# =========================
# KDE SETTINGS
# =========================
BANDWIDTH = 0.5
GRID_SIZE = 90
MIN_POINTS = 30

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_IN, low_memory=False)

df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
df[ycol] = pd.to_numeric(df[ycol], errors="coerce")

req_cols = [xcol, ycol, SUPERCLUSTER_COL, GROUP_COL]
df = df.dropna(subset=req_cols).copy()

df["group_label"] = df[GROUP_COL].map(GROUP_MAP)

if df["group_label"].isna().any():
    bad = df.loc[df["group_label"].isna(), GROUP_COL].value_counts()
    raise ValueError(f"Unmapped sample_group values found:\n{bad}")

# =========================
# FILTER TO GRANULATION TISSUE ONLY
# =========================
gran_mask = df[SUPERCLUSTER_COL].astype(str).str.contains(
    GRAN_MATCH, case=False, na=False, regex=True
)
df_gran = df[gran_mask].copy()

print("[info] granulation rows:", len(df_gran))

if len(df_gran) == 0:
    raise ValueError(
        f"[error] No rows matched GRAN_MATCH={GRAN_MATCH!r} "
        f"in column {SUPERCLUSTER_COL!r}."
    )

# =========================
# KDE MODE FUNCTION
# =========================
def kde_mode(df_in, group, bw=0.5, grid_size=90, min_points=30):
    sub = df_in[df_in["group_label"] == group].copy()

    if len(sub) < min_points:
        print(f"[warn] KDE {group}: too few points ({len(sub)}), skipping.")
        return None

    coords = sub[[xcol, ycol]].to_numpy()

    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(coords)

    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()

    # Step 1: coarse grid search to find a good starting point
    xg = np.linspace(xmin, xmax, grid_size)
    yg = np.linspace(ymin, ymax, grid_size)
    Xg, Yg = np.meshgrid(xg, yg)
    pts = np.vstack([Xg.ravel(), Yg.ravel()]).T

    log_dens = kde.score_samples(pts)
    start = pts[np.argmax(log_dens)]

    # Step 2: continuous optimization from the best grid point
    # We maximize KDE log-density by minimizing negative log-density.
    def objective(z):
        z = np.asarray(z).reshape(1, -1)
        return -kde.score_samples(z)[0]

    opt = minimize(
        objective,
        x0=start,
        method="L-BFGS-B",
        bounds=[(xmin, xmax), (ymin, ymax)],
    )

    if opt.success:
        xm, ym = opt.x
        optimization_success = True
    else:
        print(f"[warn] KDE {group}: optimization failed; using grid mode.")
        xm, ym = start
        optimization_success = False

    return {
        "group_label": group,
        "kde_x": float(xm),
        "kde_y": float(ym),
        "n_tiles": int(len(sub)),
        "bandwidth": float(bw),
        "grid_size": int(grid_size),
        "optimization_success": optimization_success,
    }

# =========================
# COMPUTE KDE MODES
# =========================
rows = []

for g in GROUPS:
    result = kde_mode(
        df_gran,
        g,
        bw=BANDWIDTH,
        grid_size=GRID_SIZE,
        min_points=MIN_POINTS,
    )

    if result is not None:
        rows.append(result)
        print(
            f"[i] {g}: mode=({result['kde_x']:.4f}, {result['kde_y']:.4f}), "
            f"n={result['n_tiles']}, "
            f"optimized={result['optimization_success']}"
        )

# =========================
# SAVE
# =========================
modes = pd.DataFrame(rows)

if modes.empty:
    raise RuntimeError("No KDE modes were computed.")

missing = [g for g in GROUPS if g not in set(modes["group_label"])]
if missing:
    print(f"[warn] Missing expected groups in output: {missing}")

modes = modes.sort_values("group_label").reset_index(drop=True)
modes.to_csv(OUTCSV, index=False)

print(f"[done] Saved KDE modes table to: {OUTCSV}")
print(modes)
