from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# =========================
# FILES
# =========================
CSV_IN = "rabbanilab/wsi_analysis/input_csv_files/n_of_3_with_rotated_coords_and_wound_info.csv"
MODES_CSV = "kde_mode_output_granulation_only/kde_modes_table.csv"

OUTDIR = Path("kde_mode_output_granulation_only")
OUTDIR.mkdir(exist_ok=True, parents=True)

OUTFIG = OUTDIR / "granulation_umap_with_kde_modes_clean.png"
OUTFIG_KDE = OUTDIR / "granulation_umap_kde_contours_downsampled_shared_levels.png"
OUTFIG_KDE_PDF = OUTDIR / "granulation_umap_kde_contours_downsampled_shared_levels.pdf"

# =========================
# COLUMNS
# =========================
xcol, ycol = "UMAP1", "UMAP2"
SUPERCLUSTER_COL = "superclusters"
GROUP_COL = "sample_group"
GRAN_MATCH = r"granulation\s*tissue"

# =========================
# GROUP MAPPING
# =========================
GROUP_MAP = {
    "Wild-type POD 5": "WT_D5",
    "Wild-type POD 10": "WT_D10",
    "Diabetic POD 5": "dbdb_D5",
    "Diabetic POD 10": "dbdb_D10",
    "Diabetic + Exo POD 5": "dbdb_Exo_D5",
    "Diabetic + Exo POD 10": "dbdb_Exo_D10",
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
# KDE CONTOUR SETTINGS
# =========================
RANDOM_SEED = 0
DOWNSAMPLE_N = None
GRID_SIZE = 200
KDE_BW_METHOD = None
N_CONTOUR_LEVELS = 8
PAD = 0.5

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_IN, low_memory=False)
modes = pd.read_csv(MODES_CSV)

df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
df = df.dropna(subset=[xcol, ycol, SUPERCLUSTER_COL, GROUP_COL]).copy()

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

# =========================
# COLORS
# =========================
colors = {
    "WT_D5": "tab:blue",
    "WT_D10": "tab:red",
    "dbdb_D5": "tab:orange",
    "dbdb_D10": "tab:green",
    "dbdb_Exo_D5": "saddlebrown",
    "dbdb_Exo_D10": "tab:purple",
}

# ==================================================
# FIGURE 1: UMAP WITH FULL-DATA KDE MODE MARKERS
# ==================================================
fig, ax = plt.subplots(figsize=(13, 8))

ax.scatter(
    df_gran[xcol],
    df_gran[ycol],
    s=3,
    c="lightgray",
    alpha=0.35,
    linewidths=0,
)

for _, row in modes.iterrows():
    g = row["group_label"]
    x = row["kde_x"]
    y = row["kde_y"]

    ax.scatter(
        x,
        y,
        s=220,
        marker="X",
        c=colors.get(g, "black"),
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )

tile_handle = Line2D(
    [0], [0],
    marker="o",
    color="none",
    markerfacecolor="lightgray",
    markeredgecolor="lightgray",
    markersize=8,
    label="Tiles (Granulation Tissue)",
)

mode_handles = []

for g in colors:
    if g in set(modes["group_label"]):
        row = modes.loc[modes["group_label"] == g].iloc[0]
        mode_handles.append(
            Line2D(
                [0], [0],
                marker="X",
                color="none",
                markerfacecolor=colors[g],
                markeredgecolor=colors[g],
                markersize=10,
                label=f"{g}  ({row['kde_x']:.2f}, {row['kde_y']:.2f})",
            )
        )

legend1 = ax.legend(
    handles=[tile_handle],
    loc="upper right",
    frameon=True,
)
ax.add_artist(legend1)

ax.legend(
    handles=mode_handles,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    title="KDE Mode Coordinates\nFull data, continuous",
    frameon=True,
)

ax.text(
    0.02,
    0.03,
    "Granulation tissue only\n"
    "Mode coordinates from full-data KDE\n"
    "Used for D5→D10 trajectory",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="gray",
        alpha=0.85,
    ),
)

ax.set_title(
    "UMAP of Granulation Tissue\nwith Full-Data KDE Mode Locations",
    fontsize=18,
    weight="bold",
)

ax.set_xlabel("UMAP1", fontsize=14)
ax.set_ylabel("UMAP2", fontsize=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUTFIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"[done] Saved figure to: {OUTFIG}")


# ==================================================
# FIGURE 2: PAPER-STYLE KDE CONTOURS
# gaussian_kde + downsampling + shared contour levels
# ==================================================

group_counts = df_gran["group_label"].value_counts()

print("\n[info] granulation counts per group:")
print(group_counts.reindex(GROUPS))

if DOWNSAMPLE_N is None:
    downsample_n = int(group_counts.reindex(GROUPS).min())
else:
    downsample_n = int(DOWNSAMPLE_N)

print(f"\n[info] downsampling each group to n={downsample_n}")

xmin, xmax = df_gran[xcol].min(), df_gran[xcol].max()
ymin, ymax = df_gran[ycol].min(), df_gran[ycol].max()

xmin -= PAD
xmax += PAD
ymin -= PAD
ymax += PAD

xg = np.linspace(xmin, xmax, GRID_SIZE)
yg = np.linspace(ymin, ymax, GRID_SIZE)
Xg, Yg = np.meshgrid(xg, yg)
grid_coords = np.vstack([Xg.ravel(), Yg.ravel()])

density_maps = {}
downsampled_points = {}
gaussian_kde_modes = {}

for g in GROUPS:
    sub = df_gran[df_gran["group_label"] == g].copy()

    if len(sub) < downsample_n:
        raise ValueError(f"{g} has fewer than downsample_n points.")

    sub_ds = sub.sample(
        n=downsample_n,
        random_state=RANDOM_SEED,
        replace=False,
    ).copy()

    coords = sub_ds[[xcol, ycol]].to_numpy().T
    kde = gaussian_kde(coords, bw_method=KDE_BW_METHOD)

    Z = kde(grid_coords).reshape(Xg.shape)

    density_maps[g] = Z
    downsampled_points[g] = sub_ds

    max_idx = np.argmax(Z)
    gaussian_kde_modes[g] = (
        float(Xg.ravel()[max_idx]),
        float(Yg.ravel()[max_idx]),
    )

all_density_values = np.concatenate([Z.ravel() for Z in density_maps.values()])
positive_density = all_density_values[all_density_values > 0]

level_min = np.percentile(positive_density, 75)
level_max = positive_density.max()
levels = np.linspace(level_min, level_max, N_CONTOUR_LEVELS)

print("\n[info] shared contour levels:")
print(levels)

print("\n[info] downsampled gaussian_kde modes, used only for contour figure:")
for g in GROUPS:
    print(f"{g}: ({gaussian_kde_modes[g][0]:.4f}, {gaussian_kde_modes[g][1]:.4f})")

# =========================
# PLOT KDE CONTOURS
# =========================
fig2, axes = plt.subplots(
    2,
    3,
    figsize=(15, 8.5),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

axes = axes.ravel()
cf = None

for ax, g in zip(axes, GROUPS):
    sub_ds = downsampled_points[g]
    Z = density_maps[g]

    ax.scatter(
        sub_ds[xcol],
        sub_ds[ycol],
        s=3,
        c="lightgray",
        alpha=0.28,
        linewidths=0,
        zorder=1,
    )

    cf = ax.contourf(
        Xg,
        Yg,
        Z,
        levels=levels,
        cmap="Blues",
        alpha=0.80,
        zorder=2,
    )

    ax.contour(
        Xg,
        Yg,
        Z,
        levels=levels,
        colors="black",
        linewidths=0.75,
        zorder=3,
    )

    # X marker here is the mode of the downsampled gaussian_kde surface.
    #gx, gy = gaussian_kde_modes[g]
    #ax.scatter(
        #gx,
        #gy,
        #s=95,
        #marker="X",
        #c=colors.get(g, "black"),
        #edgecolor="white",
        #linewidth=1.0,
        #zorder=4,
    #)

    ax.set_title(
        f"{g}\nDownsampled n={downsample_n}",
        fontsize=11,
        pad=8,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax in axes[3:]:
    ax.set_xlabel("UMAP1", fontsize=11)

for ax in axes[::3]:
    ax.set_ylabel("UMAP2", fontsize=11)

cbar = fig2.colorbar(
    cf,
    ax=axes,
    location="right",
    shrink=0.78,
    pad=0.025,
)

cbar.set_label("KDE density", fontsize=11)
cbar.ax.tick_params(labelsize=9)

fig2.suptitle(
    "Granulation Tissue Density by Group\n"
    "gaussian_kde, equal downsampling, shared contour levels",
    fontsize=16,
    weight="bold",
)

fig2.savefig(OUTFIG_KDE, dpi=300, bbox_inches="tight")
fig2.savefig(OUTFIG_KDE_PDF, dpi=300, bbox_inches="tight")
plt.show()

print(f"[done] Saved KDE contour figure to: {OUTFIG_KDE}")
print(f"[done] Saved KDE contour PDF to: {OUTFIG_KDE_PDF}")
