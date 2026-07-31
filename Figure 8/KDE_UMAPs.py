from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# FILES
# Be mindfull of the csv which one you pikc either the consensus one or the individual one - this code basically plots the KDE points on the UMAP 
# =========================
CSV_IN = "rabbanilab/wsi_analysis/input_csv_files/n_of_3_with_rotated_coords_and_wound_info.csv"
MODES_CSV = "kde_mode_output_granulation_only_POD/kde_modes_table_POD5_POD10.csv"
#kde_modes_table_POD5_POD10.csv
#kde_modes_table.csv

OUTDIR = Path("kde_mode_output_granulation_only")
OUTDIR.mkdir(exist_ok=True, parents=True)
OUTFIG = OUTDIR / "granulation_umap_with_kde_modes_clean.png"

# =========================
# COLUMNS
# =========================
xcol, ycol = "UMAP1", "UMAP2"
SUPERCLUSTER_COL = "superclusters"
GRAN_MATCH = r"granulation\s*tissue"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_IN, low_memory=False)
modes = pd.read_csv(MODES_CSV)

df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
df[ycol] = pd.to_numeric(df[ycol], errors="coerce")

df = df.dropna(subset=[xcol, ycol, SUPERCLUSTER_COL]).copy()

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
#colors = {
    #"WT_D5": "tab:blue",
    #"WT_D10": "tab:red",
    #"dbdb_D5": "tab:orange",
    #"dbdb_D10": "tab:green",
    #"dbdb_Exo_D5": "saddlebrown",
    #"dbdb_Exo_D10": "tab:purple",
#}

colors = {
    "POD5": "tab:blue",
    "POD10": "tab:red",
}


# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(13, 8))

# Granulation tissue UMAP points
ax.scatter(
    df_gran[xcol],
    df_gran[ycol],
    s=3,
    c="lightgray",
    alpha=0.35,
    linewidths=0,
)

# KDE mode markers only
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

# =========================
# LEGENDS
# =========================
tile_handle = Line2D(
    [0], [0],
    marker="o",
    color="none",
    markerfacecolor="lightgray",
    markeredgecolor="lightgray",
    markersize=8,
    label="Tiles (Granulation Tissue)",
)

#mode_handles = []

#for g in colors:
    #if g in set(modes["group_label"]):
        #row = modes.loc[modes["group_label"] == g].iloc[0]
        #mode_handles.append(
            #Line2D(
                #[0], [0],
                #marker="X",
                #color="none",
                #markerfacecolor=colors[g],
                #markeredgecolor=colors[g],
                #markersize=10,
                #label=f"{g}  ({row['kde_x']:.2f}, {row['kde_y']:.2f})",
            #)
        #)


mode_handles = []

label_map = {
    "POD5": "POD 5",
    "POD10": "POD 10",
}

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
                label=f"{label_map[g]} ({row['kde_x']:.2f}, {row['kde_y']:.2f})",
            )
        )

legend1 = ax.legend(
    handles=[tile_handle],
    loc="upper right",
    frameon=True,
)

ax.add_artist(legend1)

legend2 = ax.legend(
    handles=mode_handles,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    title="KDE Mode Coordinates\n(Continuous)",
    frameon=True,
)

# =========================
# ANNOTATION BOX
# =========================
ax.text(
    0.02,
    0.03,
    "Granulation tissue only\n"
    "KDE mode estimated with bandwidth = 0.5\n"
    "Continuous optimization",
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

# =========================
# STYLE
# =========================
ax.set_title(
    "UMAP of Granulation Tissue\nwith KDE Mode Locations",
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
