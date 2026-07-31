# SAMPLE-LEVEL PCA BIPLOT + 3D PCA
# Uses scaled features, averages tiles by sample, colors by phenotype/sample_group.
# Marker loading arrows are hardcoded as real marker names and auto-scaled to fit.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# CONFIG
# -----------------------------

calculation = "density_pixels_above_1"
threshold = "0.4_threshold"

CSV_IN = "rabbanilab/wsi_analysis/input_csv_files/n_of_3.csv"

output_path_16 = Path(
    f"rabbanilab/wsi_analysis/data/clustering/{threshold}/pcas/{calculation}/full_plots"
)
output_path_16.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(CSV_IN, low_memory=False)

sample_group_color_map = {
    "Wild-type POD 10": "#707572",
    "Wild-type POD 5": "#B4BDB7",
    "Diabetic POD 10": "#e24a4a",
    "Diabetic POD 5": "#FEA0A0",
    "Diabetic + Exo POD 10": "#3989D0",
    "Diabetic + Exo POD 5": "#99CFFF",
}

channel_names = [
    "DAPI",
    "SMA",
    "Vimentin",
    "F4/80",
    "CD31",
    "Ki67",
]

# -----------------------------
# HELPER: 2D CONFIDENCE ELLIPSE
# -----------------------------

def draw_confidence_ellipse(
    x, y, ax, n_std=2.0,
    edgecolor="black", facecolor="none",
    alpha=0.2, linewidth=1.5
):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size != y.size or x.size < 3:
        return

    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(
        (np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        alpha=alpha,
    )

    ax.add_patch(ellipse)

# -----------------------------
# HELPER: 3D CONFIDENCE ELLIPSOID
# -----------------------------

def draw_confidence_ellipsoid_3d(
    x, y, z, ax, n_std=2.0,
    color="blue", alpha=0.08,
    wireframe_alpha=0.3, n_points=30
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if x.size < 4:
        return

    data = np.vstack([x, y, z])
    mean = data.mean(axis=1)
    cov = np.cov(data)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    radii = n_std * np.sqrt(eigenvalues)

    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)

    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(u)):
        for j in range(len(v)):
            point = np.array([
                sphere_x[i, j] * radii[0],
                sphere_y[i, j] * radii[1],
                sphere_z[i, j] * radii[2],
            ])
            rotated = eigenvectors @ point + mean
            sphere_x[i, j] = rotated[0]
            sphere_y[i, j] = rotated[1]
            sphere_z[i, j] = rotated[2]

    ax.plot_wireframe(
        sphere_x, sphere_y, sphere_z,
        color=color,
        alpha=wireframe_alpha,
        linewidth=0.5,
        rstride=3,
        cstride=3,
    )

    ax.plot_surface(
        sphere_x, sphere_y, sphere_z,
        color=color,
        alpha=alpha,
        linewidth=0,
    )

# -----------------------------
# 1. BUILD SAMPLE-LEVEL FEATURE MATRIX
# -----------------------------

scaled_feature_cols = [
    c for c in df.columns if c.startswith("scaled_feature_")
]

if not scaled_feature_cols:
    raise ValueError("No scaled_feature_* columns found in df.")

if len(channel_names) != len(scaled_feature_cols):
    raise ValueError(
        f"Number of marker names ({len(channel_names)}) does not match "
        f"number of scaled features ({len(scaled_feature_cols)})."
    )

required_cols = ["sample_name", "sample_group"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"df must contain a '{col}' column.")

df["sample_name"] = df["sample_name"].astype(str).str.strip()
df["sample_group"] = df["sample_group"].astype(str).str.strip()

# Average tile-level scaled features within each sample/mouse
sample_feature_means = df.groupby("sample_name")[scaled_feature_cols].mean()

# Get one sample_group label per sample
sample_groups = df.groupby("sample_name")["sample_group"].first()
sample_groups = sample_groups.loc[sample_feature_means.index]

X_scaled = sample_feature_means.to_numpy()

label_col = "sample_group"

# -----------------------------
# 2. PCA COMPUTATION
# -----------------------------

pca = PCA(n_components=3, random_state=42)
pca_scores = pca.fit_transform(X_scaled)

loadings = pca.components_.T

pca_df = pd.DataFrame({
    "sample_name": sample_feature_means.index,
    "PCA1": pca_scores[:, 0],
    "PCA2": pca_scores[:, 1],
    "PCA3": pca_scores[:, 2],
    label_col: sample_groups.values,
})

var_exp = pca.explained_variance_ratio_ * 100
print("Explained variance ratio (%):", var_exp)

loading_df = pd.DataFrame(
    loadings,
    index=channel_names,
    columns=["PC1", "PC2", "PC3"]
)

print("\nPCA loadings:")
print(loading_df)

# -----------------------------
# 3. 2D PCA BIPLOT
# -----------------------------

def pca_biplot(pca_df_in, label_col, title, save_prefix=None):
    fig, ax = plt.subplots(figsize=(9, 9))

    unique_labels = sorted(pca_df_in[label_col].astype(str).unique())

    for lab in unique_labels:
        sub = pca_df_in[pca_df_in[label_col].astype(str) == lab]
        color = sample_group_color_map.get(lab, "#999999")

        ax.scatter(
            sub["PCA1"],
            sub["PCA2"],
            s=80,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            color=color,
            label=str(lab),
        )

        if len(sub) >= 3:
            draw_confidence_ellipse(
                sub["PCA1"],
                sub["PCA2"],
                ax=ax,
                n_std=1.5,
                edgecolor=color,
                facecolor=color,
                alpha=0.18,
                linewidth=1.5,
            )

    # Auto-scale feature loading arrows
    score_x_range = pca_df_in["PCA1"].max() - pca_df_in["PCA1"].min()
    score_y_range = pca_df_in["PCA2"].max() - pca_df_in["PCA2"].min()

    loading_x_max = np.max(np.abs(loadings[:, 0]))
    loading_y_max = np.max(np.abs(loadings[:, 1]))

    arrow_scale_x = 0.35 * score_x_range / loading_x_max if loading_x_max != 0 else 1
    arrow_scale_y = 0.35 * score_y_range / loading_y_max if loading_y_max != 0 else 1

    arrow_scale = 1.6 * min(arrow_scale_x, arrow_scale_y)

    arrow_xs = []
    arrow_ys = []

    for i, marker_name in enumerate(channel_names):
        x_loading = loadings[i, 0] * arrow_scale
        y_loading = loadings[i, 1] * arrow_scale

        arrow_xs.append(x_loading)
        arrow_ys.append(y_loading)

        ax.arrow(
            0,
            0,
            x_loading,
            y_loading,
            color="firebrick",
            width=0.005,
            head_width=0.10,
            length_includes_head=True,
            alpha=0.9,
        )

        ax.text(
            x_loading * 1.12,
            y_loading * 1.12,
            marker_name,
            color="firebrick",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Expand axis limits so arrows and marker labels stay visible
    all_x = np.concatenate([
        pca_df_in["PCA1"].to_numpy(),
        np.array(arrow_xs) * 1.3,
    ])

    all_y = np.concatenate([
        pca_df_in["PCA2"].to_numpy(),
        np.array(arrow_ys) * 1.3,
    ])

    x_margin = 0.15 * (all_x.max() - all_x.min())
    y_margin = 0.15 * (all_y.max() - all_y.min())

    ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    ax.axhline(0, color="gray", linewidth=1, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=1, alpha=0.4)

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var.)")
    ax.set_title(title, fontsize=15)

    ax.legend(
        title=label_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    fig.tight_layout()

    if save_prefix:
        pdf_path = output_path_16 / f"{calculation}_{save_prefix}_biplot.pdf"
        png_path = output_path_16 / f"{calculation}_{save_prefix}_biplot.png"

        fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
        fig.savefig(png_path, bbox_inches="tight", dpi=300)

        print("[✓] Saved:", pdf_path)
        print("[✓] Saved:", png_path)

    plt.show()
    plt.close(fig)

pca_biplot(
    pca_df,
    label_col=label_col,
    title=f"Sample-level PCA biplot colored by {label_col}",
    save_prefix=f"sample_PCA_{label_col}"
)

# -----------------------------
# 4. STATIC 3D PCA SCATTER
# -----------------------------

def pca_3d_scatter(pca_df_in, label_col, title, save_prefix=None):
    labels_in = pca_df_in[label_col].astype(str).values
    unique_labels = sorted(pd.Series(labels_in).unique())

    colors = [
        sample_group_color_map.get(lab, "#999999")
        for lab in labels_in
    ]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pca_df_in["PCA1"],
        pca_df_in["PCA2"],
        pca_df_in["PCA3"],
        c=colors,
        s=80,
        alpha=0.9,
        linewidths=0.5,
        edgecolor="black",
    )

    for lab in unique_labels:
        sub = pca_df_in[pca_df_in[label_col].astype(str) == lab]
        color = sample_group_color_map.get(lab, "#999999")

        draw_confidence_ellipsoid_3d(
            sub["PCA1"].values,
            sub["PCA2"].values,
            sub["PCA3"].values,
            ax=ax,
            n_std=1.5,
            color=color,
            alpha=0.2,
            wireframe_alpha=0.25,
        )

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({var_exp[2]:.1f}%)")
    ax.set_title(title)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=sample_group_color_map.get(lab, "#999999"),
            label=str(lab),
            markersize=6,
        )
        for lab in unique_labels
    ]

    ax.legend(
        handles=handles,
        title=label_col,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
    )

    ax.view_init(elev=20, azim=35)
    plt.tight_layout()

    if save_prefix:
        pdf_path = output_path_16 / f"{calculation}_{save_prefix}_3D.pdf"
        png_path = output_path_16 / f"{calculation}_{save_prefix}_3D.png"

        plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
        plt.savefig(png_path, bbox_inches="tight", dpi=300)

        print("[✓] Saved:", pdf_path)
        print("[✓] Saved:", png_path)

    plt.show()
    plt.close(fig)

pca_3d_scatter(
    pca_df,
    label_col=label_col,
    title=f"Sample-level 3D PCA colored by {label_col}",
    save_prefix=f"sample_PCA_{label_col}"
)

# -----------------------------
# 5. SAVE PCA SCORES AND LOADINGS
# -----------------------------

pca_scores_path = output_path_16 / f"{calculation}_sample_level_PCA_scores.csv"
pca_loadings_path = output_path_16 / f"{calculation}_sample_level_PCA_loadings.csv"

pca_df.to_csv(pca_scores_path, index=False)
loading_df.to_csv(pca_loadings_path)

print("[✓] Saved PCA scores:", pca_scores_path)
print("[✓] Saved PCA loadings:", pca_loadings_path)
