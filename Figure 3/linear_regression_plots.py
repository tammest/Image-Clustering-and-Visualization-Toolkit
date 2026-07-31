# FACET-STYLE LINEAR REGRESSION PLOTS
# Each panel: one marker vs one PCA score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------
# Choose which PC to plot
# -----------------------------

pc_to_plot = "PCA3"   # change to "PCA1", "PCA2", or "PCA3"

pc_index_map = {
    "PCA1": 0,
    "PCA2": 1,
    "PCA3": 2,
}

pc_idx = pc_index_map[pc_to_plot]

# -----------------------------
# Marker names
# -----------------------------

channel_name_map = {
    "scaled_feature_0": "DAPI",
    "scaled_feature_1": "SMA",
    "scaled_feature_2": "Vimentin",
    "scaled_feature_3": "F4/80",
    "scaled_feature_4": "CD31",
    "scaled_feature_5": "Ki67",
}

# -----------------------------
# Build feature dataframe
# -----------------------------

feature_df = pd.DataFrame(
    X_scaled,
    columns=scaled_feature_cols
)

y_pc = pca_scores[:, pc_idx]

# -----------------------------
# Facet-style plot
# -----------------------------

n_features = len(scaled_feature_cols)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(15, 4 * n_rows),
    sharey=True
)

axes = axes.flatten()

regression_results = []

for i, feature_col in enumerate(scaled_feature_cols):
    ax = axes[i]

    feature_label = channel_name_map.get(feature_col, feature_col)

    x = feature_df[[feature_col]].values
    y = y_pc

    lr = LinearRegression()
    lr.fit(x, y)

    y_pred = lr.predict(x)
    r2 = r2_score(y, y_pred)
    slope = lr.coef_[0]
    intercept = lr.intercept_

    regression_results.append({
        "PC": pc_to_plot,
        "feature": feature_col,
        "marker": feature_label,
        "slope": slope,
        "intercept": intercept,
        "R2": r2,
    })

    sort_idx = np.argsort(x[:, 0])
    x_sorted = x[:, 0][sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    for group in pca_df[label_col].astype(str).unique():
        mask = pca_df[label_col].astype(str).values == group

        ax.scatter(
            x[mask, 0],
            y[mask],
            color=sample_group_color_map.get(group, "#999999"),
            edgecolor="black",
            s=60,
            alpha=0.9,
            label=group if i == 0 else None
        )

    ax.plot(
        x_sorted,
        y_pred_sorted,
        color="black",
        linewidth=2
    )

    ax.set_title(
        f"{feature_label} vs {pc_to_plot}\n"
        f"R² = {r2:.3f}, slope = {slope:.3f}"
    )

    ax.set_xlabel(feature_label)

    if i % n_cols == 0:
        ax.set_ylabel(f"{pc_to_plot} score")

# Remove empty subplot panels if any
for j in range(n_features, len(axes)):
    fig.delaxes(axes[j])

# Single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title=label_col,
    bbox_to_anchor=(1.02, 0.5),
    loc="center left"
)

fig.suptitle(
    f"Linear regressions: markers vs {pc_to_plot}",
    fontsize=16,
    y=1.02
)

plt.tight_layout()

# Optional save
out_path = output_path_16 / f"{calculation}_marker_regressions_vs_{pc_to_plot}.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=300)

print("[✓] Saved:", out_path)

plt.show()

regression_results_df = pd.DataFrame(regression_results)
print(regression_results_df.sort_values("R2", ascending=False))


# -----------------------------
# This is hard coded right now so comment out the one you are not running above. Then make the heatmap. 
# -----------------------------

#regression_results_pc1 = regression_results_df.copy()
#regression_results_pc2 = regression_results_df.copy()
regression_results_pc3 = regression_results_df.copy()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Combine all regression result tables
# ----------------------------------------------------

all_results = pd.concat(
    [
        regression_results_pc1,
        regression_results_pc2,
        regression_results_pc3,
    ],
    ignore_index=True
)

# ----------------------------------------------------
# Heatmap table
# ----------------------------------------------------

heatmap_df = (
    all_results
    .pivot(
        index="marker",
        columns="PC",
        values="R2"
    )
)

# Order markers nicely
marker_order = [
    "DAPI",
    "SMA",
    "Vimentin",
    "F4/80",
    "CD31",
    "Ki67",
]

heatmap_df = heatmap_df.loc[marker_order]

# ----------------------------------------------------
# Plot
# ----------------------------------------------------

plt.figure(figsize=(5,6))

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    linewidths=1,
    square=True,
    vmin=0,
    vmax=1,
    cbar_kws={"label":"$R^2$"}
)

plt.title("Marker association with principal components")

plt.xlabel("Principal Component")
plt.ylabel("Marker")

plt.tight_layout()
heatmap_path = output_path_16 / f"{calculation}_PCA_marker_R2_heatmap.pdf"

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"[✓] Saved: {heatmap_path}")
plt.show()
