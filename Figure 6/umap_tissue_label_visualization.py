"""
UMAP visualization of tile-level features with inferred tissue labels.

This script computes a UMAP embedding from scaled tile-level density features
and generates plots colored by clustering labels and inferred dominant tissue
annotations.
"""


import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines

# 1. Load the CSV
df = pd.read_csv("filtered_density31_with_inferred_labels_full_and_intensity41.csv")

# 2. Define scaled density feature columns
feature_cols = [
    'scaled_density_feature_0', 'scaled_density_feature_1', 'scaled_density_feature_2',
    'scaled_density_feature_3', 'scaled_density_feature_4', 'scaled_density_feature_5'
]

# 3. Drop rows with NaNs
features = df[feature_cols].values
valid_mask = ~np.isnan(features).any(axis=1)
features_clean = features[valid_mask]
df_clean = df.loc[valid_mask].copy()

# 4. UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embeddings = reducer.fit_transform(features_clean)

# 5. Plot: density_cluster_31
unique_clusters = sorted(df_clean['density_cluster_31'].dropna().unique())
cmap = plt.cm.get_cmap('tab20', len(unique_clusters))
color_dict = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
cluster_colors = df_clean['density_cluster_31'].map(color_dict)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_colors, s=10, alpha=0.8, edgecolors='none')
handles = [mlines.Line2D([], [], marker='o', linestyle='None', markersize=8,
                         markerfacecolor=color_dict[cluster], label=str(cluster))
           for cluster in unique_clusters]
plt.legend(handles=handles, title='Density Cluster 31', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.title('UMAP: density_cluster_31')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.tight_layout()
plt.savefig("umap_density_cluster_31.pdf", bbox_inches='tight')
plt.close()

# 6. Plot: density_cluster_31_supercluster
unique_superclusters = sorted(df_clean['inferred_dominant_tissue_from_cluster'].dropna().unique())
palette = sns.color_palette("Set2", len(unique_superclusters))
supercluster_color_dict = {name: palette[i] for i, name in enumerate(unique_superclusters)}
supercluster_colors = df_clean['inferred_dominant_tissue_from_cluster'].map(supercluster_color_dict)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=supercluster_colors, s=10, alpha=0.8, edgecolors='none')
handles = [mlines.Line2D([], [], marker='o', linestyle='None', markersize=8,
                         markerfacecolor=supercluster_color_dict[name], label=name)
           for name in unique_superclusters]
plt.legend(handles=handles, title='Density Supercluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.title('UMAP: density_cluster_31_supercluster')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.tight_layout()
plt.savefig("umap_density_supercluster_31.pdf", bbox_inches='tight')
plt.close()
