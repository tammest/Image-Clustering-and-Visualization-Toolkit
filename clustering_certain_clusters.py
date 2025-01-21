#try here
import os
import zarr 
import tifffile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from scipy.stats import pearsonr  


RANDOM_SEED = 42

# Function to load Zarr data
def load_zarr_data(sample_name, root_path):
    sample_zarr_path = os.path.join(root_path, sample_name, 'data.zarr')
    data = zarr.open(sample_zarr_path, mode='r')  # Read-only access to Zarr data
    return data

# Function to load Zarr data and channel metadata
def load_zarr_w_channel(root_path, sample_name):
    sample_zarr_path = os.path.join(root_path, sample_name, 'data.zarr')
    channel_path = os.path.join(root_path, sample_name, 'channels.csv')
    data = zarr.open(sample_zarr_path, mode='r')
    channels = pd.read_csv(channel_path)  # Read the channel information
    return data, channels

# Function to load tile positions info
def load_tile_info(root_path, sample_name, tile_size):
    tile_df = pd.read_csv(os.path.join(root_path, sample_name, f'tiles/positions_{tile_size}.csv'), index_col=0)
    return tile_df

# Feature extraction: Extract features for each tile
def extract_features(data, channels, tile_df, tile_size):
    feature_list = []
    channel_indices = channels['channel'].values[channels['channel'] != 6]  # Exclude autofluorescence channel (channel 6)

    # Pre-load channel data
    channel_data = {channel_id: data[channel_id][:] for channel_id in channel_indices}

    for idx, tile in tile_df.iterrows():
        x, y = tile[['w', 'h']]  # Tile position
        features = []
        for channel_id in channel_indices:
            tile_region = channel_data[channel_id][y:y + tile_size, x:x + tile_size]  # Extract tile region
            density = np.sum(tile_region > 0)  # Count non-zero values (density)
            features.append(density)

        # Check if the tile is blank (all densities are zero)
        if np.sum(features) > 0:  # Only add non-blank tiles
            feature_list.append(features)

    features_array = np.array(feature_list)

    # Return raw density counts without normalization
    return features_array

# Compute WCSS for KMeans (elbow method)
def compute_wcss(features, max_clusters=30):
    wcss = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)
    return wcss

# Perform clustering and plot results (PCA plot)
def perform_clustering_and_plot(features, n_clusters=30):
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    #pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    explained_variance = pca.explained_variance_ratio_ * 100
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(features)
    colors = generate_colormap(n_clusters)

    # Plot the PCA result
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap=mcolors.ListedColormap(colors), s=20, alpha=0.7)
    plt.title(f'Clustering of Tile Features\nPCA1: {explained_variance[0]:.2f}% explained variance, PCA2: {explained_variance[1]:.2f}% explained variance')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'clustering_pca.pdf')  # Save the PCA plot as PDF
    plt.close()  # Close the figure to avoid display

    return clusters, pca_result, colors
    
def increase_brightness(image, brightness_factor=4.0):
    """
    Increase the brightness of the image by scaling the pixel values.
    
    :param image: The input image (2D numpy array).
    :param brightness_factor: Factor by which to increase brightness (>1 for brighter).
    :return: Brightened image.
    """
    # Normalize image to [0, 1] (if not already in this range)
    image_normalized = np.clip(image / np.max(image), 0, 1)
    
    # Increase brightness by scaling
    brightened_image = image_normalized * brightness_factor
    
    # Clip values to the valid range [0, 1] to avoid overflow
    brightened_image = np.clip(brightened_image, 0, 1)
    
    # Rescale back to the original image range if needed (e.g., [0, 255] for 8-bit images)
    brightened_image = (brightened_image * 255).astype(np.uint8)  # Assuming 8-bit image
    
    return brightened_image

def plot_cluster_tile_positions_with_autofluorescence(tile_df, clusters, colors, autofluorescence_channel, sample_name, brightness_factor=1.5):
    # Increase brightness of the autofluorescence channel
    bright_autofluorescence_channel = increase_brightness(autofluorescence_channel, brightness_factor)
    
    img_height, img_width = bright_autofluorescence_channel.shape
    print(f"Original Max Tile Positions (w, h): ({tile_df['w'].max()}, {tile_df['h'].max()})")
    
    # Check the tile grid and image dimensions
    print(f"Image Dimensions: {img_width}x{img_height}")
    
    # Calculate scaling factors
    scale_x = img_width / (tile_df['w'].max() + 1)
    scale_y = img_height / (tile_df['h'].max() + 1)
    print(f"Scaling factors - scale_x: {scale_x}, scale_y: {scale_y}")
    
    # Apply scaling to the tile positions
    w_scaled = tile_df['w'].values * scale_x
    h_scaled = tile_df['h'].values * scale_y

    # Ensure the scaled positions are within the bounds of the image
    max_width = img_width
    max_height = img_height
    w_scaled = np.clip(w_scaled, 0, max_width)
    h_scaled = np.clip(h_scaled, 0, max_height)

    # Check the scaled positions
    print(f"Scaled tile positions (first 5):")
    for w, h in zip(w_scaled[:5], h_scaled[:5]):
        print(f"w_scaled: {w}, h_scaled: {h}")
    
    num_clusters = len(colors)
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # Plot the brightened Autofluorescence Channel
    ax[0].imshow(bright_autofluorescence_channel, cmap='gray', extent=[0, img_width, img_height, 0])
    ax[0].set_title('Brightened Autofluorescence Channel')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    
    # Plot the Tile Positions for Each Cluster
    for cluster_number in range(num_clusters):
        cluster_indices = np.where(clusters == cluster_number)[0]
        h_cluster = h_scaled[cluster_indices]
        w_cluster = w_scaled[cluster_indices]
        ax[1].scatter(w_cluster, h_cluster, color=colors[cluster_number], label=f'Cluster {cluster_number}', alpha=0.5)

    ax[1].set_title('Tile Positions for Each Cluster')
    ax[1].set_xlabel('w (width)')
    ax[1].set_ylabel('h (height)')
    ax[1].invert_yaxis()
    ax[1].set_aspect('equal')
    
    # Adjust the legend position to be outside the plot
    ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Clusters", fontsize=10)

    # Adjust the layout to make room for the legend outside the plot
    plt.tight_layout()
    
    # Save the figure with the legend outside the plot area
    plt.savefig(f'cluster_tile_positions_with_autofluorescence_{sample_name}.pdf', bbox_inches='tight')
    plt.close()

def plot_channel_correlation(tile_df, data, channels, tile_size, sample_name, output_dir='plots'):
    """
    Generate a combined plot with a correlation matrix, scatter plots for channel density correlations,
    Pearson's r, and histograms for each sample in one figure.
    
    :param tile_df: DataFrame with tile positions.
    :param data: Loaded Zarr data.
    :param channels: DataFrame with channel metadata.
    :param tile_size: The size of each tile.
    :param sample_name: Name of the sample being processed.
    :param output_dir: Directory to save the plot.
    """
    # Create a directory for the plots if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract features for each channel (density values)
    channel_indices = channels['channel'].values[channels['channel'] != 6]  # Exclude autofluorescence channel (channel 6)
    feature_list = []

    # Pre-load channel data
    channel_data = {channel_id: data[channel_id][:] for channel_id in channel_indices}

    for idx, tile in tile_df.iterrows():
        x, y = tile[['w', 'h']]  # Tile position
        features = []
        for channel_id in channel_indices:
            tile_region = channel_data[channel_id][y:y + tile_size, x:x + tile_size]  # Extract tile region
            density = np.sum(tile_region > 0)  # Count non-zero values (density)
            features.append(density)

        # Only add non-blank tiles
        if np.sum(features) > 0:
            feature_list.append(features)

    features_array = np.array(feature_list)

    # Create a correlation matrix for the densities of all channels
    corr_matrix = np.corrcoef(features_array, rowvar=False)

    # Set up a figure to hold all the subplots
    num_channels = len(channel_indices)
    fig, axs = plt.subplots(3, num_channels, figsize=(16, 3 * num_channels))  # 3 rows: correlation matrix, scatter plots, histograms

    # Plot the correlation matrix (heatmap)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=channels['marker'].values[channel_indices], 
                yticklabels=channels['marker'].values[channel_indices], ax=axs[0, 0], vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
    axs[0, 0].set_title('Correlation Matrix')

    # Plot scatter plots and Pearson's r for each channel pair
    for i in range(num_channels):
        for j in range(i + 1, num_channels):  # Only plot the upper triangle to avoid duplication
            ax = axs[1, i]  # Adjust to place scatter plot in the second row
            ax.scatter(features_array[:, i], features_array[:, j], alpha=0.5, color='b')

            # Calculate Pearson's r and plot on the title
            r, _ = pearsonr(features_array[:, i], features_array[:, j])
            ax.set_title(f'Pearson r: {r:.2f}')
            
            # Set axis labels to reflect the channel names
            ax.set_xlabel(f'Channel {i + 1} - {channels["marker"].values[channel_indices][i]}')
            ax.set_ylabel(f'Channel {j + 1} - {channels["marker"].values[channel_indices][j]}')

    # Plot histograms of the density values for each channel
    for i in range(num_channels):
        ax_hist = axs[2, i]  # Third row for histograms
        ax_hist.hist(features_array[:, i], bins=20, color='gray', alpha=0.7)
        # Use 'marker' instead of 'name'
        ax_hist.set_title(f'Histogram of {channels["marker"].values[channel_indices][i]} Density')
        ax_hist.set_xlabel('Density')
        ax_hist.set_ylabel('Frequency')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the combined figure as a PDF
    combined_pdf = os.path.join(output_dir, f'{sample_name}_correlation_plots_combined.pdf')
    fig.savefig(combined_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f"Combined correlation plot for {sample_name} saved.")


def generate_colormap(num_clusters):
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_clusters, 10)))  
    if num_clusters > 10:
        additional_colors = plt.cm.viridis(np.linspace(0, 1, num_clusters - 10))
        colors = np.vstack((colors, additional_colors))
    return colors

# Function to load Zarr data and channels
def load_data_from_zarr(root_path, sample_name):
    data, channels = load_zarr_w_channel(root_path, sample_name)
    autofluorescence_channel = data[-1]  # Assuming the last channel is autofluorescence
    print(f"Autofluorescence channel loaded with shape: {autofluorescence_channel.shape}")
    return data, channels, autofluorescence_channel


def plot_tile_count_per_cluster(clusters, n_clusters, sample_name):
    # Count the number of tiles in each cluster
    cluster_counts = np.bincount(clusters, minlength=n_clusters)
    
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_clusters), cluster_counts, color='skyblue')
    plt.xlabel('Cluster Number')
    plt.ylabel('Tile Count')
    plt.title(f'Tile Count per Cluster for {sample_name}')
    plt.xticks(range(n_clusters))
    plt.tight_layout()
    
    # Save the plot as a PDF
    plt.savefig(f'tile_count_per_cluster_{sample_name}.pdf')
    plt.close()
    
    # Save the data as a CSV file
    cluster_data = {'Cluster Number': range(n_clusters), 'Tile Count': cluster_counts}
    df = pd.DataFrame(cluster_data)
    df.to_csv(f'tile_count_per_cluster_{sample_name}.csv', index=False)

    print(f'CSV file saved as tile_count_per_cluster_{sample_name}.csv')
    
def plot_cluster_tile_positions(tile_df, clusters, colors, sample_name):
    h = tile_df['h'].values
    w = tile_df['w'].values

    num_clusters = len(colors)
    plt.figure(figsize=(12, 10))  # Set the figure size
    
    for cluster_number in range(num_clusters):
        cluster_indices = np.where(clusters == cluster_number)[0]
        h_cluster = h[cluster_indices]
        w_cluster = w[cluster_indices]
        plt.scatter(w_cluster, h_cluster, color=colors[cluster_number], label=f'Cluster {cluster_number}', alpha=0.5)

    plt.title(f'Tile Positions for Each Cluster - {sample_name}')
    plt.xlabel('w (width)')
    plt.ylabel('h (height)')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    # Save the plot with the sample name
    plt.savefig(f'cluster_tile_position_{sample_name}.pdf')  # Save as PDF with the sample name
    plt.close()  # Close the plot to free up memory
    
def save_tiles_from_clusters(tile_df, clusters, selected_clusters, output_filename):
    """
    Extract tiles from the specified clusters and save their coordinates to a CSV file.
    
    :param tile_df: DataFrame containing tile positions (w, h).
    :param clusters: Array of cluster assignments for each tile.
    :param selected_clusters: List of cluster IDs to extract.
    :param output_filename: The CSV file path to save the coordinates.
    """
    # Create an empty list to hold the coordinates
    selected_tile_positions = []

    # Loop through the selected clusters and extract the tile positions
    for cluster_id in selected_clusters:
        cluster_indices = np.where(clusters == cluster_id)[0]
        for idx in cluster_indices:
            tile = tile_df.iloc[idx]
            selected_tile_positions.append([tile['w'], tile['h'], cluster_id])  # Store (w, h, cluster_id)

    # Convert the list of positions to a DataFrame
    selected_tile_df = pd.DataFrame(selected_tile_positions, columns=['w', 'h', 'cluster_id'])

    # Save to a CSV file
    selected_tile_df.to_csv(output_filename, index=False)
    print(f"Saved tile positions for clusters {selected_clusters} to {output_filename}")
        
def main(root_path, sample_names, tile_size=128):
    for sample_name in sample_names:
        print(f'Processing sample: {sample_name}')
        
        # Load Zarr data and channel metadata
        data, channels, autofluorescence_channel = load_data_from_zarr(root_path, sample_name)
        
        # Load tile positions for the current image (assumes position CSV exists)
        tile_df = load_tile_info(root_path, sample_name, tile_size)

        # Extract features for clustering
        features = extract_features(data, channels, tile_df, tile_size)

        # Perform clustering and PCA analysis
        clusters, pca_result, colors = perform_clustering_and_plot(features, n_clusters=30)

        # Plot the tile positions with autofluorescence image and cluster visualization
        plot_cluster_tile_positions_with_autofluorescence(tile_df, clusters, colors, autofluorescence_channel, sample_name)
        
        # Save tile positions from selected clusters (e.g., clusters 0, 1, and 2)
        selected_clusters = [13, 18, 19,20]  # You can modify this list to select specific clusters
        output_filename = f'{sample_name}_selected_clusters.csv'
        save_tiles_from_clusters(tile_df, clusters, selected_clusters, output_filename)
        
        plot_tile_count_per_cluster(clusters, n_clusters=30, sample_name=sample_name)

        # Generate channel correlation plots
        #plot_channel_correlation(tile_df, data, channels, tile_size, sample_name, output_dir='plots')


# Main execution
if __name__ == "__main__":
    root_path = '/gpfs/scratch/tm3475'
    sample_names = [
        '20230513_KPM-PB-17_18_Scan1T.ome',
        '20230513_KPM-PB-15_Scan1T.ome',
        '20230513_KPM-PB-17_18_Scan1.ome',
        '20230513_KPM-PB-14_Scan1T.ome',
        '20230513_KPM-PB-20_Scan1T.ome'
    ]

    main(root_path, sample_names, tile_size=128)
