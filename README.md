# Image-Clustering-and-Visualization-Toolkit

### Overview:
This repository provides a set of Python scripts for processing multi-channel imaging data, including:

- **Conversion** of TIFF files to Zarr format
- **Tile generation** from large slides
- **Clustering and PCA analysis**
- **Visualization** of clustering results and feature analysis

---

## Conversion from TIFF to Zarr Format

- **Setup input and output paths**
  - Define the file paths for TIFF and Zarr files
  - Create directories if they don’t exist

- **Check if the Zarr file already exists**
  - If the file exists, **skip conversion**

- **Process channel information**
  - Check if a channel file exists
  - If it exists, use custom channels from the file
  - If it doesn’t exist, use **default channels** provided in the function

- **Read TIFF file data**
  - Load the TIFF file using the `tifffile` library
  - Convert image data to an integer type

- **Validate channels**
  - Check if the number of channels in the TIFF matches the provided channels
  - If mismatch occurs, raise an **AssertionError**

- **Write channels to CSV**
  - Write each channel's name and marker to a CSV file

- **Convert image data to Zarr format**
  - Use the `zarr.array` function to write data in chunks

- **Print success message**
  - After conversion, print a success message

---

## Tile Generation

- **Setup output directory for tiles**
  - Create the directory if it doesn’t exist

- **Read the Zarr file containing image data**
  - Load image data from Zarr file

- **Generate a thumbnail image from the slide data**
  - Downscale the image and remove bright pixels

- **Generate a mask for the thumbnail**
  - Create a **binary mask** based on a brightness threshold

- **Generate and save tile positions**
  - Split the slide into tiles based on the mask
  - Save tile positions in a CSV file

---

## Feature Extraction and Clustering

- **Load Zarr data and channels**
  - Load data and channel information from Zarr and CSV files

- **Extract features from tiles**
  - For each tile, calculate pixel density for each channel

- **Compute Within-Cluster Sum of Squares (WCSS)**
  - Run KMeans clustering on extracted features

- **Perform PCA on features**
  - Apply PCA to reduce features to 2D for visualization
  - Plot clustering results with colors representing different clusters

---

## Cluster Visualization

- **Increase brightness of autofluorescence channel**
  - Enhance brightness by **scaling pixel values**

- **Plot tile positions on the image**
  - Scale tile positions to match image dimensions
  - Visualize tile positions with cluster colors

---

## Correlation Analysis

- **Generate a correlation matrix for channel densities**
  - Calculate **Pearson’s correlation coefficient** between channel densities
  - Plot the correlation matrix as a heatmap

- **Generate scatter plots of channel density correlations**
  - Plot scatter plots for each pair of channels
  - Display Pearson’s **r value** on each plot

- **Generate histograms of channel density distributions**
  - Plot histograms of density values for each channel

---

### Additional Enhancements:
- **Links to Documentation**: Respective libraries or related resources:
  - [tifffile documentation](https://pypi.org/project/tifffile/)
  - [Zarr Python documentation](https://zarr.readthedocs.io/en/stable/)
  - [scikit-learn clustering documentation](https://scikit-learn.org/stable/modules/clustering.html)

