# Image-Clustering-and-Visualization-Toolkit

#### This repository provides a set of Python scripts for processing multi-channel imaging data, converting TIFF files to Zarr format, generating tiles from large slides, performing clustering and PCA analysis, and visualizing the results

![#f03c15](https://www.iconsdb.com/icons/download/color/f03c15/circle-16.png) `Overview`

<ul>
  <li>Conversion from TIFF to Zarr Format:</li>
  <ul>
    <li>Setup input and output paths</li>
    <ul>
      <li>Define the file paths for TIFF and Zarr files</li>
      <li>Create directories if they don't exist</li>
    </ul>
    <li>Check if the Zarr file already exists</li>
    <ul>
      <li>If the file exists, skip conversion</li>
    </ul>
    <li>Process channel information</li>
    <ul>
      <li>Check if a channel file exists</li>
      <li>If it exists, use custom channels from the file</li>
      <li>If it doesn't exist, use default channels provided in the function</li>
    </ul>
    <li>Read TIFF file data</li>
    <ul>
      <li>Load the TIFF file using <code>tifffile</code> library</li>
      <li>Convert the image data to an integer type</li>
    </ul>
    <li>Check if the number of channels in the TIFF matches the provided channels</li>
    <ul>
      <li>If mismatch occurs, raise an assertion error</li>
    </ul>
    <li>Write channels to a CSV file</li>
    <ul>
      <li>Write each channel's name and marker to a CSV</li>
    </ul>
    <li>Convert image data to Zarr format</li>
    <ul>
      <li>Use the <code>zarr.array</code> function to write data in chunks</li>
    </ul>
    <li>Print success message</li>
  </ul>

  <li>Tile Generation:</li>
  <ul>
    <li>Setup output directory for tiles</li>
    <ul>
      <li>Create the directory if it doesn't exist</li>
    </ul>
    <li>Read the Zarr file containing the image data</li>
    <ul>
      <li>Load image data from Zarr file</li>
    </ul>
    <li>Generate a thumbnail image from the slide data</li>
    <ul>
      <li>Downscale the image and remove bright pixels</li>
    </ul>
    <li>Generate a mask for the thumbnail</li>
    <ul>
      <li>Create a binary mask based on a brightness threshold</li>
    </ul>
    <li>Generate and save tile positions</li>
    <ul>
      <li>Split the slide into tiles based on the mask</li>
      <li>Save tile positions in a CSV file</li>
    </ul>
  </ul>

  <li>Feature Extraction and Clustering:</li>
  <ul>
    <li>Load Zarr data and channels</li>
    <ul>
      <li>Load data and channel information from Zarr and CSV files</li>
    </ul>
    <li>Extract features from tiles</li>
    <ul>
      <li>For each tile, calculate density of pixels for each channel</li>
    </ul>
    <li>Compute the Within-Cluster Sum of Squares (WCSS) for KMeans clustering</li>
    <ul>
      <li>Run KMeans clustering on extracted features</li>
    </ul>
    <li>Perform PCA on features and plot clustering results</li>
    <ul>
      <li>Apply PCA to reduce features to 2D for visualization</li>
      <li>Plot the clustering results with colors representing different clusters</li>
    </ul>
  </ul>

  <li>Cluster Visualization:</li>
  <ul>
    <li>Increase brightness of autofluorescence channel</li>
    <ul>
      <li>Enhance brightness by scaling pixel values</li>
    </ul>
    <li>Plot tile positions on the image with autofluorescence channel</li>
    <ul>
      <li>Scale tile positions to match the image dimensions</li>
      <li>Visualize tile positions with cluster colors</li>
    </ul>
  </ul>

  <li>Correlation Analysis:</li>
  <ul>
    <li>Generate a correlation matrix for channel densities</li>
    <ul>
      <li>Calculate Pearson’s correlation coefficient between channel densities</li>
      <li>Plot the correlation matrix as a heatmap</li>
    </ul>
    <li>Generate scatter plots of channel density correlations</li>
    <ul>
      <li>Plot scatter plots for each pair of channels</li>
      <li>Display Pearson’s r value on each plot</li>
    </ul>
    <li>Generate histograms of channel density distributions</li>
    <ul>
      <li>Plot histograms of density values for each channel</li>
    </ul>
  </ul>
</ul>



