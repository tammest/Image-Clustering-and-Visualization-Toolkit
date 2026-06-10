# Image Clustering and Visualization Toolkit

## Overview

This repository contains scripts for processing and analyzing multiplex imaging datasets. The workflow includes image conversion, tile generation, feature extraction, clustering, dimensionality reduction, and visualization of tissue features.

The toolkit was developed for analysis of multi-channel fluorescence imaging data stored in OME-TIFF and OME-Zarr formats.

---

## Main Features

- Convert OME-TIFF images to Zarr format
- Generate tiles from large whole-slide images
- Extract density and intensity features from tiles
- Perform clustering analysis
- Generate PCA and UMAP visualizations
- Analyze tissue layers and superclusters
- Create cluster maps and tissue overlays
- Generate publication-quality figures

---

## Installation

Clone the repository:

```bash
git clone https://github.com/tammest/Image-Clustering-and-Visualization-Toolkit.git
cd Image-Clustering-and-Visualization-Toolkit
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate clustering_env
```

---

## Environment

Create an `environment.yml` file:

```yaml
name: clustering_env

channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.9
  - numpy=1.26.4
  - pandas=1.5.3
  - matplotlib
  - seaborn
  - scikit-learn
  - umap-learn
  - zarr
  - scanpy
  - anndata
  - tqdm
  - pip
  - pip:
      - pycirclize
```

---

## Workflow

### 1. TIFF to Zarr Conversion

OME-TIFF images are converted to Zarr format for efficient storage and access. Channel information is preserved and stored alongside the image data.

**Outputs**

- `data.zarr`
- Channel metadata files

### 2. Tile Generation

Whole-slide images are divided into smaller tiles for downstream analysis. Tissue-containing regions are identified using thumbnail masks.

**Outputs**

- Tile coordinate files
- Thumbnail masks

### 3. Feature Extraction

Features are calculated for each tile, including marker density and mean intensity measurements.

**Outputs**

- Feature tables (`.csv`)

### 4. Clustering and Dimensionality Reduction

Tile features can be clustered using K-means and visualized using PCA or UMAP.

**Outputs**

- Cluster assignments
- PCA plots
- UMAP embeddings

### 5. Tissue Layer and Supercluster Analysis

Clusters can be grouped into broader tissue categories based on marker expression patterns.

**Outputs**

- Tissue annotations
- Supercluster assignments

### 6. Visualization

The repository includes scripts for generating:

- Cluster maps
- Tissue overlays
- Correlation heatmaps
- UMAP plots
- Histograms
- Example tile visualizations
- Publication figures

---

## Input Data

Typical inputs include:

- OME-TIFF images
- OME-Zarr datasets
- Tile coordinate files
- Feature tables (`.csv`)

---

## Output Files

Examples of generated outputs include:

- Zarr datasets
- Feature tables
- Cluster annotations
- UMAP embeddings
- Correlation plots
- PNG figures
- PDF figures

---

## Dependencies

Main packages used in this repository:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- UMAP-learn
- Zarr
- Scanpy
- AnnData

---

## References

- tifffile: https://pypi.org/project/tifffile/
- Zarr: https://zarr.readthedocs.io/
- Scikit-learn: https://scikit-learn.org/
- UMAP: https://umap-learn.readthedocs.io/
- Scanpy: https://scanpy.readthedocs.io/

---
This work is available as a bioRxiv preprint:
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.08.19.671136v1-B31B1B.svg)](https://www.biorxiv.org/content/10.1101/2025.08.19.671136v1)

