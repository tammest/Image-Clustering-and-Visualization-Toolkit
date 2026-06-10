# Image Clustering and Visualization Toolkit

## Overview

This repository contains scripts for processing and analyzing multiplex imaging datasets. The workflow includes image conversion, tile generation, feature extraction, clustering, dimensionality reduction, and visualization of tissue features.

The toolkit was developed for analysis of multiplex fluorescence imaging datasets stored in OME-TIFF and OME-Zarr formats.

---

## Associated Manuscript

This repository accompanies the manuscript:

**Spatially resolved clustering identifies tissue microenvironments in multiplex fluorescence imaging datasets**

BioRxiv (2025)

https://www.biorxiv.org/content/10.1101/2025.08.19.671136v1

If you use this repository in your work, please cite the manuscript above.

---

## Main Features

- Convert OME-TIFF images to OME-Zarr format
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

## Repository Structure

```text
Image-Clustering-and-Visualization-Toolkit/
│
├── preprocessing/
│   ├── clustering_certain_clusters.py
│   ├── clustering_plotting.py
│   ├── gen_tiles.py
│   ├── individual_clusters.py
│   ├── read_OME_TIFF.py
│   └── tiff_to_zarr.py
│
├── Figure 4/
├── Figure 5/
├── Figure 6/
├── Figure 7/
│
├── environment.yml
├── LICENSE
└── README.md
```

The repository is organized into preprocessing scripts and figure-specific analysis directories. Scripts used to generate figures in the associated manuscript are located in their corresponding figure folders.

---

## Analyses Included

The repository contains scripts for:

- Tile generation from whole-slide images
- Density and intensity feature extraction
- Cluster and supercluster analysis
- PCA visualizations
- UMAP visualizations
- Marker co-localization analysis
- Positive and negative tile comparisons
- Genotype-specific comparisons (WT vs db/db)
- Publication-quality figure generation

---

## License

This project is distributed under the GPL-3.0 License.

See the `LICENSE` file for details.
