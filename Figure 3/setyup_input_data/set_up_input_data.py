import pandas as pd
import os
from pathlib import Path

#===============================================================
# READ DATA FOR METRICS WITH A THRESHOLD VALUE
#===============================================================
# --- configure these for desired csv --- 

algorithm = "leiden"
calculation = "density_pixels_above_1"
threshold = "0.4_threshold"

df = pd.read_csv("rabbanilab/wsi_analysis/input_csv_files/n_of_3.csv")

# --- establishes the input csv based off above configurables --- 

output_path_14 = Path(f"rabbanilab/wsi_analysis/data/clustering/{threshold}/feature_distribution/{calculation}/kde_histograms/per_sample_group")
output_path_15 = Path(f"rabbanilab/wsi_analysis/data/clustering/{threshold}/tile_counts/{calculation}")
output_path_16 = Path(f"rabbanilab/wsi_analysis/data/clustering/{threshold}/pcas/{calculation}/full_plots")
os.makedirs(output_path_14, exist_ok=True)
os.makedirs(output_path_15, exist_ok=True)
os.makedirs(output_path_16, exist_ok=True)
