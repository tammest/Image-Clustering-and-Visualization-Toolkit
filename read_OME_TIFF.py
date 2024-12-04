import xml.etree.ElementTree as ET
import tifffile
import zarr
import numpy as np
import pandas as pd
import imagecodecs
from skimage import transform
from skimage import data
from skimage import measure
from skimage.measure import block_reduce
from skimage.transform import resize
from skimage.io import imsave
import os
import tifffile
import zarr
import openslide
from matplotlib import pyplot as plt
from PIL import Image
#showing the image with my graph 
import tifffile
import matplotlib.pyplot as plt
import numpy as np

# Path to your OME-TIFF image
#image_path = 'trial1.ome.tif'
image_path =  '20230513_KPM-PB-11_Scan1T.ome.tif'

# Read OME-TIFF image using tifffile
image = tifffile.imread(image_path)

# Ensure image is in the correct shape (C, Y, X)
if image.ndim == 3 and image.shape[0] == 7:
    # Extract all channels
    all_channels = []
    for c in range(image.shape[0]):
        all_channels.append(image[c])

    # Stack channels to create RGB image
    rgb_image = np.stack([all_channels[2], all_channels[4], all_channels[0]], axis=-1)  # Adjust indices as needed

    # Display the RGB image
    plt.figure(figsize=(12, 6))

    # Plot RGB image
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)
    plt.title('RGB Composite Image')
    plt.axis('off')

    # Plot histograms for all channels
    plt.subplot(1, 2, 2)
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']  # Adjust as per your channel count
    labels = ['Channel 0 (Nuclei)', 'Channel 1 (SMA)', 'Channel 2 (Vimentin)', 'Channel 3 (F480)', 'Channel 4 (CD31)', 'Channel 5 (Ki67)', 'Channel 6 (Autofluorescence)']

    for i, channel_data in enumerate(all_channels):
        plt.hist(channel_data.flatten(), bins=50, color=colors[i], alpha=0.5, label=labels[i])

    plt.title('Intensity Histograms')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print(f"Invalid image shape: {image.shape}. Expected (C, Y, X) where C is number of channels.")
