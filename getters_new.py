"""
To ease the pain of ensuring compatibility with new data structures or datasets,
this file collects key IO functions for data, metadata, and annotations
that may be edited by a user to fit their particular use case.
"""

import h5py
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import tifffile

from PIL import Image
import matplotlib.pyplot as plt
    
def get_slice(dataset: Path, t: int) -> np.ndarray:
    """Return a slice at specified index t.
    This should return a 4-D numpy array containing multi-channel volumetric data
    with the dimensions ordered as (C, Z, Y, X).
    """
    frame_rate = 25  # frames per second

    # Calculate the file index based on time index and frame rate
    # t is assumed to be in seconds
    file_index = int(t * frame_rate)

    # Format the filename with zero-padding
    file_name = f"{file_index:08d}.tif"
    file_path = dataset / file_name

    with tifffile.TiffFile(file_path) as tif:
        data_slice = tif.asarray()

        # Assuming data_slice is 2D, expand it to 4D (C, Z, Y, X)
        # This example assumes a single-channel image (hence first dimension is 1)
        return np.expand_dims(np.expand_dims(data_slice, axis=0), axis=0)

# # Set the dataset path
# dataset_path = Path(r"D:\RIA calcium imaging\WEN0231-widefield\ctl\20231020-RIActl\w11_2023-11-08_20-08-27\0_Camera-Red_VSC-10629")

# # Get the frame at 10 seconds
# frame_test = get_slice(dataset_path, 0)
# frame_test_2D = frame_test[0, 0, :, :]
# plt.imshow(frame_test_2D, cmap='gray')
# plt.show()

def get_annotation_df(dataset: Path) -> pd.DataFrame:
    """Load and return annotations as an ordered pandas dataframe.
    This should contain the following:
    - t_idx: time index of each annotation
    - x: x-coordinate as a float between (0, 1)
    - y: y-coordinate as a float between (0, 1)
    - z: z-coordinate as a float between (0, 1)
    - worldline_id: track or worldline ID as an integer
    - provenance: scorer or creator of the annotation as a byte string
    """
    with h5py.File(dataset / 'annotations.h5', 'r') as f:
        data = pd.DataFrame()
        for k in f:
            data[k] = f[k]
    return data


# def get_metadata(dataset: Path) -> dict:
#     """Load and return metadata for the dataset as a Python dictionary.
#     This should contain at least the following:
#     - shape_t
#     - shape_c
#     - shape_z
#     - shape_y
#     - shape_x
#     """
#     json_filename = dataset / "metadata.json"
#     with open(json_filename) as json_file:
#         metadata = json.load(json_file)
#     return metadata

def get_metadata(dataset: Path) -> dict:
    """Load and return metadata for the dataset as a Python dictionary.
    This should contain at least the following:
    - shape_t
    - shape_c
    - shape_z
    - shape_y
    - shape_x
    """        
    # Count the number of files
    num_files = len(list(dataset.glob('*.tif')))

    # Frame rate
    frame_rate = 25  # fps

    # Calculate duration in seconds
    duration = num_files / frame_rate

    # Get dimensions of the first image (assuming all images are the same size)
    sample_file = dataset / "00000000.tif"
    with tifffile.TiffFile(sample_file) as tif:
        sample_image = tif.asarray()
        height, width = sample_image.shape

    # Compile metadata
    metadata = {
        "total_frames": num_files,
        "frame_rate": frame_rate,
        "duration_seconds": duration,
        "image_width": width,
        "image_height": height,
        "file_pattern": "00000000.tif to 00010142.tif",
        "dataset_path": str(dataset)
    }

    return metadata

# Set the dataset path
dataset_path = Path(r"D:\RIA calcium imaging\WEN0231-widefield\ctl\20231020-RIActl\w11_2023-11-08_20-08-27\0_Camera-Red_VSC-10629")

# Get metadata
metadata = get_metadata(dataset_path)
print(metadata)