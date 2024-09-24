
import numpy as np
import torch
from skimage import exposure


def minmax_rescale(arr):
    """Rescale the array values to the range [0, 255] for each individual array."""
    # Create an empty array to store the rescaled values
    rescaled_arr = np.zeros_like(arr, dtype=np.uint8)

    # Iterate through the first dimension (assuming it represents the number of arrays)
    for i in range(arr.shape[2]):
        # Extract the individual array
        single_band_arr = arr[:, :, i]

        # Calculate min and max values for the individual array
        min_val = single_band_arr.min()
        max_val = single_band_arr.max()

        # Rescale the individual array to the range [0, 255]
        scaled_arr = ((single_band_arr - min_val) * (1/(max_val - min_val) * 255)).to(dtype=torch.uint8)

        # Update the rescaled_arr with the individual rescaled array
        rescaled_arr[:, :, i] = scaled_arr

    return rescaled_arr

def percentile_rescale(arr, pct_list):
    """
    Rescale the array values to the range [0, 255] for each individual array using percentiles.
    """
    # Create an empty array to store the rescaled values
    rescaled_arr = np.zeros_like(arr, dtype=np.uint8)

    # Iterate through the first dimension
    for i in range(arr.shape[2]):
        # Extract the individual array
        single_band_arr = arr[:, :, i]

        # Calculate the 2nd and 98th percentiles for the individual band
        p2, p98 = np.percentile(single_band_arr, (pct_list[0], pct_list[1]))

        # Rescale the individual array using percentiles
        scaled_arr = exposure.rescale_intensity(np.array(single_band_arr), in_range=(p2, p98))

        # Scale the values to the range [0, 255] and convert to uint8
        scaled_arr = (scaled_arr * 255).astype(np.uint8)

        # Update the rescaled_arr with the individual rescaled array
        rescaled_arr[:, :, i] = scaled_arr

    return rescaled_arr

def histogram_rescale(arr):
    """
    Rescale the array values to the range [0, 255] for each individual array using histogram equalization.
    """
    # Create an empty array to store the rescaled values
    rescaled_arr = np.zeros_like(arr, dtype=np.uint8)

    # Iterate through the first dimension
    for i in range(arr.shape[2]):
        # Extract the individual array
        single_band_arr = arr[:, :, i]

        # Apply histogram equalization to the individual array
        equalized_arr = exposure.equalize_hist(np.array(single_band_arr))

        # Scale the values to the range [0, 255] and convert to uint8
        scaled_arr = (equalized_arr * 255).astype(np.uint8)

        # Update the rescaled_arr with the individual rescaled array
        rescaled_arr[:, :, i] = scaled_arr

    return rescaled_arr

def adaptive_histogram_rescale(arr, clip_limit=0.03):
    """
    Rescale the array values to the range [0, 255] for each individual array using adaptive histogram equalization.
    """
    # Create an empty array to store the rescaled values
    rescaled_arr = np.zeros_like(arr, dtype=np.uint8)

    # Iterate through the first dimension
    for i in range(arr.shape[2]):
        # Extract the individual array
        single_band_arr = arr[:, :, i]

        # Apply adaptive histogram equalization to the individual array
        adaptive_equalized_arr = exposure.equalize_adapthist(np.array(single_band_arr), clip_limit=clip_limit)

        # Scale the values to the range [0, 255] and convert to uint8
        scaled_arr = (adaptive_equalized_arr * 255).astype(np.uint8)

        # Update the rescaled_arr with the individual rescaled array
        rescaled_arr[:, :, i] = scaled_arr

    return rescaled_arr
