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


def percentile_rescale(tensor, pct_list):
    """
    Rescale the tensor values to the range [0, 255] for each individual band using percentiles.
    """
    # Create an empty tensor to store the rescaled values
    rescaled_tensor = torch.zeros_like(tensor, dtype=torch.uint8)

    # Iterate through the first dimension (assuming the tensor shape is [C, H, W])
    for i in range(tensor.shape[0]):
        # Extract the individual band tensor
        single_band_tensor = tensor[i, :, :]

        # Calculate the 2nd and 98th percentiles for the individual band
        p2 = torch.quantile(single_band_tensor, pct_list[0] / 100)
        p98 = torch.quantile(single_band_tensor, pct_list[1] / 100)

        # Rescale the individual band tensor using percentiles
        scaled_tensor = (single_band_tensor - p2) / (p98 - p2)
        scaled_tensor = torch.clip(scaled_tensor, 0, 1)  # Ensure values are in the range [0, 1]

        # Scale the values to the range [0, 255] and convert to uint8
        scaled_tensor = (scaled_tensor * 255).to(torch.uint8)

        # Update the rescaled_tensor with the individual rescaled band
        rescaled_tensor[i, :, :] = scaled_tensor

    return rescaled_tensor

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
