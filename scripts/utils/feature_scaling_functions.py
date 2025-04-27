import numpy as np
import torch
from skimage import exposure

def minmax_rescale(tensor):
    """Rescale the array values to the range [0, 255] for each individual array."""
    # Create an empty tensor to store the rescaled values
    rescaled_tensor = torch.zeros_like(tensor, dtype=torch.uint8)

    # Iterate through the first dimension (assuming the tensor shape is [C, H, W])
    for i in range(tensor.shape[0]):
        # Extract the individual band tensor
        single_band_tensor = tensor[i, :, :]

        # Calculate the min and max values for the individual band
        min_val = single_band_tensor.min()
        max_val = single_band_tensor.max()

        # Rescale the individual band tensor using min-max scaling
        if max_val - min_val == 0:
            scaled_tensor = torch.zeros_like(single_band_tensor, dtype=torch.uint8)
        else:
            scaled_tensor = (single_band_tensor - min_val) / (max_val - min_val)
            scaled_tensor = torch.clip(scaled_tensor, 0, 1)  # Ensure values are in the range [0, 1]
            scaled_tensor = (scaled_tensor * 255).to(torch.uint8)  # Scale to [0, 255] and convert to uint8

        # Update the rescaled_tensor with the individual rescaled band
        rescaled_tensor[i, :, :] = scaled_tensor

    return rescaled_tensor

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

def histogram_rescale(tensor):
    """
    Rescale the array values to the range [0, 255] for each individual array using histogram equalization.
    """
    # Create an empty tensor to store the rescaled values
    rescaled_tensor = torch.zeros_like(tensor, dtype=torch.uint8)

    # Iterate through the first dimension (assuming the tensor shape is [C, H, W])
    for i in range(tensor.shape[0]):
        # Extract the individual band tensor
        single_band_tensor = tensor[i, :, :]

        # Convert the tensor to a NumPy array for histogram equalization
        single_band_arr = single_band_tensor.cpu().numpy()

        # Apply histogram equalization to the individual array
        equalized_arr = exposure.equalize_hist(single_band_arr)

        # Scale the values to the range [0, 255] and convert to uint8
        scaled_arr = (equalized_arr * 255).astype(np.uint8)

        # Convert the scaled array back to a tensor
        scaled_tensor = torch.from_numpy(scaled_arr)

        # Update the rescaled_tensor with the individual rescaled band
        rescaled_tensor[i, :, :] = scaled_tensor

    return rescaled_tensor

def adaptive_histogram_rescale(tensor, clip_limit=0.03):
    """
    Rescale the array values to the range [0, 255] for each individual array using adaptive histogram equalization.
    """
    # Create an empty tensor to store the rescaled values
    rescaled_tensor = torch.zeros_like(tensor, dtype=torch.uint8)

    # Iterate through the first dimension (assuming the tensor shape is [C, H, W])
    for i in range(tensor.shape[0]):
        # Extract the individual band tensor
        single_band_tensor = tensor[i, :, :]

        # Convert the tensor to a NumPy array for adaptive histogram equalization
        single_band_arr = single_band_tensor.cpu().numpy()

        # Apply adaptive histogram equalization to the individual array
        adaptive_equalized_arr = exposure.equalize_adapthist(single_band_arr, clip_limit=clip_limit)

        # Scale the values to the range [0, 255] and convert to uint8
        scaled_arr = (adaptive_equalized_arr * 255).astype(np.uint8)

        # Convert the scaled array back to a tensor
        scaled_tensor = torch.from_numpy(scaled_arr)

        # Update the rescaled_tensor with the individual rescaled band
        rescaled_tensor[i, :, :] = scaled_tensor

    return rescaled_tensor
