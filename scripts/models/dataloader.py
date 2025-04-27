import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import rasterio
import numpy as np
import re
from sklearn.decomposition import PCA
import glob

from scripts.models.helper_functions import load_scenedata, get_band_info
from scripts.utils.point_sampling_methods import extract_manual_prompts, prompts_from_spectralclusters
# TODO: Documentation
class SamForMarineDebris(Dataset):
    def __init__(self, tif_path, polygon_path, band_list=[4, 3, 2], equation='bc', atm_level='L2A'):
        self.patch_size = 128
        self.band_list = band_list
        self.equation = equation
        self.atm_level = atm_level
        
        # Pre-process and cache the Sentinel-2 scene into smaller patches
        # If large polygons are used (i.e., exceeding 1280 meters), switch patch_size to appropriate size (e.g., 256px, 512px, to a maximum of 1024px)
        self.patches, self.masks = load_scenedata(tif_path, polygon_path, self.patch_size)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        mask = self.masks[idx]
        
        band_list = self.band_list
        equation = self.equation
        atm_level = self.atm_level

        # Define the function for reading all spectral bands
        def scene_to_raw(patch):
            return patch
                
        def scene_to_rgb(patch, band_list):
            # Extract specified bands from the tensors
            # Use idx [4, 3, 2] to retrieve true-colour image
            red_band = patch[band_list[0], :, :]
            green_band = patch[band_list[1], :, :]
            blue_band = patch[band_list[2], :, :]

            # Stack the bands to create a false-colour composite
            rgb_image = torch.stack([red_band, green_band, blue_band], dim=0)

            return rgb_image
            
        def scene_to_ndi(patch, band_list):
            # Extract the specified bands from the tensor
            band1 = patch[band_list[0], :, :]
            band2 = patch[band_list[1], :, :]

            # Calculate the Normalized Difference Index
            ndi = (band1 - band2) / (band1 + band2 + 1e-10)  # Add epsilon to avoid division by zero

            # Stack the NDI x3 to create a grayscale image as SAM requires 3-channel input
            rgb_image = torch.stack([ndi, ndi, ndi], dim=0)

            return rgb_image
            
        def scene_to_ssi(patch, band_list, atm_level):
            # Get zero-indexed list of central wavelengths
            s2_wavelengths = get_band_info(atm_level)

            lambda1 = s2_wavelengths[band_list[0] - 1]
            lambda2 = s2_wavelengths[band_list[1] - 1]
            lambda3 = s2_wavelengths[band_list[2] - 1]

            # Extract the specified bands from the tensor
            band1 = patch[band_list[0], :, :]
            band2 = patch[band_list[1], :, :]
            band3 = patch[band_list[2], :, :]

            # Calculate prime number
            band2_prime = band1 + (band3 - band1) * ((lambda2 - lambda1) / (lambda3 - lambda1))

            # Calculate SSI
            ssi = band2 - band2_prime

            # Stack the SSI x3 to create a grayscale image as SAM requires 3-channel input
            rgb_image = torch.stack([ssi, ssi, ssi], dim=0)

            return rgb_image

        def scene_to_top(patch, band_list, atm_level):
            rgb_list = []

            # Unpack individual tuples and create false colour composite
            for band_combination in band_list:
                if len(band_combination) > 2:
                    # Calculate the Spectral Shape Index
                    ssi = scene_to_ssi(patch, band_combination, atm_level)[0, :, :]  # Extracts single SSI from shape (3, 128, 128)
                    rgb_list.append(ssi)
                else:
                    # Calculate the Normalized Difference Index
                    ndi = scene_to_ndi(patch, band_combination)[0, :, :]
                    rgb_list.append(ndi)

            rgb_image = torch.stack(rgb_list, dim=0)
            return rgb_image
        
        def scene_to_pca(patch):
            # Reshape the tensor into [num_pixels, num_bands]
            bands_reshaped = patch.permute(1, 2, 0).reshape(-1, patch.size(0))

            # Perform PCA
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(bands_reshaped.cpu().numpy())  # Convert to numpy for PCA

            # Reshape back to original image dimensions
            pca_result = torch.tensor(pca_result).reshape(patch.size(1), patch.size(2), -1).permute(2, 0, 1)

            return pca_result

        def scene_to_fdi(patch, band_list, atm_level):
            # Calculate the Floating Debris Index which is similar to the SSI equation
            # Main difference is that FDI uses four bands as it includes the central wavelength for the red band
            # Therefore, we manually implement it.

            # Get dictionary of all wavelengths
            band_dict = get_band_info(atm_level)

            # Extract central wavelength corresponding to the band
            lambda_nir = band_dict[band_list[0] - 1]    # B8
            lambda_swir1 = band_dict[band_list[2] - 1]  # B11
            lambda_red = band_dict[4 - 1]  # Hard-coded as this is unique to FDI

            # Extract the specified bands from the tensor
            nir = patch[band_list[0], :, :]    # B8
            red2 = patch[band_list[1], :, :]   # B6
            swir1 = patch[band_list[2], :, :]  # B11

            # Calculate NIR prime
            nir_prime = red2 + (swir1 - red2) * 10 * (lambda_nir - lambda_red) / (lambda_swir1 - lambda_red)

            # Calculate FDI
            fdi = nir - nir_prime

            # Stack the FDI x3 to create a grayscale image
            rgb_image = torch.stack([fdi, fdi, fdi], dim=0)

            return rgb_image
            
        # Define visualization module options
        equation_functions = {
            'none':(scene_to_raw, [patch]),                     # Read in raw Sentinel-2 scene
            'bc':  (scene_to_rgb, [patch, band_list]),          # Band Composite
            'ndi': (scene_to_ndi, [patch, band_list]),          # Normalized Difference Index
            'ssi': (scene_to_ssi, [patch, band_list, atm_level]), # Spectral Shape Index
            'top': (scene_to_top, [patch, band_list, atm_level]), # RSI-top10
            'pca': (scene_to_pca, [patch]),                     # Principal Component Analysis
            'fdi': (scene_to_fdi, [patch, band_list, atm_level])  # Floating Debris Index
        }
        
        # Check if the given vizualization method is valid
        assert equation in equation_functions, f"Invalid visualization module selected: {equation}"
        image = equation_functions[equation][0](*equation_functions[equation][1])
        image = image.to(torch.float32) # To prevent issues with scaling later on, convert integers into floats

        # Load other variables
        patchid = idx + 1

        def retrieve_point_prompts(image):
            # Check if a .shp file exists for the given sceneid
            shapefile_path = glob.glob(os.path.join("data/*_pt.shp"))
            
            if shapefile_path:
                # If the .shp file exists, execute the extract_marinedebris_points function.
                # UNUSED: argument removed in favour for automated K-means point prompts
                point_prompts, point_labels = extract_manual_prompts(shapefile_path, image, mask, prompt_type='positive')
            else:
                # Execute K-means on each individual RGB channel to generate 10 prompts
                # Note: Plotting is set to False to speed up computation. Set to True for clustering results
                point_prompts, point_labels = prompts_from_spectralclusters(image, mask, num_clusters=10, prompt_type='both', plotting=False) # K-means

            return point_prompts, point_labels
    
        # Retrieve point prompts
        point_prompts, point_labels = retrieve_point_prompts(image)

        return image, mask, point_prompts, point_labels, patchid