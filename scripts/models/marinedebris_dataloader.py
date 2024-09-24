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

from models.helper_functions import load_scenedata, get_band_info
from utils.point_sampling_methods import extract_marinedebris_points, cluster_marinedebris, \
    extract_prompts_skeleton, extract_all_prompts, extract_prompts_centroid

# TODO: Documentation
class SamForMarineDebris(Dataset):
    def __init__(self, sceneid, band_list=[4, 3, 2], equation='bc', sensor_type='S2B'):
        # Use load_scenedata to get the required paths
        tif_files, png_files = load_scenedata(sceneid)

        # Set the attributes
        self.image_paths = tif_files
        self.label_paths = png_files
        self.band_list = band_list
        self.equation = equation
        self.sceneid = sceneid
        self.sensor_type = sensor_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        band_list = self.band_list
        equation = self.equation
        sceneid = self.sceneid
        sensor_type = self.sensor_type

        # Define the function for reading all spectral bands
        def scene_to_raw(image_path):
            rast = rasterio.open(image_path).read()
            return np.transpose(rast, (1, 2, 0)) # Transpose to work with Resize() in self.transform
        
        def scene_to_rgb(input_path, band_list):
            # Open the Sentinel-2 raster file and convert into RGB image
            with rasterio.open(input_path) as src:
                # Read three bands from .tif file; Use idx [4, 3, 2] for true-colour image
                red_band = src.read(band_list[0])
                green_band = src.read(band_list[1])
                blue_band = src.read(band_list[2])

                # Stack the bands to create a false-color composite
                rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)

            # Create a PIL Image from the array
            return rgb_image
        
        def scene_to_ndi(input_path, band_list):
            # Calculate Normalized Difference Index (NDI)
            with rasterio.open(input_path) as src:
                # Read two bands from .tif file
                band1 = src.read(band_list[0])
                band2 = src.read(band_list[1])

                # Calculate the Normalized Difference Index
                ndi = (band1 - band2) / (band1 + band2)

                # Stack the NDI x3 to create a grayscale image as SAM requires 3-channel input
                rgb_image = np.stack([ndi, ndi, ndi], axis=-1)

                return rgb_image 
            
        def scene_to_ssi(input_path, band_list, sceneid, sensor_type):
            # Get zero-indexed list of central wavelengths
            s2_wavelengths = get_band_info(sceneid, sensor_type)

            lambda1 = s2_wavelengths[band_list[0] - 1]
            lambda2 = s2_wavelengths[band_list[1] - 1]
            lambda3 = s2_wavelengths[band_list[2] - 1]

            # Calculate Spectral Shape Index (SSI)
            with rasterio.open(input_path) as src:
                # Read three bands from .tif file
                band1 = src.read(band_list[0])
                band2 = src.read(band_list[1])
                band3 = src.read(band_list[2])

                # Calculate prime number
                band2_prime = band1 + (band3 - band1) * ((lambda2 - lambda1) / (lambda3 - lambda1))

                # Calculate SSI
                ssi = band2 - band2_prime

                # Stack the SSI x3 to create a grayscale image as SAM requires 3-channel input
                rgb_image = np.stack([ssi, ssi, ssi], axis=-1)

            return rgb_image

        def scene_to_top(input_path, band_list, sceneid):
            rgb_list = []

            # Unpack individual tuples and create false colour composite
            for band_combination in band_list:
                if len(band_combination) > 2:
                    # Calculate the Spectral Shape Index
                    ssi = scene_to_ssi(input_path, band_combination, sceneid)[:, :, 0] # Extracts single SSI from shape (128, 128, 3)
                    rgb_list.append(ssi)
                else:
                    # Calculate the Normalized Difference Index
                    ndi = scene_to_ndi(input_path, band_combination)[:, :, 0] 
                    rgb_list.append(ndi)

            rgb_image = np.stack(rgb_list, axis=-1)
            return rgb_image
        
        def scene_to_pca(input_path):
            with rasterio.open(input_path) as src:
                # Read all bands
                bands = src.read()

                # Reshape the array into [num_pixels, num_bands]
                bands_reshaped = bands.transpose(1, 2, 0).reshape(-1, bands.shape[0])

                # Perform PCA
                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(bands_reshaped)

                # Print statistics
                #print(f"Relative variance explained in the principal components: {np.round(pca.explained_variance_ratio_, 2)}")

                # Reshape back to original image dimensions
                return pca_result.reshape(bands.shape[1], bands.shape[2], -1)
        
        def scene_to_fdi(input_path, band_list, sceneid, sensor_type):
            # Calculate the Floating Debris Index which is similar to the SSI equation
            # Main difference is that FDI uses four bands as it includes the central wavelength for the red band
            # Therefore, we manually implement it.

            # Get dictionary of all wavelengths
            band_dict = get_band_info(sceneid, sensor_type)

            # Extract central wavelength corresponding to the band
            lambda_nir = band_dict[band_list[0] - 1]    # B8
            lambda_swir1 = band_dict[band_list[2] - 1]  # B11

            lambda_red = band_dict[4 - 1] # Hard-coded as this is unique to FDI

            with rasterio.open(input_path) as src:
                # Read three bands from .tif file
                nir = src.read(band_list[0])    # B8
                red2 = src.read(band_list[1])   # B6
                swir1 = src.read(band_list[2])  # B11

                # Calculate NIR prime
                nir_prime = red2 + (swir1 - red2) * 10 * (lambda_nir - lambda_red) / (lambda_swir1 - lambda_red)

                # Calculate FDI
                fdi = nir - nir_prime

                # Stack the FDI x3 to create a grayscale image
                rgb_image = np.stack([fdi, fdi, fdi], axis=-1)

                return rgb_image 
            
        # Define visualization module options
        equation_functions = {
            'none':(scene_to_raw, [image_path]),                     # Read in raw Sentinel-2 scene
            'bc':  (scene_to_rgb, [image_path, band_list]),          # Band Composite
            'ndi': (scene_to_ndi, [image_path, band_list]),          # Normalized Difference Index
            'ssi': (scene_to_ssi, [image_path, band_list, sceneid, sensor_type]), # Spectral Shape Index
            'top': (scene_to_top, [image_path, band_list, sceneid]), # RSI-top10
            'pca': (scene_to_pca, [image_path]),                     # Principal Component Analysis
            'fdi': (scene_to_fdi, [image_path, band_list, sceneid, sensor_type])  # Floating Debris Index
        }
        
        # Check if the given vizualization method is valid
        assert equation in equation_functions, f"Invalid visualization module selected: {equation}"
        image = equation_functions[equation][0](*equation_functions[equation][1])

        # Load other variables
        label = transforms.PILToTensor()(Image.open(label_path))
        patchid = re.findall(r'\d+', image_path)[-1]

        def retrieve_point_prompts(sceneid):
            # Check if a .shp file exists for the given sceneid
            shapefile_path = glob.glob(os.path.join("data/", sceneid, "*_pt.shp"))
            
            if shapefile_path:
                #print(f"Shapefile with point prompts found at '{shapefile_path}'")
                # If the .shp file exists, execute the extract_marinedebris_points function.
                 point_prompts, point_labels = extract_marinedebris_points(shapefile_path, image_path, label, prompt_type='positive')
            else:
                #print("No shapefile for point prompts found (*_pt.shp). K-means (K=10) is executed for generating point prompts")

                # Execute K-means on each individual RGB channel to generate 10 prompts
                # Note: Plotting is set to False to speed up computation. Set to True for clustering results
                point_prompts, point_labels = cluster_marinedebris(image, label, num_clusters=10, prompt_type='both', plotting=False) # K-means

                # Other options for semi-automated approaches
                #point_prompts, point_labels = extract_all_prompts(label, min_pixel_count=10, prompt_type='positive')  # Random sampling 
                #point_prompts, point_labels = extract_prompts_centroid(label, min_pixel_count=0, prompt_type='both')  # Centroids
                #point_prompts, point_labels = extract_prompts_skeleton(label, min_pixel_count=10, prompt_type='both') # Skeletons
            return point_prompts, point_labels
    
        # Retrieve point prompts
        point_prompts, point_labels = retrieve_point_prompts(sceneid)

        return image, label, point_prompts, point_labels, patchid