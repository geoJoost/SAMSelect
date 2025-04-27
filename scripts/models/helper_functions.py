import os
import glob
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from PIL import Image
import numpy as np
import torch

def load_scenedata(tif_path, polygon_path, patch_size):
    # Define cache filename
    sceneid = os.path.splitext(os.path.basename(tif_path))[0]
    cache_dir = os.path.join("data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_patches = os.path.join(cache_dir, f"{sceneid}_patches.pt")
    cache_file_masks = os.path.join(cache_dir, f"{sceneid}_masks.pt")
    
    # Define output, in case no cache is found
    patches, masks = [], []

    # Check if patches/masks are initialized beforehand
    if os.path.exists(cache_file_patches):
        #print("[INFO]: Files are pre-processed. Loading cache.")
        patches = torch.load(cache_file_patches)
        masks = torch.load(cache_file_masks)
        return patches, masks
        
    print(f"[INFO] Sentinel-2 scene is not pre-processed. Processing into patches ({patch_size}px) based on polygon annotations.")
    # If not, process the Sentinel-2 scene into smaller image patches
    # Read the files
    polygons = gpd.read_file(polygon_path)
    with rasterio.open(tif_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        
        polygons = polygons.to_crs(src.crs)
                     
        # First, we create smaller patches of the Sentinel-2 scene
        # Get window size in geographic coordinates
        window_size = patch_size * transform[0] # transform[0] gets pixel size in meters
        
        # Calculate the bounds of the entire scene
        minx, miny, maxx, maxy = polygons.total_bounds
        
        # Calculate the width and height of the bounding box
        width = maxx - minx
        height = maxy - miny
        
        # Check if the polygon bounds are smaller than the window size
        if width <= window_size and height <= window_size:
            # Calculate the centroid of the polygons
            centroid = polygons.geometry.unary_union.centroid
            cx, cy = centroid.x, centroid.y

            # Create a new window centered on the centroid
            half_window = window_size / 2
            window = rasterio.windows.from_bounds(
                cx - half_window, cy - half_window,
                cx + half_window, cy + half_window,
                transform
            )

            # Read the image data within the window
            # This method is necessary to guarantee the final patch has a shape equal to that of the mask
            patch = src.read(window=window)

            # Rasterize the polygon
            rasterized = rasterize(
                [(geom, 1) for geom in polygons.geometry],
                out_shape=(patch_size, patch_size),
                transform=rasterio.windows.transform(window, transform),
                fill=0,
                all_touched=True,  # Set to False for smaller labels
                dtype='uint8'
            )

            # Rescale [0, 1] to [0, 255] for Segment Anything
            mask = (rasterized * 255).astype(np.uint8)

            # Add to list for storage
            patches.append(patch)
            masks.append(mask)
        
        # For larger extents, create a uniform grid to sample patches from
        else:
            # Create a uniform grid of points based on patch_size
            x_points = np.arange(minx, maxx, window_size)
            y_points = np.arange(miny, maxy, window_size)

            # Iterate through each grid point
            for x in x_points:
                for y in y_points:
                    window = rasterio.windows.from_bounds(
                        x, y,
                        x + window_size, y + window_size,
                        transform)

                    # Read the image data within the window
                    patch = src.read(window=window)

                    # Clip the annotations to image patch
                    window_bounds = (x, y, x + window_size, y + window_size)
                    polygons_clip = gpd.clip(polygons, box(*window_bounds))

                    # Rasterize the clipped polygon
                    rasterized = rasterize(
                        [(geom, 1) for geom in polygons_clip.geometry],
                        out_shape=(patch_size, patch_size),
                        transform=rasterio.windows.transform(window, transform), #transform,
                        fill=0,
                        all_touched=True,  # Set to False for smaller labels
                        dtype='uint8'
                    )

                    # Rescale [0, 1] to [0, 255] for Segment Anything
                    mask = (rasterized * 255).astype(np.uint8)

                    # Check if the mask contains more than 10 pixels labeled
                    if np.sum(mask == 255) > 10:
                        # Add to list for storage
                        patches.append(patch)
                        masks.append(mask)

    # Convert to tensor and save as Pytorch cache's
    patches = torch.tensor(np.array(patches), dtype=torch.float32) # (N, C, patch_size, patch_size)
    masks = torch.tensor(np.array(masks)).unsqueeze(1) # [N, 1, patch_size, patch_size]
    
    torch.save(patches, cache_file_patches)
    torch.save(masks, cache_file_masks)
        
    return patches, masks

def get_band_info(atm_level):
    assert atm_level in ('L1', 'L2A', 'L2R'), f"Invalid atmospheric corrected product '{atm_level}. Please insert one of the following products: Sen2Cor (L1C, L2A) or ACOLITE (L1R, L2R)"

    # Sentinel-2B central wavelengths (nm)
    # Different with Sentinel-2A is negligible
    bands_dict = { 
    "B1": 442.2,    # Coastal Aerosol
    "B2": 492.1,    # Blue
    "B3": 559.0,    # Green
    "B4": 664.9,    # Red
    "B5": 703.8,    # Red Edge 1
    "B6": 739.1,    # Red Edge 2
    "B7": 779.7,    # Red Edge 3
    "B8": 832.9,    # NIR
    "B8A": 864.0,   # Narrow NIR
    "B9": 943.2,    # Water vapour
    "B10": 1376.9,  # Cirrus
    "B11": 1610.4,  # SWIR 1
    "B12": 2185.7   # SWIR 2
    }

    if atm_level == 'L2A': # Sen2Cor removes B10
        del bands_dict['B10']

    if atm_level == 'L2R': # ACOLITE removes B9 and B10
        del bands_dict['B9']
        del bands_dict['B10']

    return list(bands_dict.values()) # Convert to list as bands_idx does not allow gaps in-between for missing bands

def select_top_bands(sceneid, model_type, equation_list, top_number=10):
    """
    Selects the top-performing band combinations for the given scene and equations (Spectral Index Composite).

    Args:
        sceneid (str): The scene ID.
        model_type (str): The ViT-model.
        equation_list (list): List of equations to process (e.g., ['ndi', 'ssi']).
        top_number (int): Number of top band combinations to return.

    Returns:
        list: Top-performing band combinations.
    """
    top_combinations = pd.DataFrame()

    for equation in equation_list:
        # Construct the file path 
        file_path = f"data/processed/{sceneid}_{equation}_{model_type}_results.csv"
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Skipping '{equation}' as the file '{file_path}' does not exist")
            continue

        # Read in the processed data
        print(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)

        # Convert individual bands into tuple as this is expected for 'band_searchspace'
        if equation in ('ssi'):
            df['band_combination'] = df[['band_1', 'band_2', 'band_3']].apply(tuple, axis=1)
        elif equation in ('ndi'):
            df['band_combination'] = df[['band_1', 'band_2']].apply(tuple, axis=1)
        
        # Calculate the average Jaccard for the NDI/SSI combinations
        df_stats = df.groupby('band_combination')[['jaccard_lvl1', 'jaccard_lvl2', 'jaccard_lvl3']].mean().reset_index()

        # Melt the dataset to get a single row for each unique Jaccard level
        df_melt = pd.melt(df_stats, id_vars='band_combination', value_vars=['jaccard_lvl1', 'jaccard_lvl2', 'jaccard_lvl3'], var_name='mask_level', value_name='miou')
        df_top = df_melt.nlargest(top_number, 'miou')['band_combination'] # Extract top-5 best performing bands on mIoU

        # Concatenate results
        top_combinations = pd.concat([top_combinations, df_top], axis=0)
    
    return top_combinations['band_combination'].tolist()

def get_atmospheric_level(tif_path, band_list):
    """
    Determines the atmospheric correction level based on the number of bands in a .tif file.

    Args:
        tif_path (str): The file-path of the Sentinel-2 scene.
        band_list (list): Expected list of bands for validation.

    Returns:
        str: Atmospheric correction level ('L1', 'L2A', 'L1R'), or None if bands are insufficient.
    """
    if not os.path.exists(tif_path):
            raise FileNotFoundError(f"The file '{tif_path}' does not exist. Stopping execution.")
   
    # Open the .tif file and check the number of bands
    with rasterio.open(tif_path) as src:
        num_bands = src.count
        
        # Validate band list against the number of bands
        if len(band_list) != num_bands:
            raise ValueError(f"Mismatch between provided band list: ({len(band_list)} bands) "
                            f"and number of bands in the file ({num_bands} bands).")
            
        # Constants for atmospheric correction levels
        ATMOSPHERIC_LEVELS = {
            13: 'L1',
            12: 'L2A',
            11: 'L2R'
        }
        return ATMOSPHERIC_LEVELS.get(num_bands, None)