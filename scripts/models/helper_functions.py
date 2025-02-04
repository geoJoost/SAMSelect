import os
import glob
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from PIL import Image
import numpy as np

def load_scenedata(sceneid):
    data_folder = os.path.join(f"data/", sceneid)

    # Use glob to find all .tif and .png files in the specified folder
    # The sorted is absolutely crucial as this matches the scene-data with reference
    tif_files = sorted(glob.glob(os.path.join(data_folder, '*.tif'))) # Sentinel-2 data
    png_files = sorted(glob.glob(os.path.join(data_folder, '*.png'))) # Reference masks

    # If no .png's are found, we create new reference masks
    # We check this to avoid recomputing labels for each spectral band combination
    if not png_files:
        create_reference_masks(data_folder, tif_files)

        # Now that labels are created, we can load in the data
        png_files = sorted(glob.glob(os.path.join(data_folder, '*.png')))

    if len(tif_files) != len(png_files):
        raise ValueError("Number of patches for scene-data and reference masks should be identical!")

    return tif_files, png_files

def create_reference_masks(data_folder, tif_files):
    # Read in the polygon shapefile with annotations
    shapefile_path = glob.glob(os.path.join(data_folder, "*qualitative*.shp"))[0]
    assert shapefile_path, "No shapefile containing 'qualitative' found in the data folder. Please use this naming convention for your polygon annotations."
    
    polygons = gpd.read_file(shapefile_path)

    for index, tif_file in enumerate(tif_files):
        with rasterio.open(tif_file) as src:
            polygons = polygons.to_crs(src.crs)

            # Clip the polygon to the current TIFF file's bounds
            clipped_polygon = polygons.clip(mask=src.bounds)

            # Rasterize the clipped polygon
            out_shape = (src.height, src.width)
            transform = src.transform
            rasterized = rasterize(
                [(geom, 1) for geom in clipped_polygon.geometry],
                out_shape=out_shape,
                transform=transform,
                #all_touched=True, # Select this option if you want thicker masks
                fill=0,
                dtype='uint8'
            )
            
            # Rescale [0, 1] to [0, 255]
            scaled_data = (rasterized * 255).astype(np.uint8)

            # Use the existing raster name and append '_gt' to it
            # IMPORTANT: The TIFFs and PNGs need to be ordered and match with each other
            # If they dont, metrics will not be reliable
            filename = os.path.splitext(os.path.basename(tif_file))[0]
            output_file = os.path.join(data_folder, f'{filename}_ref.png')

            image = Image.fromarray(scaled_data)
            image.save(output_file)

            print(f"Wrote annotated data: '{output_file}' to file")


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

def get_atmospheric_level(sceneid, band_list):
    """
    Determines the atmospheric correction level based on the number of bands in a .tif file.

    Args:
        sceneid (str): The scene ID corresponding to the dataset folder.
        band_list (list): Expected list of bands for validation.

    Returns:
        str: Atmospheric correction level ('L1', 'L2A', 'L1R'), or None if bands are insufficient.
    """
    data_folder = os.path.join("data", sceneid)
    
    # Find all .tif files in the folder, read in the first file
    tif_files = sorted(glob.glob(os.path.join(data_folder, '*.tif')))
    
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {data_folder}")
    
    # Open the .tif file and check the number of bands
    with rasterio.open(tif_files[0]) as src:
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