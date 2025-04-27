import pandas as pd
from itertools import combinations
import os

# Custom modules
from scripts.models.sam_predictor import execute_SAM
from scripts.models.helper_functions import select_top_bands, get_atmospheric_level
from scripts.utils.get_band_idx import get_band_idx
from scripts.utils.process_band_columns import process_band_columns

def samselect(tif_path, polygon_path, band_list, narrow_search_bands=None, scaling='percentile_1-99', equation='bc', model_type='vit_b', atm_level='L2A'):
    """
    Executes the SAMSelect algorithm to evaluate the Segment Anything Model (SAM) on a Sentinel-2 scene for objects-of-interest detection.

    This function processes a Sentinel-2 scene and corresponding polygon annotations, iterates over various band combinations,
    and applies the SAM model to visualize and evaluate the objects-of-interest. It supports different visualization modules and scaling methods to
    preprocess the image data before feeding it to the SAM model. The function evaluates the model's performance using the
    Jaccard index (IoU) for different mask levels and saves the results to a CSV file.

    Parameters:
    - tif_path (str): Path to the Sentinel-2 scene (TIFF format).
    - polygon_path (str): Path to the polygon annotations for objects-of-interest (shapefile format).
    - band_list (list of int): List of Sentinel-2 bands to be used. Default is [4, 3, 2], which corresponds to the RGB bands.
    - narrow_search_bands (list of int, optional): List of bands to narrow the spectral search space. Default is None.
    - scaling (str, optional): Scaling method to normalize the image data. Default is 'percentile_1-99'. Options include:
        - 'none': No scaling.
        - 'min-max': Min-Max normalization.
        - 'equalize': Histogram equalization.
        - 'percentile_1-99': Percentile scaling (1% - 99%).
        - 'percentile_2-98': Percentile scaling (2% - 98%).
        - 'percentile_5-95': Percentile scaling (5% - 95%).
        - 'adaptive_equalize_01': Adaptive histogram equalization (clip limit 0.01).
        - 'adaptive_equalize_03': Adaptive histogram equalization (clip limit 0.03).
        - 'adaptive_equalize_05': Adaptive histogram equalization (clip limit 0.05).
        
    - equation (str, optional): Option for the visualization module. Default is 'bc' (Band Composite). Other options include:
        - 'bc': Band Composite
        - 'ndi': Normalized Difference Index
        - 'ssi': Spectral Shape Index
        - 'top': Spectral Index Composite
        - 'pca': Principal Component Analysis
        - 'fdi': Floating Debris Index
        
    - model_type (str, optional): SAM encoder type. Default is 'vit_b' (ViT-Base). Other options include:
        - 'vit_h': ViT-Huge
        - 'vit_l': ViT-Large
        - 'vit_b': ViT-Base
        
    - atm_level (str, optional): Indicates the Sentinel-2 product level. Default is 'L2A'. Supported levels include:
        - 'L1C': Top-of-atmosphere reflectance (Sen2Cor)
        - 'L2A': Bottom-of-atmosphere reflectance (Sen2Cor)
        - 'L1R': Rayleigh-corrected reflectance (ACOLITE)
        - 'L2R': Rayleigh and aerosol-corrected reflectance (ACOLITE)

    Returns:
    - None

    The function performs the following steps:
    1. Checks if the results CSV file already exists and returns early if it does.
    2. Defines the number of bands to utilize based on the selected visualization module.
    3. Defines the spectral search space based on the visualization module and user input.
    4. Retrieves the atmospheric correction level of the Sentinel-2 scene.
    5. Iterates over each possible band combination and executes the SAM model.
    6. Processes the band columns and saves the final results to a CSV file.
    """
    sceneid = os.path.splitext(os.path.basename(tif_path))[0]
    # Define the output file path
    output_csv = f"data/processed/{sceneid}_{equation}_{model_type}_results.csv"
    
    # Check if the CSV file already exists, if yes, then return early
    if os.path.exists(output_csv):
        print(f"Results already exist at '{output_csv}'. Skipping execution.")
        return

    print(f"################################\n\nEvaluating on {sceneid} using equation: {equation}\n\n################################\n\n")
    df_results = pd.DataFrame()
    
    # Define number of bands to utilize in the calculations
    equation_mapping = {'bc': 3, 'ssi': 3, 'ndi': 2, 'top': 3}
    assert equation in equation_mapping, "Invalid equation provided."
    num_bands = equation_mapping[equation]

    # Define spectral search space
    if equation == 'top': # Spectral Index Composite method
        # Since this method uses spectral indices instead of spectral bands, it uses pre-computed information from the NDI and SSIs
        top_combinations = select_top_bands(sceneid, model_type, ['ndi', 'ssi'], 10) # Select Top-10 best performing indices
        band_searchspace =  list(combinations(top_combinations, num_bands))
    else:
        # Spectral search space can be exhaustive (i.e., uses all possible spectral bands; default option)
        # Or narrow, depending on the user input
        available_bands = narrow_search_bands if narrow_search_bands else band_list
        band_searchspace =  list(combinations(available_bands, num_bands))    

    # Get atmospheric correction level (L1, L2A or L2R)
    atm_level = get_atmospheric_level(tif_path, band_list)
    
    # Check for SSI visualization mode with insufficient bands
    if equation == 'ssi' and atm_level is None:
        print(F"Skipping execution of SSI: Insufficient number of bands for 'get_band_info()' to retrieve central wavelengths")
        return None
    
    # For-loop to iterate over each possible combination from the bands
    for band_combination in band_searchspace:
        print(f'...\nEvaluating on bands: {band_combination}')

        # Get index positions corresponding to the bands; e.g., [B1, B2, B3] => [1, 2, 3]
        bands_idx = get_band_idx(band_list, band_combination, equation)

        # Execute SAM and obtain a dataframe of IoU scores
        df_interim, _ = execute_SAM(tif_path, polygon_path, bands_idx, scaling, equation, model_type, atm_level)

        # Append the current DataFrame to the combined DataFrame
        df_results = pd.concat([df_results, df_interim], ignore_index=True)

    # Check number of bands used, depending on the equation
    band_columns = ['band_1', 'band_2']
    if 'band_3' in df_results.columns:
        band_columns.append('band_3')
    
    # Process band columns from index positions into band names
    df_results = process_band_columns(df_results, band_columns, band_list, equation)

    # Save final results to .csv
    os.makedirs("data/processed", exist_ok=True)   
    df_results.to_csv(output_csv, index=False)

    print(f"SAMSelect finished running, file saved as {output_csv}")