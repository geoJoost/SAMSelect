import pandas as pd
from itertools import combinations
import os

# Custom modules
from models.sam_predictor import execute_SAM
from models.helper_functions import select_top_bands
from utils.get_band_idx import get_band_idx
from utils.process_band_columns import process_band_columns

def samselect(sceneid, band_list, narrow_search_bands=None, scaling='percentile_1-99', equation='bc', model_type='vit_b', sensor_type='S2B'):
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
    if equation == 'top': # RSI-top10 method
        # Since this method uses spectral indices instead of spectral bands, it uses pre-computed information from the NDI and SSIs
        top_combinations = select_top_bands(sceneid, model_type, ['ndi', 'ssi'], 10) # Select Top-10 best performing indices
        band_searchspace =  list(combinations(top_combinations, num_bands))
    else:
        # Spectral search space can be exhaustive (i.e., uses all possible spectral bands; default option)
        # Or narrow, depending on the user input
        available_bands = narrow_search_bands if narrow_search_bands else band_list
        band_searchspace =  list(combinations(available_bands, num_bands))
    
    # For-loop to iterate over each possible combination from the bands
    for band_combination in band_searchspace:
        print(f'...\nEvaluating on bands: {band_combination}')

        # Get index positions corresponding to the bands; e.g., [B1, B2, B3] => [1, 2, 3]
        bands_idx = get_band_idx(band_list, band_combination, equation)

        # Execute SAM and obtain a dataframe of mIoU scores
        df_interim, _ = execute_SAM(sceneid, bands_idx, scaling, equation, model_type, sensor_type)

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