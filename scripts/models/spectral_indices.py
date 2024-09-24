from models.sam_predictor import execute_SAM
from models.helper_functions import select_top_bands, get_band_info
from utils.get_band_idx import get_band_idx
from utils.process_band_columns import process_band_columns

def execute_ndvi(sceneid, band_list, scaling, equation='ndi', model_type='vit_b', sensor_type='S2B'):
    # Define NDVI bands
    band_combination = ["B8", "B4"]

    # Get index positions corresponding to the bands; e.g., [B1, B2, B3] => [1, 2, 3]
    bands_idx = get_band_idx(band_list, band_combination, equation)

    # Execute SAM and obtain a dataframe of Jaccard scores
    df_ndvi, _ = execute_SAM(sceneid, bands_idx, scaling, equation, model_type, sensor_type)

    # Check number of bands used, depending on the equation
    band_columns = ['band_1', 'band_2']

    # Process band columns from index positions into band names
    df_ndvi = process_band_columns(df_ndvi, band_columns, band_list, equation)

    # Compute the mean statistic for both mask levels
    print(f"\nNDVI results for {sceneid}:\n{df_ndvi.round(3)}\n\n")

def execute_pca(sceneid, _, scaling, equation='pca', model_type='vit_b', sensor_type='S2B'):
    # Execute SAM and obtain a dataframe of Jaccard scores
    df_pca, _ = execute_SAM(sceneid, _, scaling, equation, model_type, sensor_type)

    # Compute the mean statistic for both mask levels
    print(f"\nPCA results for {sceneid}:\n{df_pca.round(3)}\n\n")

def execute_fdi(sceneid, band_list, scaling, equation='fdi', model_type='vit_b', sensor_type='S2B'):
    # Define FDI bands
    band_combination = ["B8", "B6", "B11"]

    # Get index positions corresponding to the bands; e.g., [B1, B2, B3] => [1, 2, 3]
    bands_idx = get_band_idx(band_list, band_combination, equation)

    # Execute SAM and obtain a dataframe of Jaccard scores
    df_fdi, img_fdi = execute_SAM(sceneid, bands_idx, scaling, equation, model_type, sensor_type)

    # Check number of bands used, depending on the equation
    band_columns = ['band_1', 'band_2', 'band_3']

    # Process band columns from index positions into band names
    df_fdi = process_band_columns(df_fdi, band_columns, band_list, equation)

    # Compute the mean statistic for both mask levels
    print(f"\nFDI results for {sceneid}:\n{df_fdi.round(3)}\n\n")