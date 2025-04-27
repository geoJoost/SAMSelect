import os

# Custom imports
from scripts.models.sam_predictor import execute_SAM
from scripts.models.helper_functions import select_top_bands, get_band_info, get_atmospheric_level
from scripts.utils.get_band_idx import get_band_idx
from scripts.utils.process_band_columns import process_band_columns

def execute_ndvi(tif_path, polygon_path, band_list, scaling, equation='ndi', model_type='vit_b'):
    # Define NDVI bands
    band_combination = ["B8", "B4"]
    
    # Get atmospheric correction level (L1, L2A or L2R)
    atm_level = get_atmospheric_level(tif_path, band_list)

    # Get index positions corresponding to the bands; e.g., [B1, B2, B3] => [1, 2, 3]
    bands_idx = get_band_idx(band_list, band_combination, equation)

    # Execute SAM and obtain a dataframe of Jaccard scores
    df_ndvi, _ = execute_SAM(tif_path, polygon_path, bands_idx, scaling, equation, model_type, atm_level)

    # Check number of bands used, depending on the equation
    band_columns = ['band_1', 'band_2']

    # Process band columns from index positions into band names
    df_ndvi = process_band_columns(df_ndvi, band_columns, band_list, equation)

    # Compute the mean statistic for both mask levels
    sceneid = os.path.splitext(os.path.basename(tif_path))[0]
    
    # Rename the columns for pretty-print
    df_print = df_ndvi.rename(columns={
        'jaccard_lvl1': 'Level-1: IoU (%)',
        'jaccard_lvl2': 'Level-2: IoU (%)',
        'jaccard_lvl3': 'Level-3: IoU (%)'
    })
    print(f"\nNDVI results for {sceneid}:\n{(df_print[['Level-1: IoU (%)', 'Level-2: IoU (%)', 'Level-3: IoU (%)']] * 100).round(2)}\n\n")

def execute_pca(tif_path, polygon_path, band_list, scaling, equation='pca', model_type='vit_b'):
    # Get atmospheric correction level (L1, L2A or L2R)
    atm_level = get_atmospheric_level(tif_path, band_list)
    
    # Execute SAM and obtain a dataframe of Jaccard scores
    df_pca, _ = execute_SAM(tif_path, polygon_path, band_list, scaling, equation, model_type, atm_level)

    # Compute the mean statistic for both mask levels
    sceneid = os.path.splitext(os.path.basename(tif_path))[0]
    
    # Rename the columns for pretty-print
    df_print = df_pca.rename(columns={
        'jaccard_lvl1': 'Level-1: IoU (%)',
        'jaccard_lvl2': 'Level-2: IoU (%)',
        'jaccard_lvl3': 'Level-3: IoU (%)'
    })
    print(f"\nPCA results for {sceneid}:\n{(df_print[['Level-1: IoU (%)', 'Level-2: IoU (%)', 'Level-3: IoU (%)']] * 100).round(2)}\n\n")

def execute_fdi(tif_path, polygon_path, band_list, scaling, equation='fdi', model_type='vit_b'):
    # Get atmospheric correction level (L1, L2A or L2R)
    atm_level = get_atmospheric_level(tif_path, band_list)
    
    # Define FDI bands
    band_combination = ["B8", "B6", "B11"]

    # Get index positions corresponding to the bands; e.g., [B1, B2, B3] => [1, 2, 3]
    bands_idx = get_band_idx(band_list, band_combination, equation)

    # Execute SAM and obtain a dataframe of Jaccard scores
    df_fdi, img_fdi = execute_SAM(tif_path, polygon_path, bands_idx, scaling, equation, model_type, atm_level)

    # Check number of bands used, depending on the equation
    band_columns = ['band_1', 'band_2', 'band_3']

    # Process band columns from index positions into band names
    df_fdi = process_band_columns(df_fdi, band_columns, band_list, equation)

    # Compute the mean statistic for both mask levels
    sceneid = os.path.splitext(os.path.basename(tif_path))[0]
    df_print = df_fdi.rename(columns={
        'jaccard_lvl1': 'Level-1: IoU (%)',
        'jaccard_lvl2': 'Level-2: IoU (%)',
        'jaccard_lvl3': 'Level-3: IoU (%)'
    })
    print(f"\nFDI results for {sceneid}:\n{(df_print[['Level-1: IoU (%)', 'Level-2: IoU (%)', 'Level-3: IoU (%)']] * 100).round(2)}\n\n")