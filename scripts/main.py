# Custom modules
from scripts.models.spectral_indices import execute_ndvi, execute_fdi, execute_pca
from scripts.models.samselect import samselect
from scripts.visualization.viz_patches import plot_patches
from scripts.visualization.viz_spectraldata import get_spectral_statistics

def samselect_wrapper(tif_path, polygon_path, band_list, narrow_search_bands=None, scaling='percentile_1-99', equation_list=['bc', 'ndi', 'ssi', 'top'], model_type='vit_b', atm_level='L2A'):   
    """ SAMSelect documentation: see README.md or scripts/models/samselect.py
    NOTE: Computation can take up to 1-8 hours, depending on GPU / CPU, and number of patches
    NOTE: If the SAMSelect output exists, the runtime will be skipped and immediately go into tables & graphs
    """
    for equation in equation_list: # Iterate over the chosen visualization options
        samselect(tif_path, polygon_path, band_list, narrow_search_bands, scaling, equation, model_type)
    
    """ Visualization & Graphs
    The first script prints the statistics for hte top-5 best scoring visualization for each visualization module available.
    It also prints a graph showcasing the frequency of each spectral band occuring in 'doc/figures/'

    NOTE: Spectral shading is made for Sentinel-2 L2A bands, and will not be correct when using different sensors/bands. See variable 'spectral_shading'

    The second script plots all available TIFF files in: 
        1. True-colour (VIS)
        2. NDVI
        3. FDI
        4. NDI using B2 and B8
        5. The top-1 result from SAMSelect
        6. Patch label
        7. The top-1 visualization with SAM-predictions

    NOTE: Similar to the comparison scripts, band selection for NDVI and FDI are hard-coded    
    """
    top1_combination, top1_equation, top1_masklevel = get_spectral_statistics(tif_path, polygon_path, band_list, equation_list, model_type, spectral_shading=False)

    plot_patches(tif_path, polygon_path, band_list, top1_combination, top1_equation, top1_masklevel)

    """ Domain indices for marine debris
    Notes for users:
    - The indices are hard-coded into their respective bands, such as NDVI using 'B8' and 'B4'
    - If using a different sensor (e.g., Landsat or PlanetScope data), modify this code or comment it out
    """
    # NDVI [B8, B4]
    execute_ndvi(tif_path, polygon_path, band_list, scaling, equation='ndi', model_type=model_type)
    
    # Floating Debris Index (FDI) [B8, B6, B11 + B4 (central wavelength value)]
    execute_fdi(tif_path, polygon_path, band_list, scaling, equation='fdi', model_type=model_type)
    
    # Principal Component Analysis (PCA) [All available bands]
    execute_pca(tif_path, polygon_path, band_list, scaling, equation='pca', model_type=model_type)

# TODO: Remove the function calls. Kept for documentation
# Define Sentinel-2 spectral bands
#l1_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"] # L1C / L1R
#l2a_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]       # L2A
#l2r_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]             # L2R

# Execute the SAMSelect wrapper
#samselect_wrapper(tif_path='data/demo_durban_20190424.tif',#data/durban_20190424_l2a.tif' , 
#                    polygon_path= "data/demo_durban_20190424.shp", #"data/durban_20190424_qualitative_poly.shp",
#                    band_list= l2a_bands, #=> Sentinel-2 L2A bands 
#                    narrow_search_bands= ['B1', 'B2','B3', 'B4'], #None, #=> Manual selection of bands like: ['B3', 'B4', 'B8', 'B8A']. Naming convention needs to match 'band_list' variable
#                    scaling= 'percentile_1-99', #=> Normalization function. See dataloader.py 
#                    equation_list= ['bc'],#['bc', 'ndi', 'ssi', 'top'], #=> Visualization modules. Current: Band Composites (BC), Normalized Difference Index (NDI), Spectral Shape Index, and RSI-top10 ('top' in code) 
#                    model_type= 'vit_b') #=> SAM encoder