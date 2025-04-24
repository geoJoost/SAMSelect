# Custom modules
from scripts.models.spectral_indices import execute_ndvi, execute_fdi, execute_pca
from scripts.models.samselect import samselect
from scripts.visualization.viz_patches import plot_patches
from scripts.visualization.viz_spectraldata import get_spectral_statistics

def samselect_wrapper(tif_path, polygon_path, band_list, narrow_search_bands=None, scaling='percentile_1-99', equation_list=['bc', 'ndi', 'ssi', 'top'], model_type='vit_b', atm_level='L2A'):   
    """ SAMSelect 
    Notes for users:
    - SceneID should refer to the multispectral scene and data-folder in data/sceneID/, and needs to contain the following:
        1. TIFF patches of the sensor in square patches. We used 128px x 128px
        2. Labels can be EITHER 2.1. Binary PNG files having 100% overlap with the corresponding TIFF files
                             OR 2.2. Polygon annotations of the objects (containing '*qualitative*.shp' somewhere in the filepath) which can be rasterized within SAMSelect
        3. Prompts can be EITHER 3.1. Point prompts manually annotated by the user (containing *_pt*.shp somewhere in the filepath) AND having a 'type' column with binary (0, 1) values
                              OR 3.2 if none are given, points are generated from the PNG files using K-means as default (other options are available)

    - Band_list should refer to ALL available multispectral bands in the TIFF files. For Sentinel-2 this would look like: ["B1", "B2", "B3"] etc
    - Narrow_search_bands is an optional argument, but can be used to narrow down the spectral search space. The given bands NEED to match with those in band_list
    - Scaling refers to the normalization function. Default is percentile scaling (1% - 99%)
    - Equation refers to the visualization module with the options:
        - BC --> Band composites being false colour composites using three spectral bands
        - NDI --> Normalized Difference Indices such as NDVI, NDWI
        - SSI --> Spectral Shape Indices such as FAI
        - SIC --> Spectral Index Composite, being the top-10 most informative NDI and SSIs (requires the previous two have been computed). Uses spectral indices instead of spectral bands
    - Model_type refers to the image encoder of SAM (vit_b, vit_l, vit_h)
    - Sensor_type refers to which Sentinel-2 satellite is used (S2A, S2B). Only used for SSI and FDI calculations. Support for other satellites not integrated yet

    NOTE: Computation can take up to 1-2 hours, depending on GPU / CPU
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

    """ Domain indices (marine debris)
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

# Define Sentinel-2 spectral bands
#l1_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"] # L1C / L1R
#l2a_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]       # L2A
#l2r_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]             # L2R

# Execute the SAMSelect wrapper
#samselect_wrapper(tif_path='data/durban_20190424_l2a.tif' , 
#                    polygon_path= "data/durban_20190424_qualitative_poly.shp",
#                    band_list= l2a_bands, #=> Sentinel-2 L2A bands 
#                    narrow_search_bands= ['B1', 'B2','B3', 'B4'], #None, #=> Manual selection of bands like: ['B3', 'B4', 'B8', 'B8A']. Naming convention needs to match 'band_list' variable
#                    scaling= 'percentile_1-99', #=> Normalization function. See dataloader.py 
#                    equation_list= ['bc'],#['bc', 'ndi', 'ssi', 'top'], #=> Visualization modules. Current: Band Composites (BC), Normalized Difference Index (NDI), Spectral Shape Index, and RSI-top10 ('top' in code) 
#                    model_type= 'vit_b') #=> SAM encoder