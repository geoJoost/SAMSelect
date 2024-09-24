import os
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_s2patches(data_root, s2scene, point_id, window_size=1280):
    raster_path = os.path.join(data_root, f"{s2scene}.tif") # Example: 'data/raw/accra_20181031_l1c.tif'
    point_path = os.path.join(data_root, f"{s2scene.split('_l')[0]}.shp")  # Example: 'data/raw/accra_20181031.shp'
    out_path = os.path.join("data/interim", s2scene)

    # Choose correct raster
    raster = rasterio.open(raster_path)

    # Load point data
    points = gpd.read_file(point_path).to_crs(raster.crs)

    # Create folder for output patches
    os.makedirs(out_path, exist_ok=True)

    # Select specific points from the list
    selected_points = points[points.index.isin(point_id)]
    
    for index, point in selected_points.iterrows():
        # Extract coordinates of the point
        x, y = point['geometry'].x, point['geometry'].y

        # Define a window around the point based on window size (current = 1280)
        window = from_bounds(x - window_size / 2, y - window_size / 2, x + window_size / 2, y + window_size / 2,
                             raster.transform)

        # Read data within the specified window
        windowed_data = raster.read(window=window)

        # Transform DN numbers into reflectance values between 0-1. Only for Sen2Cor data
        if s2scene.split('_')[-1] in ['l1c', 'l2a']:   
            windowed_data = (windowed_data * 1e-4).astype(float)

        # Retrieve metadata and update for the cropped patch
        out_meta = raster.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": windowed_data.shape[1],
            "width": windowed_data.shape[2],
            "transform": raster.window_transform(window),
            "crs": raster.crs,
            "dtype": rasterio.float32
        })

        # Save Sentinel-2 patches with all bands
        output_file = os.path.join(out_path, f"patch_s2_{index}.tif")
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(windowed_data)

        print(f"Wrote Sentinel-2 data: '{output_file}' to file")


def create_labelpatches(data_root, s2scene, point_id, window_size=1280):
    label_path = os.path.join(data_root, f"{s2scene.split('_l')[0]}_qualitative_poly.tif") # Example: 'data/floatingobjects/accra_20181031_qualitative_poly.tif'
    point_path = os.path.join(data_root, f"{s2scene.split('_l')[0]}.shp") # Example: 'data/floatingobjects/accra_20181031.shp'

    # Create folder for output patches
    out_path = os.path.join("data/interim", s2scene)
    os.makedirs(out_path, exist_ok=True)

    # Choose correct raster
    raster = rasterio.open(label_path)

    # Load point data
    points = gpd.read_file(point_path).to_crs(raster.crs)

    # Select specific points from the list
    selected_points = points[points.index.isin(point_id)]

    for index, point in selected_points.iterrows():
        # Extract coordinates of the point
        x, y = point['geometry'].x, point['geometry'].y

        # Define a window around the point based on window size (current = 1280)
        window = from_bounds(x - window_size / 2, y - window_size / 2, x + window_size / 2, y + window_size / 2,
                             raster.transform)

        # Read data within the specified window
        windowed_data = raster.read(window=window)
        
        # Since data is in [0, 1], it needs to be rescaled to [0, 255]
        scaled_data = (windowed_data * 255).astype(np.uint8)

        # Transpose from (128, 128, 1) into (1, 128, 128)
        transposed_data = scaled_data.transpose(1, 2, 0)

        # Drop the last channel to get (128, 128) as .fromarray() requires a 2-tuple containing width x height
        transposed_data = transposed_data[:, :, 0]

        # Save labelled patches
        output_file = os.path.join(out_path, f"patch_gt_{index}.png")

        # Create an image from the array
        image = Image.fromarray(transposed_data)
        image.save(output_file)

        print(f"Wrote annotated data: '{output_file}' to file")

# Define static variables
data_root = r"data/raw/floatingobjects"

#s2_accra = "accra_20181031" # ID number of the Sentinel-2 scene
point_id_accra = [389, 394,       # Bbox-1: Floating plastic from waste dump on beach
                  507, 522, 555]  # Bbox-2: Floating Sargassum patches

point_id_durban = [37, 32, 14,        # Bbox-1
                   56, 2,         # Bbox-2
                   147, 144, 141, # Bbox-3
                   94, 104, 112,      # Bbox-4
                   121, 133, 161, 72, 154 # Misc patches
                   ]       


# Accra patches
#s2_accra = ["accra_20181031_l1c", "accra_20181031_l1r", "accra_20181031_l2a", "accra_20181031_l2r"]

#for scene in s2_accra:
#    create_s2patches(data_root, scene, point_id_accra)    # Sentinel-2 scene into .tif
#    create_labelpatches(data_root, scene, point_id_accra) # Labeled dataset into .png

# Durban patches
s2_durban = ["durban_20190424_l1c", "durban_20190424_l1r", "durban_20190424_l2a", "durban_20190424_l2r"]
#for scene in s2_durban:
#    create_s2patches(data_root, scene, point_id_durban)  
#    create_labelpatches(data_root, scene, point_id_durban)

create_landmask_patches(data_root, 'durban_20190424_l2a', point_id_durban, window_size=1280)
create_landmask_patches(data_root, 'accra_20181031_l2a', point_id_accra, window_size=1280)



