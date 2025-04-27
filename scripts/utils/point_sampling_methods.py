import os
import rasterio
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.ndimage import label as nd_label # Prevent naming conflicts with variable 'label'
from scipy.spatial import distance
from skimage.morphology import skeletonize

# Custom imports
from scripts.utils.feature_scaling_functions import percentile_rescale

torch.manual_seed(42), random.seed(42)

def sample_random_prompts(label_np, num_prompts, prompt_value):
    """
    Generate negative (background) prompts from a binary label image.

    Parameters:
    - label_np: numpy.ndarray, binary array with values 0 for background and 255 (==1) for foreground.
    - num_positive_prompts: int, the number of positive prompts to match with the same number of negative prompts.

    Returns:
    - negative_prompts: list of tuples, coordinates of negative prompts as (x, y).
    """
    np.random.seed(42)

    # Find all background pixels
    background_px = np.argwhere(label_np == prompt_value)

    # Randomly sample the background pixels to match the number of positive prompts
    sampled_negative_prompts = background_px[np.random.choice(len(background_px), num_prompts, replace=False)]

    # Convert to a list of (x, y) tuples
    negative_prompts = [(int(y), int(x)) for y, x in sampled_negative_prompts]

    return np.array(negative_prompts)

def extract_manual_prompts(shapefile_path, raster_path, label, prompt_type="positive"):
    """
    Extracts manual prompts (positive, negative, or both) for a given raster and shapefile input.

    Args:
        shapefile_path (str): Path to the shapefile containing annotation points.
        raster_path (str): Path to the raster file.
        label_tensor (torch.Tensor): Tensor containing labeled raster data.
        prompt_type (str): Type of prompts to extract - "positive", "negative", or "both". Defaults to "positive".

    Returns:
        torch.Tensor: Tensor of padded coordinate points for the prompts.
        torch.Tensor: Tensor of padded labels for the prompts.
    """
    # Read raster metadata
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs             # CRS for reprojection
        raster_bbox = box(*(src.bounds)) # Bounding box for clipping
        raster_transform = src.transform # Affine transformation for reprojection into cartesian coordinates

    # Load and clip the annotations to the raster extent
    scene_prompts = gpd.read_file(shapefile_path[0]).to_crs(raster_crs)
    point_prompts = gpd.clip(scene_prompts, raster_bbox)

    # If the 'type' column is missing, assume all points are positive
    if 'type' not in point_prompts.columns:
        #print("'type' column is missing from the point prompts. Assuming all points are positive prompts")
        point_prompts['type'] = 1
    
    # Extract positive and negative prompts based on the 'type' column
    positive_prompts = point_prompts[point_prompts['type'] == 1]
    label_array = label.squeeze().numpy()
    
    # Sample negative prompts to match the number of positive prompts
    num_positive_prompts = len(positive_prompts)
    negative_prompts = sample_random_prompts(label_array, num_positive_prompts, prompt_value=0)

    # Transform CRS into cartesian coordinates => output in rows, cols => y, x
    positive_coords_rowcol = rasterio.transform.rowcol(
        raster_transform,
        positive_prompts['geometry'].x,
        positive_prompts['geometry'].y
    )
    positive_prompts_xy = np.flip(np.array(positive_coords_rowcol).T, axis=1)  # Convert (row, col) to (x, y)

    # Filter points based on prompt type
    if prompt_type == "positive":
        priors = positive_prompts_xy
        prior_labels = np.ones(len(positive_prompts_xy))

    elif prompt_type == "negative":
        priors = negative_prompts
        prior_labels = np.zeros(len(negative_prompts))

    else:  # "both"
        priors = np.concatenate((positive_prompts_xy, negative_prompts), axis=0)
        prior_labels = np.concatenate((np.ones(len(positive_prompts_xy)), np.zeros(len(negative_prompts))))

    # Pad both arrays to be of equal size between patches
    padded_priors = np.pad(priors, ((0, 50 - priors.shape[0]), (0, 0)), mode='constant', constant_values=0)
    padded_labels = np.pad(prior_labels, (0, padded_priors.shape[0] - len(prior_labels)), mode='constant', constant_values=99)

    return torch.tensor(padded_priors, dtype=torch.uint8), torch.tensor(padded_labels, dtype=torch.uint8)

def prompts_from_spectralclusters(image, label, num_clusters=10, prompt_type="both", plotting=False):
    """
    Cluster annotated pixels and return the coordinates of pixels closest to cluster centroids.
    
    Parameters:
    - image (np.array): The input image array.
    - label: torch.Tensor, binary tensor of shape [1, height, width] with values 0 or 255.
    - num_clusters (int): The number of clusters for K-means. Default is 10.
    
    Returns:
    - List of tuples with coordinates of pixels closest to cluster centroids.
    """
    
    label = label.squeeze().numpy()
    debris_coords = np.argwhere(label == 255) # XY locations within annotations

    # Extract values of pixels within annotations
    spectral_values = image[:, label == 255].T # Transpose from [n_features, n_samples] into [n_samples, n_features]
       
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(spectral_values)
    
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    # Find the closest pixel to each cluster center in the RGB space
    closest_pixels = []
    for center in cluster_centers:
        # Calculate the distance from each annotated pixel's RGB value to the cluster center
        distances = np.linalg.norm(spectral_values - center, axis=1)
        closest_index = np.argmin(distances)
        closest_pixels.append(tuple(debris_coords[closest_index]))
    
    closest_pixels = [(point[1], point[0]) for point in closest_pixels] # YX to XY
    
    # Sample negative prompts equal to number of clusters
    negative_prompts = sample_random_prompts(label, num_clusters, prompt_value=0)

    if prompt_type == "positive":
        priors = closest_pixels
        prior_labels = np.ones(len(closest_pixels))
    elif prompt_type == "negative":
        priors = negative_prompts
        prior_labels = np.zeros(len(negative_prompts))
    else:  # "both"
        priors = np.concatenate((closest_pixels, negative_prompts), axis=0)
        prior_labels = np.concatenate((np.ones(len(closest_pixels)), np.zeros(len(negative_prompts))))

    # Convert to NumPy array for padding
    priors = np.array(priors)

    # Pad arrays so sizes remain equal between different patches
    priors_padded = np.pad(priors, ((0, 50 - priors.shape[0]), (0, 0)), mode='constant', constant_values=0)
    prior_labels_padded = np.pad(prior_labels, (0, priors_padded.shape[0] - len(prior_labels)), mode='constant', constant_values=99)

    if plotting: # Default set to False to speed up computation
        # Generate colors for each cluster
        colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
        
        # Visualize the results
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Create the pairwise RGB plots
        color_pairs = [('Red', 'Green'), ('Green', 'Blue'), ('Blue', 'Red')]
        color_indices = [(0, 1), (1, 2), (2, 0)]

        for i, ((x_color, y_color), (x_idx, y_idx)) in enumerate(zip(color_pairs, color_indices)):
            scatter = axes[i].scatter(spectral_values[:, x_idx], spectral_values[:, y_idx], c=cluster_labels, cmap='tab10', s=10)
            axes[i].set_title(f'{x_color} vs. {y_color}')
            axes[i].set_xlim(0, 255)  # Set limits for RGB values
            axes[i].set_ylim(0, 255)  # Set limits for RGB values
            axes[i].grid(True)  # Add grid lines to the RGB plots

        # Overlay the cluster centers on the original image
        axes[3].imshow(image)
        for cluster_index, (x, y) in enumerate(closest_pixels):
            axes[3].plot(x, y, 'o', color=colors[cluster_index], markersize=10)

        axes[3].set_title('Image with K-means prompts')
        #axes[3].axis('off')  # Turn off the axis for a cleaner look

        # Add a color bar for the scatter plot
        cbar = fig.colorbar(scatter, ax=axes[:3], orientation='horizontal', fraction=0.02, pad=0.1)
        cbar.set_label('Cluster Labels')  # Optional label for color bar

        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        os.makedirs('doc/figures', exist_ok=True)
        fig.savefig('doc/figures/kmeans.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        print("K-means figure printed")
    
    return torch.tensor(priors_padded), torch.tensor(prior_labels_padded, dtype=torch.uint8)