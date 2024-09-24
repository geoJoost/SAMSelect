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

from utils.feature_scaling_functions import percentile_rescale

torch.manual_seed(42), random.seed(42)


### Segment Anything ##

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

def extract_marinedebris_points(shapefile_path, input_path, label, prompt_type="positive"):
    with rasterio.open(input_path) as src:
        raster_crs = src.crs             # CRS for reprojection
        raster_bbox = box(*(src.bounds)) # Bounding box for clipping
        raster_transform = src.transform # Affine transformation for reprojection into cartesian coordinates

    # Read and clip points to patches extent
    floatingobjects_pt = gpd.read_file(shapefile_path[0]).to_crs(raster_crs)
    md_points = gpd.clip(floatingobjects_pt, raster_bbox)

    # Check if the 'type' column exists. This is used for generating positive labels and filtering negative points (type == 0)
    assert 'type' in md_points.columns, "'type' column is missing from the DataFrame. This is a requirement for filtering positive/negative prompts"

    # For negative point sampling
    label_np = label.squeeze().numpy()
    num_positive_prompts = md_points[md_points['type'] == 1].count().iloc[0]
    negative_prompts = sample_random_prompts(label_np, num_positive_prompts, prompt_value=0)

    # Positive point sampling
    positive_prompts = md_points.loc[md_points['type'] == 1] # Marine Debris annotations

    # Transform CRS into cartesian coordinates => output in rows, cols => y, x
    positive_prompts_yx  = rasterio.transform.rowcol(raster_transform, positive_prompts['geometry'].x, positive_prompts['geometry'].y)
    positive_prompts_xy = np.flip(np.array(positive_prompts_yx).T, axis=1) # YX => XY

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
    priors_padded = np.pad(priors, ((0, 50 - priors.shape[0]), (0, 0)), mode='constant', constant_values=0)
    prior_labels_padded = np.pad(prior_labels, (0, priors_padded.shape[0] - len(prior_labels)), mode='constant', constant_values=99)

    return torch.tensor(priors_padded, dtype=torch.uint8), torch.tensor(prior_labels_padded, dtype=torch.uint8)

def extract_all_prompts(label, min_pixel_count=10, prompt_type="positive"):
    """
    Extract randomly sampled points from masks larger than a specified number of pixels.
    
    Parameters:
    - label: torch.Tensor, binary tensor of shape [1, height, width] with values 0 or 255.
    - min_pixel_count: int, minimum number of pixels for a mask to be considered.
    - num_samples: int, number of points to sample.
    - prompt_type: str, type of prompts to return ("positive", "negative", "both").
    
    Returns:
    - points: torch.Tensor, padded list of (x, y) tuples representing the sampled points.
    - point_labels: torch.Tensor, corresponding labels (1 for mask, 0 for background, 99 for padding).
    """
    np.random.seed(42)

    # Convert the binary image to 0 and 1 format for skeletonize
    binary_image = (label.squeeze().numpy() == 255).astype(np.uint8)

    # Find connected components (i.e., masks) in the binary image
    labeled_array, num_masks = nd_label(binary_image)

    # Create a new array to store only large components
    large_component_mask = np.zeros_like(labeled_array)

    # Filter out masks based on size (<10px)
    for mask_idx in range(1, num_masks + 1):
        mask_points = np.argwhere(labeled_array == mask_idx)

        # Filter out masks with fewer than min_pixel_count pixels
        if len(mask_points) < min_pixel_count:
            continue
        large_component_mask[labeled_array == mask_idx] = 1

    # Identify all positive points instead of random sampling
    positive_prompts = np.argwhere(large_component_mask == 1)

    # Randomly sample negative point prompts identical to the number of positive prompts
    negative_prompts = sample_random_prompts(large_component_mask, len(positive_prompts), prompt_value=0)

    if prompt_type == "positive":
        priors = positive_prompts
        prior_labels = np.ones(len(positive_prompts))
    elif prompt_type == "negative":
        priors = negative_prompts
        prior_labels = np.zeros(len(negative_prompts))
    else:  # "both"
        priors = np.concatenate((positive_prompts, negative_prompts), axis=0)
        prior_labels = np.concatenate((np.ones(len(positive_prompts)), np.zeros(len(negative_prompts))))
    
    # Convert to NumPy array for padding
    priors = np.array(priors)

    # Pad arrays so sizes remain equal between different patches
    # We take an extremely high sample as the number of positive prompts can be significantly larger compared to the other methods
    priors_padded = np.pad(priors, ((0, 1000 - priors.shape[0]), (0, 0)), mode='constant', constant_values=0)
    prior_labels_padded = np.pad(prior_labels, (0, priors_padded.shape[0] - len(prior_labels)), mode='constant', constant_values=99)

    return torch.tensor(priors_padded), torch.tensor(prior_labels_padded, dtype=torch.uint8)

def cluster_marinedebris(image, label, num_clusters=5, prompt_type="both", plotting=False):
    """
    Cluster marine debris pixels and return the coordinates of pixels closest to cluster centroids.
    
    Parameters:
    - image (np.array): The input image array.
    - label: torch.Tensor, binary tensor of shape [1, height, width] with values 0 or 255.
    - num_clusters (int): The number of clusters for K-means. Default is 5.
    
    Returns:
    - List of tuples with coordinates of pixels closest to cluster centroids.
    """
    
    label = label.squeeze().numpy()
    debris_coords = np.argwhere(label == 255) # XY locations within marine debris patches

    # Extract values of pixels within marine debris patches
    image_scaled = percentile_rescale(image, [1, 99]) # Normalize image into 0-255 range for RGB images
    debris_values = image_scaled[label == 255]
       
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(debris_values)
    
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    # Find the closest pixel to each cluster center in the RGB space
    closest_pixels = []
    for center in cluster_centers:
        # Calculate the distance from each debris pixel's RGB value to the cluster center
        distances = np.linalg.norm(debris_values - center, axis=1)
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
            scatter = axes[i].scatter(debris_values[:, x_idx], debris_values[:, y_idx], c=cluster_labels, cmap='tab10', s=10)
            axes[i].set_title(f'{x_color} vs. {y_color}')
            axes[i].set_xlim(0, 255)  # Set limits for RGB values
            axes[i].set_ylim(0, 255)  # Set limits for RGB values
            axes[i].grid(True)  # Add grid lines to the RGB plots

        # Overlay the cluster centers on the original image
        axes[3].imshow(image_scaled)
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


def extract_prompts_skeleton(label, min_pixel_count=10, prompt_type="both"):
    """
    Extract point prompts from the skeleton for masks with a minimum pixel count.
    
    Parameters:
    - label: torch.Tensor, binary tensor of shape [1, height, width] with values 0 or 255.
    - min_pixel_count: int, minimum number of pixels for a mask to be considered.
    - prompt_type: str, type of prompts to return ("positive", "negative", "both").
    
    Returns:
    - points: torch.Tensor, padded list of (x, y) tuples representing the skeleton points of the masks.
    - point_labels: torch.Tensor, corresponding labels (1 for foreground, 0 for background, 99 for padding).
    """
    np.random.seed(42)

    # Convert the tensor to a NumPy array and squeeze to remove the single batch dimension
    label_np = label.squeeze().numpy()

    # Convert the binary image to 0 and 1 format for skeletonize
    binary_image = (label_np == 255).astype(np.uint8)

    # Find connected components (i.e., masks) in the binary image
    labeled_array, num_masks = nd_label(binary_image)

    skeleton_points = []
    for mask_idx in range(1, num_masks + 1):
        mask_points = np.argwhere(labeled_array == mask_idx)

        # Filter out masks with fewer than <min_pixel_count> pixels
        if len(mask_points) < min_pixel_count:
            continue

        # Create binary mask for current mask index
        mask_binary = (labeled_array == mask_idx).astype(np.uint8)

        # Skeletonize the binary mask
        skeleton = skeletonize(mask_binary)

        # From the skeleton sample single point (i.e., one prompt per mask)
        skel_points = np.argwhere(skeleton)
        skel_point = skel_points[np.random.choice(len(skel_points))]
        skel_point = (int(skel_point[1]), int(skel_point[0]))  # Convert to (x, y)
        
        # Append skeleton points to the list
        skeleton_points.append(skel_point)

    # Generate negative (i.e., background) prompts
    num_positive_prompts = len(skeleton_points)
    negative_prompts = sample_random_prompts(label_np, num_positive_prompts, prompt_value=0)

    if prompt_type == "positive":
        priors = skeleton_points
        prior_labels = np.ones(len(skeleton_points))
    elif prompt_type == "negative":
        priors = negative_prompts
        prior_labels = np.zeros(len(negative_prompts))
    else:  # "both"
        priors = np.concatenate((skeleton_points, negative_prompts), axis=0)
        prior_labels = np.concatenate((np.ones(len(skeleton_points)), np.zeros(len(negative_prompts))))

    # Convert to NumPy array for padding
    priors = np.array(priors)

    # Pad arrays so sizes remain equal between different patches
    priors_padded = np.pad(priors, ((0, 50 - priors.shape[0]), (0, 0)), mode='constant', constant_values=0)
    prior_labels_padded = np.pad(prior_labels, (0, priors_padded.shape[0] - len(prior_labels)), mode='constant', constant_values=99)

    return torch.tensor(priors_padded), torch.tensor(prior_labels_padded, dtype=torch.uint8)

def extract_prompts_centroid(label, min_pixel_count=10, prompt_type="both"):
    """
    Extract point prompts (centroids or negative prompts) for masks with a minimum pixel count.
    
    Parameters:
    - label: torch.Tensor, binary tensor of shape [1, height, width] with values 0 or 255.
    - min_pixel_count: int, minimum number of pixels for a mask to be considered.
    - prompt_type: str, type of prompts to return ("positive", "negative", "both").
    
    Returns:
    - points: torch.Tensor, padded list of (x, y) tuples representing the prompts.
    - point_labels: torch.Tensor, corresponding labels (1 for foreground, 0 for background, 99 for padding).
    """
    np.random.seed(42)

    def is_point_in_mask(point, mask):
        x, y = point
        return mask[y, x] == 255

    # Convert the tensor to a NumPy array and squeeze to remove the single batch dimension
    label_np = label.squeeze().numpy()

    # Find connected components (i.e., masks) in the binary image
    labeled_array, num_masks = nd_label(label_np == 255)

    centroids = []
    for mask_idx in range(1, num_masks + 1):
        mask_points = np.argwhere(labeled_array == mask_idx)

        # Filter out masks with fewer than <min_pixel_count> pixels
        if len(mask_points) < min_pixel_count:
            continue

        # Calculate the geometric centroid
        centroid = np.mean(mask_points, axis=0)
        centroid = int(centroid[1]), int(centroid[0])  # Convert to (x, y)

        # Check if the geometric centroid is inside the mask
        if is_point_in_mask(centroid, label_np):
            centroids.append(centroid)
        else:
            # If centroid is not in the mask, find the closest mask point to the geometric centroid
            distances = distance.cdist([centroid], mask_points)
            closest_point = mask_points[np.argmin(distances)]
            closest_point = int(closest_point[1]), int(closest_point[0])  # Convert to (x, y)
            centroids.append(closest_point)
    
    # Generate negative (i.e., background) prompts
    num_positive_prompts = len(centroids)
    negative_prompts = sample_random_prompts(label_np, num_positive_prompts, prompt_value=0)

    if prompt_type == "positive":
        priors = centroids
        prior_labels = np.ones(len(centroids))
    elif prompt_type == "negative":
        priors = negative_prompts
        prior_labels = np.zeros(len(negative_prompts))
    else:  # "both"
        priors = np.concatenate((centroids, negative_prompts), axis=0)
        prior_labels = np.concatenate((np.ones(len(centroids)), np.zeros(len(negative_prompts))))

    # Convert to NumPy array for padding
    priors = np.array(priors)
    
    # Pad arrays so sizes remain equal between different patches
    priors_padded = np.pad(priors, ((0, 50 - priors.shape[0]), (0, 0)), mode='constant', constant_values=0)
    prior_labels_padded = np.pad(prior_labels, (0, priors_padded.shape[0] - len(prior_labels)), mode='constant', constant_values=99)

    return torch.tensor(priors_padded), torch.tensor(prior_labels_padded, dtype=torch.uint8)