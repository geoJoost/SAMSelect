import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import pandas as pd

# Custom modules
from scripts.models.dataloader import SamForMarineDebris
from scripts.utils.feature_scaling_functions import minmax_rescale, percentile_rescale, histogram_rescale, adaptive_histogram_rescale
from scripts.utils.metrics import calculate_metrics

def execute_SAM(tif_path, polygon_path, bands_idx, scaling, equation='bc', model_type='vit_b', atm_level='L2A'):
    """
    Executes the Segment Anything Model (SAM) on a Sentinel-2 scene to detect marine debris.

    This function loads a Sentinel-2 scene and corresponding polygon annotations, processes the data into smaller patches,
    and applies the Segment Anything Model (SAM) to detect marine debris. It supports various visualization modules and
    scaling methods to preprocess the image data before feeding it to the SAM model. The function evaluates the model's
    performance using the Jaccard index (IoU) for different mask levels.

    Parameters:
    - tif_path (str): Path to the Sentinel-2 scene (TIFF format).
    - polygon_path (str): Path to the polygon annotations for objects-of-interest (shapefile format).
    - bands_idx (list of integers): List of Sentinel-2 bands idx's to be used.
    - scaling (str): Scaling method to normalize the image data from DN into RGB range (0-255). Options include:
        - 'none': No scaling.
        - 'min-max': Min-Max normalization.
        - 'equalize': Histogram equalization.
        - 'percentile_1-99': Percentile scaling (1% - 99%).
        - 'percentile_2-98': Percentile scaling (2% - 98%).
        - 'percentile_5-95': Percentile scaling (5% - 95%).
        - 'adaptive_equalize_01': Adaptive histogram equalization (clip limit 0.01).
        - 'adaptive_equalize_03': Adaptive histogram equalization (clip limit 0.03).
        - 'adaptive_equalize_05': Adaptive histogram equalization (clip limit 0.05).
        
    - equation (str): Option for the visualization module. Default is 'bc' (Band Composite). Other options include:
        - 'bc': Band Composite
        - 'ndi': Normalized Difference Index
        - 'ssi': Spectral Shape Index
        - 'top': Spectral Index Composite
        - 'pca': Principal Component Analysis
        - 'fdi': Floating Debris Index
        
    - model_type (str): SAM encoder type. Default is 'vit_b' (ViT-Base). Other options include:
        - 'vit_h': ViT-Huge
        - 'vit_l': ViT-Large
        - 'vit_b': ViT-Base
        
    - atm_level (str): Indicates the Sentinel-2 product level. Default is 'L2A'. Supported levels include:
        - 'L1C': Top-of-atmosphere reflectance (Sen2Cor)
        - 'L2A': Bottom-of-atmosphere reflectance (Sen2Cor)
        - 'L1R': Rayleigh-corrected reflectance (ACOLITE)
        - 'L2R': Rayleigh and aerosol-corrected reflectance (ACOLITE)

    Returns:
    - df_results (pd.DataFrame): DataFrame containing the results of the SAM model, including Jaccard index (IoU) for different mask levels.
    - image_scaled (torch.Tensor): The scaled image tensor used for SAM prediction.

    The function performs the following steps:
    1. Loads the dataset using the `SamForMarineDebris` class.
    2. Defines and loads the SAM model.
    3. Processes the image data using the specified scaling method.
    4. Feeds the processed image data to the SAM predictor.
    5. Evaluates the model's performance using the Jaccard index (IoU) for different mask levels.
    6. Returns the results and the scaled image tensor.
    """
    # Load dataset for marine debris
    dataset = SamForMarineDebris(tif_path, polygon_path, bands_idx, equation, atm_level)
    dataloader = DataLoader(dataset, batch_size=45, shuffle=False, num_workers=0)

    # Define Segment Anything Model
    sam_models = {
    'vit_h': r"data/models/sam_vit_h_4b8939.pth", # ViT-Huge
    'vit_l': r"data/models/sam_vit_l_0b3195.pth", # ViT-Large
    'vit_b': r'data/models/sam_vit_b_01ec64.pth', # ViT-Base
    }
    sam_checkpoint = sam_models[model_type]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device used: {device}")

    # Download the SAM encoder if it does not exist
    if not os.path.exists(sam_checkpoint):
        import requests
        print(f"SAM encoder not found at {sam_checkpoint}. Downloading...")
        model_dir = os.path.dirname(sam_checkpoint)
        model_url = f"https://dl.fbaipublicfiles.com/segment_anything/{os.path.basename(sam_checkpoint)}" # Download ViT encoder from Meta AI
        os.makedirs(model_dir, exist_ok=True)
        response = requests.get(model_url)
        with open(sam_checkpoint, 'wb') as f:
            f.write(response.content)            
    
    assert os.path.exists(sam_checkpoint), "SAM encoder not found. Please add one to 'data/models'."

    # Load Segment Anything Model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)

    masks_lvl1, masks_lvl2, masks_lvl3 = [], [], []
    for batch_image, batch_label, batch_point_prompts, batch_point_labels, batch_patchid in dataloader:
        for idx in range(batch_image.shape[0]):
            # Unpack individual items
            image, label, point_prompts, point_labels, patchid = batch_image[idx], batch_label[idx], batch_point_prompts[idx], batch_point_labels[idx], batch_patchid[idx]

            # Pre-processing
            points_prompt = point_prompts[~torch.all(point_prompts == 0, dim=1)].numpy() # Remove padding
            point_labels = point_labels[point_labels != 99]                              # Remove padding
            label[label == 255] = 1                                                      # Convert binary values from (0, 255) => (0, 1)         

            # Retrieve the feature scaling / normalization method
            scaling_methods = {
                'none': ('None', lambda img: img),
                'min-max': ('Min-Max normalization', minmax_rescale),
                'equalize': ('Histogram equalization', histogram_rescale),
                'percentile_1-99': (f'Percentile (1% - 99%)', lambda img: percentile_rescale(img, [1, 99])),
                'percentile_2-98': (f'Percentile (2% - 98%)', lambda img: percentile_rescale(img, [2, 98])),
                'percentile_5-95': (f'Percentile (5% - 95%)', lambda img: percentile_rescale(img, [5, 95])),
                'adaptive_equalize_01': ('Adaptive histogram equalization (0.01)', lambda img: adaptive_histogram_rescale(img, 0.01)),
                'adaptive_equalize_03': ('Adaptive histogram equalization (0.03)', lambda img: adaptive_histogram_rescale(img, 0.03)),
                'adaptive_equalize_05': ('Adaptive histogram equalization (0.05)', lambda img: adaptive_histogram_rescale(img, 0.05)),
            }
            assert scaling in scaling_methods, f"Invalid scaling method: {scaling}"

            # Scale each individual band into [0, 255] range for SAM
            image_scaled = scaling_methods[scaling][1](image)

            # Feed image to SAM predictor
            predictor.set_image(np.array(image_scaled.permute(1, 2, 0)))

            # Execute and extract statistics
            masks, scores, logits = predictor.predict(
                point_coords=points_prompt,
                point_labels=point_labels,
                multimask_output=True,
            )
            masks_lvl1.append(masks[0]), masks_lvl2.append(masks[1]), masks_lvl3.append(masks[2])

    # Evalute model results
    pred_lvl1 = torch.stack([torch.from_numpy(mask) for mask in masks_lvl1]).unsqueeze(1)
    pred_lvl2 = torch.stack([torch.from_numpy(mask) for mask in masks_lvl2]).unsqueeze(1)
    pred_lvl3 = torch.stack([torch.from_numpy(mask) for mask in masks_lvl3]).unsqueeze(1)

    # Append aggregated results to the list
    results = {
        **{f'band_{i + 1}': bands_idx[i] for i in range(len(bands_idx))}, # Number of bands is varied, with three in BC & SSI, but only two for NDI
        'scaler': scaling_methods[scaling][0],
        'model': model_type,
        'jaccard_lvl1': calculate_metrics(batch_label, pred_lvl1)['jaccard'], # IoU for mask-level 1 from SAM
        'jaccard_lvl2': calculate_metrics(batch_label, pred_lvl2)['jaccard'],
        'jaccard_lvl3': calculate_metrics(batch_label, pred_lvl3)['jaccard'],
    }

    df_results = pd.DataFrame([results])

    return df_results, image_scaled