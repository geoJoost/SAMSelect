import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import pandas as pd

from models.marinedebris_dataloader import SamForMarineDebris
from utils.feature_scaling_functions import minmax_rescale, percentile_rescale, histogram_rescale, adaptive_histogram_rescale
from utils.metrics import calculate_metrics

# Set random seed for selecting random points, if num_points argument is given
np.random.seed(42)

def execute_SAM(sceneid, band_list, scaling, equation='bc', model_type='vit_b', sensor_type='S2B'):
    np.random.seed(42)
    # Load dataset for marine debris
    dataset = SamForMarineDebris(sceneid, band_list, equation, sensor_type)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Define Segment Anything Model
    sam_models = {
    'vit_h': r"data/models/sam_vit_h_4b8939.pth", # ViT-Huge
    'vit_l': r"data/models/sam_vit_l_0b3195.pth", # ViT-Large
    'vit_b': r'data/models/sam_vit_b_01ec64.pth', # ViT-Base
    }
    sam_checkpoint = sam_models[model_type]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device used: {device}")

    # Load Segment Anything Model
    assert os.path.exists(sam_checkpoint), "SAM encoder not found. Please add one to 'data/models'."
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
            predictor.set_image(np.array(image_scaled))

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
        **{f'band_{i + 1}': band_list[i] for i in range(len(band_list))}, # Number of bands is varied, with three in BC & SSI, but only two for NDI
        'scaler': scaling_methods[scaling][0],
        'model': model_type,
        'jaccard_lvl1': calculate_metrics(batch_label, pred_lvl1)['jaccard'],
        'jaccard_lvl2': calculate_metrics(batch_label, pred_lvl2)['jaccard'],
        'jaccard_lvl3': calculate_metrics(batch_label, pred_lvl3)['jaccard'],
    }

    df_results = pd.DataFrame([results])

    return df_results, image_scaled
