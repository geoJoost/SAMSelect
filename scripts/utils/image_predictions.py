import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# Custom modules
from utils.get_band_idx import get_band_idx
from models.marinedebris_dataloader import SamForMarineDebris
from utils.feature_scaling_functions import percentile_rescale
from utils.metrics import calculate_metrics

torch.manual_seed(42), random.seed(42)


def get_img(sceneid, band_list, band_combination, equation, sensor_type, atm_level):
    bands_idx = get_band_idx(band_list, band_combination, equation)

    # Load dataset for marine debris
    dataset = SamForMarineDebris(sceneid, bands_idx, equation, sensor_type, atm_level)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize a list to store images
    images, patch_ids = [], []

    # Iterate over each batch item
    for batch_image, batch_label, batch_point_prompts, batch_point_labels, batch_patchid in dataloader:
        for idx in range(batch_image.shape[0]):
            # Unpack individual items
            image, label, point_prompts, point_labels, patchid = batch_image[idx], batch_label[idx], batch_point_prompts[idx], batch_point_labels[idx],  batch_patchid[idx]

            # Scale image
            image_scaled = percentile_rescale(image, [1, 99])

            # Append the scaled image to the list
            images.append(image_scaled), patch_ids.append(patchid)
        
    return images, batch_label, batch_point_prompts, patch_ids


def get_img_pred(sceneid, band_list, band_combination, equation, sensor_type, atm_level, mask_level):
    bands_idx = get_band_idx(band_list, band_combination, equation)

    # Select mask level
    mask_level_map = {'level-1':0, 'level-2': 1, 'level-3': 2}
    mask_level_idx = mask_level_map[mask_level]

    dataset = SamForMarineDebris(sceneid, bands_idx, equation, sensor_type, atm_level)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Define Segment Anything Model
    sam_checkpoint = r'data/models/sam_vit_b_01ec64.pth'
    model_type = "vit_b"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice used: {device}")

    # Load Segment Anything Model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)

    # Initialize a list to store images
    images, masks_lst = [], []

    # Iterate over each batch item
    for batch_image, batch_label, batch_point_prompts, batch_point_labels, batch_patchid in dataloader:
        for idx in range(batch_image.shape[0]):
            # Unpack individual items
            image, label, point_prompts, point_labels, patchid = batch_image[idx], batch_label[idx], batch_point_prompts[idx], batch_point_labels[idx], batch_patchid[idx]

            # Scale image
            image_scaled = percentile_rescale(image, [1, 99])

            # We conduct a simple prediction for vizualisation; for actual documentation read sam_predictor.py
            # Pre-processing
            point_prompts = point_prompts[~torch.all(point_prompts == 0, dim=1)] # Remove rows with (0,0) coordinates
            point_labels = point_labels[point_labels != 99]

            #points_labels = np.ones(points_prompt.shape[0]) # Generate foreground labels ([1]) for SAM
            label[label == 255] = 1 # Convert unique values from (0, 255) => (0, 1)

            # Feed image to SAM predictor
            predictor.set_image(np.array(image_scaled))

            # Execute and extract statistics
            masks, scores, logits = predictor.predict(
                point_coords= np.array(point_prompts),
                point_labels= np.array(point_labels),
                multimask_output=True,
            )

            # Append the scaled image to the list
            images.append(image_scaled), masks_lst.append(masks[mask_level_idx])
    
    predicted_masks = torch.stack([torch.from_numpy(mask) for mask in masks_lst]).unsqueeze(1)
    metrics = calculate_metrics(batch_label, predicted_masks)
    print(f"Additional metrics for top-1 visualization ('{band_combination}')")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    return images, batch_label, batch_point_prompts, batch_point_labels, batch_patchid, masks_lst