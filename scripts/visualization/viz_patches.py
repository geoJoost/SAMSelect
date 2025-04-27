import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

# Custom imports
from scripts.utils.image_predictions import get_img, get_img_pred

def plot_patches(tif_path, polygon_path, band_list, top1_combination, top1_equation, top1_masklevel):
    sceneid = os.path.splitext(os.path.basename(tif_path))[0]
    
    # Make patches for true-colour (VIS), NDVI and FDI
    # Also load the labels and point prompts
    images_vis, batch_label, point_prompts, point_labels, patch_ids = get_img(tif_path, polygon_path, band_list, ['B4', 'B3', 'B2'], 'bc') # True-colour
    images_ndvi = get_img(tif_path, polygon_path, band_list, ['B8', 'B4'], 'ndi')[0] # NDVI
    images_fdi = get_img(tif_path, polygon_path, band_list, ["B8", "B6", "B11"], 'fdi')[0] # FDI

    # Retrieve top-1 results from SAMSelect
    images_top1 = get_img(tif_path, polygon_path, band_list, top1_combination, top1_equation)[0] # Spectral Index Composites
    images, batch_label, point_prompts, point_labels, batch_patchid, masks_lst = get_img_pred(tif_path, polygon_path, band_list, top1_combination, top1_equation, top1_masklevel)
        
    # Plot all images in a single row
    no_img = len(images_vis)
    no_rows = 6
    fig, axes = plt.subplots(no_rows, no_img , figsize=(no_img * 3, no_rows * 2.5), squeeze=False)

    binary_cmap = ListedColormap(['#FCF3EE', '#68000D'])

    ### Data ###
    # Row 1: True-colour
    for i, (image, patchid) in enumerate(zip(images_vis, patch_ids)):
        axes[0,i].imshow(image.permute(1, 2, 0))
        axes[0,i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        axes[0,i].set_title(f"#{patchid}")

    # Row 2: NDVI
    for i, image_ndvi in enumerate(images_ndvi):
        axes[1,i].imshow(image_ndvi.permute(1, 2, 0))
        axes[1,i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Row 3: FDI
    for i, image_fdi in enumerate(images_fdi):
        axes[2,i].imshow(image_fdi.permute(1, 2, 0))
        axes[2,i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Row 4: Top-1 from SAMSelect
    for i, image_top1 in enumerate(images_top1):
        axes[3,i].imshow(image_top1.permute(1, 2, 0))
        axes[3,i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Row 5: Label / Reference
    for i, (label, points, p_labels) in enumerate(zip(batch_label.squeeze(0), point_prompts, point_labels)):
        if label.dim() == 2:
            axes[4, i].imshow(label, cmap=binary_cmap)
        else:
            axes[4, i].imshow(label.permute(1, 2, 0), cmap=binary_cmap)
        axes[4,i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Plot point prompts used for prediction
        valid_points = points[~torch.all(points == 0, dim=1)].numpy()
        valid_p_labels = p_labels[p_labels != 99].numpy()

        # Plot each point with the appropriate color
        for point, p_label in zip(valid_points, valid_p_labels):
            color = '#DF9C2E' if p_label == 1 else 'red'
            axes[4, i].scatter(point[0], point[1], c=color, s=10, label='Prompt')

    # Row 6: Top-1 prediction
    for i, (mask, points, p_labels) in enumerate(zip(masks_lst, point_prompts, point_labels)):
        axes[5,i].imshow(mask.astype(int), cmap=binary_cmap)
        axes[5,i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        valid_points = points[~torch.all(points == 0, dim=1)].numpy()
        valid_p_labels = p_labels[p_labels != 99].numpy()

        # Plot each point with the appropriate color
        for point, p_label in zip(valid_points, valid_p_labels):
            color = '#DF9C2E' if p_label == 1 else 'red'
            axes[5, i].scatter(point[0], point[1], c=color, s=10, label='Prompt')

    # Text for row labels
    rows = ['VIS', 'NDVI', 'FDI', 'Top-1', 'Label', 'Top-1\nprediction']
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='large', horizontalalignment='right')

    # Save
    plt.tight_layout()
    plt.show()
    os.makedirs("doc/figures", exist_ok=True)  
    plt.savefig(f"doc/figures/{sceneid}_patches.png", dpi=600, transparent=False)
    plt.close()