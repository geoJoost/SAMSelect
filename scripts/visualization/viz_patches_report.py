import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from utils.image_predictions import get_img, get_img_pred

def plot_combined_patches(sceneids, band_lists, top1_combinations, mask_levels, patch_indices):
    no_scenes = len(sceneids)
    no_img_per_scene = len(patch_indices[0])  # Number of columns per scene, based on patch_indices
    no_rows = 7  # Number of rows
    total_columns = no_img_per_scene * no_scenes

    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(no_rows, total_columns, figsize=(6.789, 7.5))  # Width corresponds to #textwidth in LaTeX

    binary_cmap = ListedColormap(['#FCF3EE', '#68000D'])

    for scene_index, (sceneid, band_list, top1_combination, mask_level, indices) in enumerate(zip(sceneids, band_lists, top1_combinations, mask_levels, patch_indices)):
        # Load optical data + labels for current scene
        images_vis, batch_label, point_prompts, patch_ids = get_img(sceneid, band_list, ['B4', 'B3', 'B2'], 'bc') # True-colour
        images_ndvi = get_img(sceneid, band_list, ['B8', 'B4'], 'ndi')[0] # NDVI
        images_fdi = get_img(sceneid, band_list, ["B8", "B6", "B11"], 'fdi')[0] # FDI
        
        images_topndi = get_img(sceneid, band_list, ['B2', 'B8'], 'ndi')[0]  # New NDI row using B2 and B8
        images_top1 = get_img(sceneid, band_list, top1_combination, 'top')[0] # Best combination of NDI & SSI
        images, batch_label, point_prompts, point_labels, batch_patchid, masks_lst = get_img_pred(sceneid, band_list, top1_combination, 'top', mask_level)
        
        no_img = len(indices)
        start_col = scene_index * no_img_per_scene

        # Plot all images for current scene
        # Row 1: True-colour
        for i, idx in enumerate(indices):
            image = images_vis[idx]
            patchid = patch_ids[idx]
            axes[0, start_col + i].imshow(image)
            axes[0, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Row 2: NDVI
        for i, idx in enumerate(indices):
            image_ndvi = images_ndvi[idx]
            axes[1, start_col + i].imshow(image_ndvi)
            axes[1, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Row 3: FDI
        for i, idx in enumerate(indices):
            image_fdi = images_fdi[idx]
            axes[2, start_col + i].imshow(image_fdi)
            axes[2, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Row 4: NDI using B2 and B8
        for i, idx in enumerate(indices):
            image_topndi = images_topndi[idx]
            axes[3, start_col + i].imshow(image_topndi)
            axes[3, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Row 5: Top-1
        for i, idx in enumerate(indices):
            image_top1 = images_top1[idx]
            axes[4, start_col + i].imshow(image_top1)
            axes[4, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Row 6: Label / Reference
        for i, idx in enumerate(indices):
            label = batch_label.squeeze(0)[idx]
            points = point_prompts[idx]
            p_labels = point_labels[idx]
            axes[5, start_col + i].imshow(label.permute(1, 2, 0), cmap=binary_cmap)
            axes[5, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            # Plot point prompts used for prediction
            valid_points = points[~torch.all(points == 0, dim=1)].numpy()
            valid_p_labels = p_labels[p_labels != 99].numpy()

            # Plot each point with the appropriate color
            for point, p_label in zip(valid_points, valid_p_labels):
                color = 'green' if p_label == 1 else 'red'
                #axes[5, start_col + i].scatter(point[0], point[1], c=color, s=10, marker='s', label='MD point')

        # Row 7: Top-1 prediction
        for i, idx in enumerate(indices):
            mask = masks_lst[idx]
            points = point_prompts[idx]
            p_labels = point_labels[idx]
            axes[6, start_col + i].imshow(mask.astype(int), cmap=binary_cmap)
            axes[6, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            valid_points = points[~torch.all(points == 0, dim=1)].numpy()
            valid_p_labels = p_labels[p_labels != 99].numpy()

            # Plot each point with the appropriate color
            for point, p_label in zip(valid_points, valid_p_labels):
                color = 'green' if p_label == 1 else 'red'
                #axes[6, start_col + i].scatter(point[0], point[1], c=color, s=10, marker='s', label='MD point')

    # Text for row labels
    rows = ['VIS', 'NDVI', 'FDI', r'NDI$_{B2, B8}$', 'Top-1', 'Label', 'Top-1\nprediction']
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, horizontalalignment='right', fontsize=10)

    # Titles for scenes
    fig.text(0.545, 0.986, '← Accra', ha='right', va='center', fontsize=10)
    fig.text(0.565, 0.986, 'Durban →', ha='left', va='center', fontsize=10)

    # Reduce white space between columns and rows
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    # Save
    plt.tight_layout()

    plt.savefig(f"reports/figures/combined_patches.png", dpi=600, transparent=True)
    plt.savefig(f'reports/figures/combined_patches.pdf') 
    plt.close()
    
    print(f"Finished printing combined plot.")

sceneids = ['accra_20181031_l2a', 'durban_20190424_l2a']
band_lists = [["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
              ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]]
top1_combinations = [['B2', 'B8'], ['B1', 'B8', 'B11'], ['B2', 'B8', 'B11']], [['B2', 'B8'], ['B1', 'B8A'], ['B3', 'B8']]
mask_levels = ['level-3', 'level-3']
patch_indices = [
    [0, 2, 3],  # Indices for Accra to pick more representative patches
    [0, 2, 3]   # Indices for Durban
]

plot_combined_patches(sceneids, band_lists, top1_combinations, mask_levels, patch_indices)