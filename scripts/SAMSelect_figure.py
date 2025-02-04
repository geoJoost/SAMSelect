import matplotlib.pyplot as plt
from utils.image_predictions import get_img

""" Stand-alone script for printing the figure used in the article """

def plot_combined_patches(sceneids, band_lists, top1_combinations, patch_indices):
    no_scenes = len(sceneids)
    no_img_per_scene = len(patch_indices[0])  # Number of columns per scene, based on patch_indices
    no_rows = 5  # Number of rows
    total_columns = no_img_per_scene * no_scenes

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    fig, axes = plt.subplots(no_rows, total_columns, figsize=(3.5, 4.5))

    for scene_index, (sceneid, band_list, top1_combination, indices) in enumerate(zip(sceneids, band_lists, top1_combinations, patch_indices)):
        # Load optical data + labels for current scene
        images_vis, batch_label, point_prompts, point_labels, patch_ids = get_img(sceneid, band_list, ['B4', 'B3', 'B2'], 'bc') # True-colour
        images_ndvi = get_img(sceneid, band_list, ['B8', 'B4'], 'ndi')[0] # NDVI
        images_fdi = get_img(sceneid, band_list, ["B8", "B6", "B11"], 'fdi')[0] # FDI
        
        images_topndi = get_img(sceneid, band_list, ['B2', 'B8'], 'ndi')[0]  # New NDI using B2 and B8
        images_top1 = get_img(sceneid, band_list, top1_combination, 'top')[0] # Best combination of NDI & SSI (RSI-top10)
        
        start_col = scene_index * no_img_per_scene

        # Plot all images for current scene
        # Row 1: True-colour
        for i, idx in enumerate(indices):
            image = images_vis[idx]
            patchid = patch_ids[idx]
            axes[0, start_col + i].imshow(image)
            axes[0, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        # Row 2: Top-1 result; RSI-top10 in this case
        for i, idx in enumerate(indices):
            image_top1 = images_top1[idx]
            axes[1, start_col + i].imshow(image_top1)
            axes[1, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Row 3: NDVI
        for i, idx in enumerate(indices):
            image_ndvi = images_ndvi[idx]
            axes[2, start_col + i].imshow(image_ndvi)
            axes[2, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Row 4: FDI
        for i, idx in enumerate(indices):
            image_fdi = images_fdi[idx]
            axes[3, start_col + i].imshow(image_fdi)
            axes[3, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Row 5: NDI using B2 and B8
        for i, idx in enumerate(indices):
            image_topndi = images_topndi[idx]
            axes[4, start_col + i].imshow(image_topndi)
            axes[4, start_col + i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Text for row labels
    rows = ['VIS', r'FCC$_{index}$', 'NDVI', 'FDI', r'NDI$_{B2, B8}$']
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=90, fontsize=10)#, horizontalalignment='right', fontsize=10)

    # Titles for scenes
    scene_titles = ['Accra', 'Durban', 'PLP']
    for i, title in enumerate(scene_titles):
        axes[0, i * no_img_per_scene].set_title(title, fontsize=10)  # Set title for each column

    # Reduce white space between columns and rows
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    #plt.tight_layout()

    # Save
    plt.savefig(f"doc/figures/samselect_patches.png", dpi=600, transparent=False, bbox_inches='tight')
    plt.savefig(f'doc/figures/samselect_patches.pdf', bbox_inches='tight') 
    plt.close()
    
    print(f"Finished printing combined plot.")

sceneids = ['accra_20181031_l2a', 'durban_20190424_l2a', 'PLP2021']
band_lists = [["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
              ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"],
              ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]]

top1_combinations = [[['B2', 'B8'], ['B1', 'B8', 'B11'], ['B2', 'B8', 'B11']],
                    [['B2', 'B8'], ['B1', 'B8A'], ['B3', 'B8']],
                    [['B2', 'B8'], ['B1', 'B8A'], ['B3', 'B8'],]] # Top-1 result of Durban selected for PLP2021

patch_indices = [
    [0], # Representative debris patch in Accra
    [2], # Representative debris patch in Durban
    [0]  # PLP2021 image acquired on 2021-06-21
]

plot_combined_patches(sceneids, band_lists, top1_combinations, patch_indices)