import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_spectral_statistics(sceneid, band_list, equation_list, model_type, spectral_shading):

    """ Spectral statistics & top visualizations """
    # Create an empty DataFrame to store the results
    df_allstats =  pd.DataFrame()

    # Loop over equations
    for equation in equation_list:
        # Read in the CSV
        df_scene = pd.read_csv(f"data/processed/{sceneid}_{equation}_{model_type}_results.csv")

        # Add two columns for sceneID and equation used
        df_scene[['sceneid',  'equation']] = sceneid, equation

        # Check number of band columns, which is varied depending on the equation used
        band_columns = ['band_1', 'band_2']
        if 'band_3' in df_scene.columns:
            band_columns.append('band_3')

        # For easier handling, create a single column from the band combinations
        df_scene['band_combination'] = df_scene.apply(lambda row: ' | '.join(row[band_columns]), axis=1)

        # Append items to the front of the list for GroupBy statement
        # Bit convoluted but we want to keep individual bands for later plotting
        keep_columns = ['sceneid', 'band_combination', 'equation'] + band_columns
        
        # Further melt the dataset to get a single row for each mask level
        df_melt = pd.melt(df_scene, id_vars=keep_columns, 
                            value_vars=['jaccard_lvl1', 'jaccard_lvl2', 'jaccard_lvl3'], var_name='mask_level', value_name='miou')
        df_melt.drop_duplicates(subset=['band_combination', 'mask_level'], inplace=True)

        # Select the top-5 best performing combinations
        df_top5 = df_melt.nlargest(5, 'miou')

        print(f"{'-'*40} Scene: {sceneid} with {equation} {'-'*40}")
        df_top5['miou'] = df_top5['miou'].round(3) * 100 # Convert decimals into percentages
        print(f"{df_top5[['sceneid', 'band_combination', 'mask_level', 'miou']]}\n")

        # Combine the dataframes
        df_allstats = pd.concat([df_allstats, df_top5])

    # We create a seperate dataframe excluding top-5 data, mainly as this would substantially alter the statistics of band frequency
    individual_bands = df_allstats[df_allstats['equation'] != 'top']

    # Count band occurrences in both scenes
    band_counts = individual_bands.groupby('sceneid')[['band_1', 'band_2', 'band_3']].apply(lambda x: x.stack().value_counts())
    band_counts = band_counts.reset_index().rename(columns={'level_1': 'band', 0:'count'})
    band_counts = band_counts.melt(id_vars='sceneid', var_name='band', value_name='count')

    # Create a DataFrame with the desired x-ticks and their corresponding frequencies to make sure dataframe fits in the correct order
    df_bands = pd.DataFrame({'band': band_list})
    df_bands = df_bands.merge(band_counts, on='band', how='left').fillna({'count': 0})

    """ Plotting of bar graph """
    def plot_shaded_area(ax, start, end, label):
        """
        Plot shaded area between two x-values with label within the figure.
        
        Parameters:
            ax (matplotlib.axes.Axes): The axes object to plot on.
            start (float): The starting x-coordinate of the shaded area.
            end (float): The ending x-coordinate of the shaded area.
            label (str): The label to be displayed within the figure.
        """
        # Extract y-limits from the axes object
        ymin, ymax = ax.get_ylim()

        # Define the x-coordinate for the center of the span
        x_center = (start + end) / 2

        # Create the shaded area with zero fill color
        #plt.rcParams['hatch.linewidth'] = 0.1
        ax.fill_betweenx([ymin, ymax], start, end, color='none', edgecolor='black', hatch='/', linewidth=0.5, alpha=0.1)

        # Add dashed lines as edges
        ax.plot([start, start], [ymin, ymax], color='black', linestyle='--', linewidth=0.5)
        ax.plot([end, end], [ymin, ymax], color='black', linestyle='--', linewidth=0.5)

        # Add label within the figure
        ax.annotate(label, xy=(x_center, ymax - 0.05 * (ymax - ymin)), xytext=(x_center, ymax - 0.05 * (ymax - ymin)),
                        ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))

    ## Plot ##
    # Define custom colors
    custom_palette = ['#fca311']

    # Create a barchart using seaborn
    plt.rcParams["font.family"] = "sans-serif"
    plt.figure(figsize=(6.789, 4)) # LaTeX width

    # Actual data
    ax = sns.barplot(x='band', y='count', hue='sceneid', data=df_bands, 
                     palette=custom_palette, linewidth=.5, edgecolor="#14213d", zorder=3)
    
    ax.set_axisbelow(True)

    # Set shaded areas to group bands
    if spectral_shading == True:
        plot_shaded_area(ax, 0.5, 3.5, 'VIS')       # B2 - B4
        plot_shaded_area(ax, 3.5, 6.5, 'Red Edge')  # B5 - B7
        plot_shaded_area(ax, 6.5, 8.5, 'NIR')       # B8 - B8A
        plot_shaded_area(ax, 9.5, 12.5, 'SWIR')    # B11 - B12
    else:
        pass

    # Figure aesthetics
    ax.grid(lw=0.2)
    sns.despine(offset=0, trim=False, bottom=False)

    # Legend
    plt.legend()

    # Text
    plt.title('Spectral bands chosen by SAMSelect in top-5 visualizations')
    plt.xlabel('Sentinel-2 bands', fontsize=11)
    plt.ylabel('Frequency in BC, NDI or SSI', fontsize=11)

    plt.tight_layout()
    ax.set_rasterized(True)
    
    # Saving
    os.makedirs("doc/figures", exist_ok=True)
    output_path = f'doc/figures/{sceneid}_band_frequency'
    plt.savefig(f'{output_path}.png', dpi=600) # PNG
    plt.savefig(f'{output_path}.pdf')          # PDF
    plt.show()
    plt.close()
    
    print(f"Top-5 most frequently selected spectral bands saved in '{output_path}")

    """ Retrieve top-1 result to use in function plot_patches() """
    # Get the visualization with the maximum mIoU score
    best_row = df_allstats[df_allstats['miou'] == df_allstats['miou'].max()]

    # For use in plot_patches, we need to convert the pretty-print bands back into a proper list to give it as argument
    # For this we need to distinguish single nesting (BC, NDI, SSI) or double-nesting (RSI-top10)
    if (best_row['equation'] == 'top').any():  # RSI-top10
        top1_combination = [
            [band.strip() for band in best_row[col].iloc[0].split(', ')] # => [['B2', 'B8'], ['B1', 'B8A'], ['B3', 'B8']]
            for col in ['band_1', 'band_2', 'band_3']
    ]
    else:  # BC, NDI, SSI
        # For other equations, just create lists from non-null values
        top1_combination = best_row[['band_1', 'band_2', 'band_3']].apply(
            lambda row: [band for band in row if pd.notna(band)], axis=1).iloc[0] # => ['B2', 'B8']
    
    # And extract the equation / visualization type
    equation_type = best_row['equation'].iloc[0]

    print(f"Best visualization found by SAMSelect is: '{equation_type}' with an mIoU score of {best_row['miou'].iloc[0]}%")
    print(f"Using bands: {top1_combination}")  

    return top1_combination, equation_type 



