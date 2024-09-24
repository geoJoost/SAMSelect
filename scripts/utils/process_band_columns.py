def process_band_columns(df_results, band_columns, band_list, equation):
    # Map band indices to band names for readability
    if equation in ('top'):
        # Top-5 has to be unpacked twice
        df_results[band_columns] = df_results[band_columns].map(lambda x: ', '.join(band_list[i - 1] for i in x))
    else: 
        df_results[band_columns] = df_results[band_columns].map(lambda x: band_list[x - 1])

    return df_results