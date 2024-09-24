def get_band_idx(band_list, band_combination, equation):
    # Retrieve band names, and replace with indices within the list; e.g., (B4, B3, B2) => [4, 3, 2]
    if equation ==  'top':
        # Since top-5 has double-nested values, we have to unpack the tuples twice
        # Like (('B2', 'B8'), ('B3', 'B8'), ('B2', 'B7')) => [[2, 8], [3, 8], [2, 7]]
        bands_idx = [[band_list.index(band) + 1 for band in bands] for bands in band_combination]
    else:
        bands_idx = [band_list.index(band) + 1 for band in band_combination]

    return bands_idx