import argparse
from scripts.main import samselect_wrapper

def main():
    """Main entry point for the samselect command-line tool."""
    parser = argparse.ArgumentParser(description="SAMSelect Command-Line Tool")
    parser.add_argument(
        "-i", "--image", type=str, help="Path to the Sentinel-2 scene (TIFF)", required=True
    )
    parser.add_argument(
        "-a", "--annotations", type=str, help="Path to the shapefile with polygon annotations", required=True
    )
    parser.add_argument(
        "-b", "--bands", type=str, nargs='+', help="List of Sentinel-2 L2A bands. Default: ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']",
        default=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']  # Default L2A bands
    )
    parser.add_argument(
        "-n", "--narrow_search_bands", type=str, nargs='+', help="Manual selection of bands for narrow spectral search. Define as a list like [B1, B2, B3, B4]. Default: None for exhaustive search space.",
        default=['B1', 'B2', 'B3', 'B4']  # Default narrow search bands
    )
    parser.add_argument(
        "-s", "--scaling", type=str, help="Normalization function to arrange DN into RGB range (0-255). Default: percentile scaling with 1-99%",
        default='percentile_1-99'  # Default scaling
    )
    parser.add_argument(
        "-e", "--equations", type=str, nargs='+',
        help="Visualization modules. Default: ['bc']. Options include: 'bc': Band Composites, 'ndi': Normalized Difference Indices, 'ssi': Spectral Shape Indices, 'sic': Spectral Index Composite.",
        default=['bc']  # Default equation list
    )
    parser.add_argument(
        "-m", "--model_type", type=str, help="SAM encoder type. Default: 'vit_b'",
        default='vit_b'  # Default model type
    )

    args = parser.parse_args()

    # Check if required arguments are provided
    if not args.image or not args.annotations:
        parser.error("The following arguments are required: --image (Sentinel-2 scene as .tif), --annotations (polygon annotations as .shp)")

    print("SAMSelect is running...")
    print("Note: Computation time can take several hours, depending on the number of images and size of the search space.")

    # Call the samselect_wrapper function with the parsed arguments
    samselect_wrapper(
        tif_path=args.image,
        polygon_path=args.annotations,
        band_list=args.bands,
        narrow_search_bands=args.narrow_search_bands,
        scaling=args.scaling,
        equation_list=args.equations,
        model_type=args.model_type
    )

    print("Processing complete!")

if __name__ == "__main__":
    main()
