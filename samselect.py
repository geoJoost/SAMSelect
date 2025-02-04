import argparse

def main():
    """Main entry point for the samselect command-line tool."""
    parser = argparse.ArgumentParser(description="SAMSelect Command-Line Tool")
    parser.add_argument(
        "-i", "--image", type=str, help="Input Sentinel-2 image", required=True
    )
    parser.add_argument(
        "-a", "--annotations", type=str, help="Shapefile with object annotations in the Sentinel-2 image", required=True
    )

    args = parser.parse_args()

    # Placeholder logic
    print("SAMSelect is running...")
    print("Processing complete!")

if __name__ == "__main__":
    main()