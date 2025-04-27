# SAMSelect: An Automated Spectral Index Search using Segment Anything

<!-- let's add those when they are ready
[[`paper`](google.com)][[`demo`](google.com)][[`dataset`](google.com)]
-->

> SAMSelect discovers three-channel visualizations from multispectral imagery where pre-specified objects are most visible. It builds on Meta's [Segment Anything Model (SAM)](https://segment-anything.com/) to rank band combinations based on segmentation accuracy in the 3-channel visualization. 

![SAMSelect](./doc/Flowchart_SAMSelect.png)

<!--
If you've found SAMSelect helpful in your research, we'd love to hear about it! Your feedback helps us continue to improve the tool.

**Want to share your work?** Cite us in your publications using the reference in the article.

**Have questions, suggestions, or encountered a bug?** Reach out to Joost van Dalen directly.

**Let's collaborate and make SAMSelect even better together!**
-->

## Getting Started

```
# install samselect and its dependencies
pip install git+ssh://git@github.com/geoJoost/SAMSelect.git

# run samselect on the demo data on the data folder
samselect --image data/accra_20181031.tif --annotations data/accra_20181031.tif

# please check for more instructions on usage
samselect --help 
```
To run SAMSelect on your own data, you can modify [this Google Earth Engine script](https://code.earthengine.google.com/b31594853f8b1752f7fcf79883062bf3) to 
1. download a Sentinel-2 image
2. annotate some of the objects that you would like to visualize better
3. download them to your Google drive (click "run" in "Tasks" tab)

## Example Visualizations

![SAMSelect visualizations](./doc/samselect_patches.png)

## Installation
```
conda create --name samselect

conda activate samselect

pip3 install torch torchvision torchaudio

conda env update --f environment.yml
```

## Model Checkpoints
First download the checkpoint for Segment Anything, and store this into /data/models/
- `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- **`default` or `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)**

## Dataset Requirements
We require the following data to initiate SAMSelect:

1. **Multispectral patches**: SAMSelect operates on georeferenced multispectral patches to generate the spectral search space. We recommend sizes of 128px or 256px for an optimal feature-to-image ratio, which improves the model's performance. 
2. **Labels**: For SAMSelect, labels must be provided to optimize (and rank) mIoU for band combinations. Labels can either be binary raster files (e.g., PNG) or polygon shapefiles following the `qualitative.shp` naming convention.
3. **Point prompts**: To prompt SAM, point data needs to be provided. These can either be manually annotated (`_pt` in the file name) or generated using K-means clustering (K=10) based on the given labels. Manual annotations are required for practical applications, as they are typically more straightforward to create and yield more reliable segmentation performance, especially when multiple smaller objects are present in the image.

An example of a dataset for SAMSelect is given below.
```bash
# Multispectral patches:
/patch_gt_2.tif
/patch_gt_14.tif
/patch_gt_32.tif

# Corresponding binary labels:
/patch_gt_2.png
/patch_gt_14.png
/patch_gt_32.png

# Alternatively, give polygon annotations:
/durban_20190424_qualitative_poly.cpg
/durban_20190424_qualitative_poly.dbf 
/durban_20190424_qualitative_poly.prj
/durban_20190424_qualitative_poly.qix
/durban_20190424_qualitative_poly.shp
/durban_20190424_qualitative_poly.shx 

# Point prompts:
/durban_20190424_pt.cpg
/durban_20190424_pt.dbf
/durban_20190424_pt.fix
/durban_20190424_pt.prj 
/durban_20190424_pt.qix 
/durban_20190424_pt.shp 
/durban_20190424_pt.shx 
```

## Configuration Settings
This section lists the user-configurable settings for running SAMSelect in `main.py`:
1. **band_list** Specifies the available multispectral bands in the TIFF files. For Sentinel-2 data, it might look like `["B1", "B2", "B3", "B4", ...]`
2. **narrow_search_bands** Limits the spectral search space to a subset of the bands in `band_list`. This can significantly reduce computation time, which can otherwise take several hours. Note that the bands listed here must exactly match those provided in `band_list`. An example would be `["B2", "B3", "B4", "B8", "B8A"]` for narrowing down analysis to visible and near-infrared bands.
3. **scaling** The normalization function applied to the reflectance data to scale into a `[0, 255]` range for RGB data. Default is percentile scaling (1%-99%), but different scaling options are available in the dataloader.
4. **equation** Specifies the type of visualizations SAMSelect will compute. The following visualization options are available:
- **bc**: Band Composites, which generate false-colour composites using three spectral bands
- **ndi**: Normalized Difference Indices, such as NDVI or NDWI
- **ssi**: Spectral Shape Indices, such as the Floating Algae Index
- **top**: RSI-top10, which uses a similar approach to `bc` but instead of individual spectral bands, the top-10 most informative indices (NDIs and SSIs) are assigned to RGB channels. This option requires prior calculation of NDIs and SSIs
5. **model_type**: Defines the image encoder variant for SAM. Default is `vit_b` which is recommended for its performance and computation time.
6. **sensor_type** Specifies the satellite sensor used to collect the images. Options are `S2A` (Sentinel-2A) and `S2B` (Sentinel-2B). Note: this setting is required for SSI and FDI calculations. Support for other satellites is not yet integrated.

Additional notes:
- Depending on the hardware (GPU/CPU), the number of spectral bands, and the number of images, processing time can vary from one hour to several. As a rule of thumb, processing of each visualization takes about 10 seconds using a GPU and 64-74 seconds using a CPU.
- If a previous SAMSelect output exists for the same configuration, the system will skip computation and directly move to generating tables and graphs from the cached data, reducing runtime significantly.

## Comparison Methods
In addition to SAMSelect's visualizations, three other methods are used for comparison purposes:
1. Floating Debris Index ([Biermann et al., 2020](https://www.nature.com/articles/s41598-020-62298-z))
2. Normalized Difference Vegetation Index (NDVI)
3. Principal component analysis (PCA).

Important note:
If using data from different sensors, you may need to adjust the associated functions in `spectral_indices.py`, as certain indices (e.g., NDVI) are hard-coded to specific band combinations, such as `['B4', 'B8']` for Sentinel-2. Ensure the correct bands are selected for your sensor type.
