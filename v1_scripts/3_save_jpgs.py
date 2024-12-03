
from coastsat import SDS_preprocess
import os

def get_filepath(inputs, satname,coregistered_name:str=None):
    """
    Create filepath to the different folders containing the satellite images.

    KV WRL 2018

    Arguments:
    -----------
    inputs: dict with the following keys
        'sitename': str
            name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
            ```
        'dates': list of str
            list that contains 2 strings with the initial and final dates in
            format 'yyyy-mm-dd':
            ```
            dates = ['1987-01-01', '2018-01-01']
            ```
        'sat_list': list of str
            list that contains the names of the satellite missions to include:
            ```
            sat_list = ['L5', 'L7', 'L8', 'L9', 'S2']
            ```
        'filepath': str
            filepath to the directory where the images are downloaded
    satname: str
        short name of the satellite mission ('L5','L7','L8','S2')

    coregistered_name: str
        name of the coregistered folder
    Returns:
    -----------
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images

    """
    sitename = inputs["sitename"]
    filepath_data = inputs["filepath"]

    fp_ms = os.path.join(filepath_data, sitename, satname, "ms")
    if coregistered_name:
        fp_ms = os.path.join(filepath_data, sitename, satname,  "ms",coregistered_name)
    fp_mask = os.path.join(filepath_data, sitename, satname, "mask")

    # access the images
    if satname == "L5":
        # access downloaded Landsat 5 images
        filepath = [fp_ms, fp_mask]
    elif satname in ["L7", "L8", "L9"]:
        # access downloaded Landsat 7 images
        fp_pan = os.path.join(filepath_data, sitename, satname, "pan")
        fp_mask = os.path.join(filepath_data, sitename, satname, "mask")
        filepath = [fp_ms, fp_pan, fp_mask]
    elif satname == "S2":
        # access downloaded Sentinel 2 images
        fp_swir = os.path.join(filepath_data, sitename, satname, "swir")
        fp_mask = os.path.join(filepath_data, sitename, satname, "mask")
        filepath = [fp_ms, fp_swir, fp_mask]

    return filepath

coregistered_name = "max_center_size_no_histogram_match_window_size_256x256"
session_dir = r'C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52'
apply_cloud_mask = True
cloud_mask_issue = False
cloud_threshold = 0.99
satname = "S2"

inputs = {
    "sitename": "ID_1_datetime11-04-24__04_30_52\coregistered",
    "filepath": r"C:\development\doodleverse\coastseg\CoastSeg\data",
    "satname": "S2",
}

# creates a list of paths to the tif files
# tif_paths = get_filepath(inputs, satname, coregistered_name)
tif_paths = get_filepath(inputs, satname)

# this gets the paths to the coregistered folders
print(tif_paths)

# im_fn["ms"] is the filepath of the ms file
filename = r"2023-04-02-22-28-04_S2_ID_1_datetime11-04-24__04_30_52_ms.tif"
# filename = r"2023-04-02-22-27-50_S2_ID_1_datetime11-04-24__04_30_52_ms.tif"

ms_dir = os.path.join(inputs["filepath"], inputs["sitename"], satname, "ms")

for filename in os.listdir(ms_dir):
    if filename.endswith("_ms.tif"):
        SDS_preprocess.save_single_jpg(
            filename= filename,#filename=im_fn["ms"],
            tif_paths=tif_paths,
            satname=satname,
            sitename=inputs["sitename"],
            cloud_thresh=cloud_threshold,
            cloud_mask_issue=cloud_mask_issue,
            filepath_data=inputs["filepath"],
            collection='C02',
            apply_cloud_mask=apply_cloud_mask,
        )
