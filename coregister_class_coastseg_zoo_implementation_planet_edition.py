# Standard library imports
import os
import json
import glob
import re
import shutil
import planet_to_coastseg
import threading
from collections import OrderedDict

# Third-party library imports
import numpy as np
import pandas as pd
import tqdm

# Internal imports
import coregister_class
import plotting
import file_utilites as file_utils
import filters
import geo_utils

# For this specific script that's meant to work with planetscope data.
# OR AT LEAST this script will require this. For now it expects the analytic tifs and udm files to be in the same directory and have the format 20200603_203636_82_1068_3B_udm2_clip.tif
# You will need to run the preprocessing script to put all the files into the correct format
# sitename
#     - PS
#         - ms
#         - udm2
#         - meta
#     -jpg_files
#         -preprocessed
#             -RGB
#             -NIR
#     -coregistered



# Coregistation Steps (Specific to CoastSeg Zoo Implementation)
# 1. Give it a session
# 2. Give it a template
# 3. Give it a window size
# 4. Give it a matching window strategy
# 5. Give it a verbose setting
# 6. Give it a settings dictionary
# 7. Coregister the files
# 8. Save the results to a json file
# 9. Apply outlier filtering for each satellite ( each satellite has different rates of errors)
# 10. Delete the bad coregistrations
# 11. Apply the shifts to the other files
# 12. Create new jpgs for the coregistered files
# 13. Save the coregistered files to a new directory
# 14. Copy the unregistered files to this new directory
# 15. Update the config.json file with the coregistration settings as well as the location of the coregistered files


# Coregistration Settings
WINDOW_SIZE=(64,64)
VERBOSE = False
# these are the settings that are used by filter
filter_settings = {
    'min_shift_meters': (-200,-200),
    'max_shift_meters': (200,200),
    'z_threshold': 1.5,         # Set to None to disable outlier detection using the z score
}
settings = {
    'max_translation': 1000,
    'min_translation': -1000,
}
# matching_window_strategy='max_overlap' # this strategy does not work...
coregister_settings  = {
    'window_size': WINDOW_SIZE,
    'matching_window_strategy': 'max_center_size',
    'verbose': False,
    'settings': settings,
}


# Script behavior settings
replace_failed_files = False # If True, then replace bad coregistrations them with the original unregistered files in the coregistered directory 
create_jpgs = True # Create jpgs for the files that passed the filtering

# Session and Template Paths
base_dir = r'C:\development\coastseg-planet\downloads\UNALAKLEET_pier_cloud_0.7_TOAR_enabled_2020-06-01_to_2023-08-01\e2821741-0677-435a-a93a-d4f060f3adf4'
session_dir = r'C:\development\coastseg-planet\downloads\UNALAKLEET_pier_cloud_0.7_TOAR_enabled_2020-06-01_to_2023-08-01\e2821741-0677-435a-a93a-d4f060f3adf4\PSScene_but_less'
template_path = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52_original\L9\ms\2023-06-30-22-01-55_L9_ID_1_datetime11-04-24__04_30_52_ms.tif"

# make a new coregistered directory to save everything in ( not a problem is it already exists)
coregistered_dir=  os.path.join(base_dir,'coregistered')
os.makedirs(coregistered_dir,exist_ok=True)
result_json_path = os.path.join(coregistered_dir, 'transformation_results.json')

# get the 

results = {}
tif_files = file_utils.get_matching_files(session_dir, 'tif','AnalyticMS')


# # coregister all the tif files for this satellite
results = coregister_class.coregister_files(tif_files, template_path,coregistered_dir, coregister_settings)

results = file_utils.merge_list_of_dicts(results)

# # # after each set of tif files are coregistered, save the results
file_utils.save_coregistered_results(results, WINDOW_SIZE, template_path, result_json_path, settings)



# FILTER ANY OUTLIER SHIFTS
# this creates a csv file in the same directory as the json file that contains how each file should be filtered
results = file_utils.open_json_file(result_json_path)
output_csv_path = result_json_path.replace('.json', '_filtered.csv')
output_csv_path = filters.apply_filtering(results,output_csv_path, filter_settings)
df = pd.read_csv(output_csv_path)

# Note still need to move the planet files that failed the filtering to a new directory in the coregistered directory

# # Move the files that failed the filtering to a new directory in coregistered directory
failed_coregs = list(df[~df['filter_passed']]['filename'])
file_utils.move_failed_files(failed_coregs,coregistered_dir,session_dir,)

# Copy the cloud masks
cloud_masks=file_utils.get_matching_files(session_dir, 'tif', 'udm2')
print(f"cloud_masks: {cloud_masks}")
# Filter out any cloud masks that do not exist in the coregistered directory
ms_files = file_utils.get_matching_files(coregistered_dir, 'tif', 'AnalyticMS')
print(f"ms_files: {ms_files}")

#r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
planet_pattern = r"^\d{8}_\d{6}_\d{4}"
cloud_masks = [cloud_mask for cloud_mask in cloud_masks if file_utils.extract_date_from_filename(cloud_mask,pattern=planet_pattern) in [file_utils.extract_date_from_filename(ms_file,pattern=planet_pattern) for ms_file in ms_files]]
print(f"cloud_masks: {cloud_masks}")

# copy the cloud masks to the coregistered directory
file_utils.copy_filepaths_to_dir(cloud_masks, coregistered_dir)

# Apply the shifts to the udm2 files
geo_utils.apply_shifts_to_tiffs(df,coregistered_dir,session_dir,apply_shifts_filter_passed=True)


jpg_folder = os.path.join(coregistered_dir, 'jpg_files')
planet_to_coastseg.make_planet_jpgs(coregistered_dir,coregistered_dir)