# Standard library imports
import os
import json
import glob
import re
import shutil
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

lock = threading.Lock()

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




def copy_remaining_tiffs(df,coregistered_dir,session_dir,satellites,replace_failed_files=False):
    """
    Applies shifts to TIFF files based on the provided DataFrame and copies unregistered files to the coregistered directory.
    Parameters:
    df (pandas.DataFrame): DataFrame containing information about the files, including whether they passed filtering.
    coregistered_dir (str): Directory where the coregistered files will be stored.
    session_dir (str): Directory of the current session containing the original files.
    satellites (list): List of satellite names to process.
    replace_failed_files (bool): Whether to replace failed coregistrations with the original unregistered files.
    Returns:
    None
    """    
    # this means that all the files should be copied over to the coregistered file whether the coregistration passed or not
    if replace_failed_files:
        # Copy the remaining unregistered files for the swir, mask, meta and pan directories to the coregistered directory
        filenames = df['filename']  # this copies all files regardless of whether they passed the filtering
        file_utils.copy_files_for_satellites(filenames, coregistered_dir, session_dir, satellites,)
    else:
        # Only copy the meta directories to the coregistered directory for the files that passed the filtering
        filenames = df[df['filter_passed']==True]['filename']
        file_utils.copy_meta_for_satellites(filenames, coregistered_dir, session_dir, satellites)






def coregister_files(tif_files, template_path, coregistered_dir, coregister_settings):
    """
    Coregisters a list of .tif files to a given template and saves the results in a specified directory.
    Args:
        tif_files (list of str): List of file paths to the .tif files to be coregistered.
        template_path (str): File path to the template .tif file.
        coregistered_dir (str): Directory where the coregistered files will be saved.
        coregister_settings (dict): Dictionary of settings to be passed to the coregistration function.
    Returns:
        list: A list of results from the coregistration process for each file.
    """
    results = []
    
    if tif_files == []:
        print(f"No files porvided to coregister")
        return results
    
    for target_path in tqdm.tqdm(tif_files,desc=f'Coregistering files:'):
        output_path = os.path.join(coregistered_dir, os.path.basename(target_path))
        result = coregister_class.coregister_single(target_path, template_path, output_path, **coregister_settings)
        results.append(result)

    return results

def get_config(config_path,roi_id=None):
    with open(config_path, 'r') as f:
        config = json.load(f)
    if roi_id:
        if roi_id not in config:
            raise ValueError(f"ROI ID {roi_id} not found in config file.")
        config = config[roi_id]
    return config


def save_coregistered_config(config_path,output_dir,settings:dict):
    #open the config.json file, modify it to save the coregistered directory as the new sitename and add the coregistered settings
    with open(config_path, 'r') as f:
        config = json.load(f)

    # update the sitename for each ROI ID
    roi_ids = config.get('roi_ids', [])
    for roi_id in roi_ids:
        inputs = config[roi_id]
        inputs.update({'sitename': config[roi_id]['sitename'] + os.path.sep + 'coregistered'})
    config.update({'coregistered_settings': settings})

    new_config_path = os.path.join(output_dir, 'config.json')
    # write the config to the coregistered directory
    with open(new_config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return new_config_path

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
# results = coregister_files(tif_files, template_path,coregistered_dir, coregister_settings)

# results = file_utils.merge_list_of_dicts(results)

# # # after each set of tif files are coregistered, save the results
# save_coregistered_results(results, WINDOW_SIZE, template_path, result_json_path, settings)



# FILTER ANY OUTLIER SHIFTS
# this creates a csv file in the same directory as the json file that contains how each file should be filtered
results = file_utils.open_json_file(result_json_path)
output_csv_path = result_json_path.replace('.json', '_filtered.csv')
output_csv_path = filters.apply_filtering(results,output_csv_path, filter_settings)
df = pd.read_csv(output_csv_path)

# Note still need to move the planet files that failed the filtering to a new directory in the coregistered directory

# # Move the files that failed the filtering to a new directory in coregistered directory
failed_coregs = df[~df['filter_passed']].groupby('satellite')['filename'].apply(list)


# file_utils.process_failed_coregistrations(failed_coregs, coregistered_dir, session_dir,replace=replace_failed_files, copy_only=False, move_only=True, subfolder_name='ms')

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

# # Copy the files (meta, swir, pan ) that passed coregistration and apply the shifts to them
geo_utils.shift_files(df,coregistered_dir,session_dir,apply_shifts_filter_passed=True)

import planet_to_coastseg

jpg_folder = os.path.join(coregistered_dir, 'jpg_files')
planet_to_coastseg.make_planet_jpgs(coregistered_dir,coregistered_dir)