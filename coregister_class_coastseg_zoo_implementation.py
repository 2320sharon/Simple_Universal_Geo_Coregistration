# Standard library imports
import os
import json
import glob
import re
import shutil
import threading
from collections import OrderedDict
from enum import Enum

# Third-party library imports
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation
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

class Satellite(Enum):
    L5 = 'L5'
    L7 = 'L7'
    L8 = 'L8'
    L9 = 'L9'
    S2 = 'S2'


def extract_date_from_filename(filename: str) -> str:
    """Extracts the first instance date string "YYYY-MM-DD-HH-MM-SS" from a filename.
     - The date string is expected to be in the format "YYYY-MM-DD-HH-MM-SS".
     - Example 2024-05-28-22-18-07 would be extracted from "2024-05-28-22-18-07_S2_ID_1_datetime11-04-24__04_30_52_ms.tif"
    """
    pattern = r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    match = re.match(pattern, filename)
    if match:
        return match.group(0)
    else:
        return ""

def find_satellite_in_filename(filename: str) -> str:
    """Use regex to find the satellite name in the filename.
    Satellite name is case-insensitive and can be separated by underscore (_) or period (.)"""
    for satellite in Satellite:
        # Adjusting the regex pattern to consider period (.) as a valid position after the satellite name
        if re.search(fr'(?<=[\b_]){satellite.value}(?=[\b_.]|$)', filename, re.IGNORECASE):
            return satellite.value
    return ""

def get_filtered_dates_dict(directory: str, file_type: str, ) -> dict:
    """
    Scans the directory for files with the given file_type and extracts the date from the filename and returns a dictionary with the satellite name as the key and a set of dates as the value.


    Parameters:
    -----------
    directory : str
        The directory where the files are located.

    file_type : str
        The filetype of the files to be included.
        Ex. 'jpg'


    Returns:
    --------
    dict
        a dictionary where each key is a satellite name and each value is a set of the dates in the format "YYYY-MM-DD-HH-MM-SS" that represents the time the scene was captured.
    
    Example:
        {
        "L5":{'2014-12-19-18-22-40',},
        "L7":{},
        "L8":{'2014-12-19-18-22-40',},
        "L9":{},
        "S2":{},
    }
    
    """
    filepaths = glob.iglob(os.path.join(directory, f"*.{file_type}"))

    satellites = {"L5": set(), "L7": set(), "L8": set(), "L9": set(), "S2": set()}
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        date = extract_date_from_filename(filename)
        if not date:
            continue

        satname = find_satellite_in_filename(filename)
        if not satname:
            continue
        
        if satname in satellites:
            satellites[satname].add(date)

    return satellites

def save_coregistered_results(results, satellite, WINDOW_SIZE, template_path, result_json_path, settings):
    """
    Process and save coregistration results ensuring 'settings' is the last item in the dictionary.

    Args:
        results (dict): The coregistration results dictionary.
        satellite (str): The satellite name.
        WINDOW_SIZE (int): The window size setting.
        template_path (str): The template path for coregistration.
        result_json_path (str): Path to save the resulting JSON file.
        settings (dict): Additional settings to include in the results.

    Returns:
        OrderedDict: The processed results dictionary with 'settings' as the last item.
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    # Merge results for the current satellite
    print(f"results: {results}")
    results[satellite] = plotting.merge_list_of_dicts(results[satellite])
    print(f"results: {results}")

    # Update settings and add to results
    settings.update({'window_size': WINDOW_SIZE, 'template_path': template_path})
    results['settings'] = settings

    # Ensure 'settings' is the last key
    results_ordered = OrderedDict(results)
    results_ordered.move_to_end('settings')

    # Save to JSON file
    with open(result_json_path, 'w') as json_file:
        json.dump(results_ordered, json_file, indent=4,cls=NumpyEncoder)

    print(f"Saved results to: {result_json_path}")

    return results_ordered

def apply_shifts_to_tiffs(df,coregistered_dir,session_dir,satellites,apply_shifts_filter_passed=True):
    """
    Applies shifts to TIFF files based on the provided DataFrame and copies unregistered files to the coregistered directory.
    Parameters:
    df (pandas.DataFrame): DataFrame containing information about the files, including whether they passed filtering.
    coregistered_dir (str): Directory where the coregistered files will be stored.
    session_dir (str): Directory of the current session containing the original files.
    satellites (list): List of satellite names to process.
    apply_shifts_filter_passed (bool): If True, apply the shifts to only the files that passed the filtering. If False, apply the shifts to all files.
    Returns:
    None
    """    
    if  apply_shifts_filter_passed:
        # Apply the shifts to the other files if they passed the filtering
        filenames = df[df['filter_passed']==True]['filename']
    else: # get all the filenames whether they passed the filtering or not
        filenames = df['filename']

    geo_utils.apply_shifts_for_satellites(df,filenames,coregistered_dir,session_dir,satellites)

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
    results = []
    
    if tif_files == []:
        print(f"No files found for {satellite} in {ms_dir}")
        return results
    
    for target_path in tqdm.tqdm(tif_files,desc=f'Detecting shifts for {satellite} files:'):
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
WINDOW_SIZE=(256,256)
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
ROI_ID = ""

session_dir = r'C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52_original_mess_with'
template_path = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52_original\L9\ms\2023-06-30-22-01-55_L9_ID_1_datetime11-04-24__04_30_52_ms.tif"
sorted_jpg_path = r'C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52_original_mess_with\jpg_files\preprocessed\RGB'

# CoastSeg Session
# 1. read config.json file to get the satellites
config_path = os.path.join(session_dir, 'config.json')
config = file_utils.read_json_file(config_path)
# Allow the user to select a specific ROI or just use the first one
# Enter a specific ROI ID or or just use the first one by entering roi_id = None
satellites = file_utils.get_satellites(config_path,roi_id=None)
# get the first ROI ID or replace this one
roi_id = config['roi_ids'][0] if not ROI_ID else ROI_ID


# remove L7 since we can't coregister it
if 'L7' in satellites:
    satellites.remove('L7')
print(f"Satellites: {satellites}")


# make a new coregistered directory to save everything in ( not a problem is it already exists)
coregistered_dir= file_utils.create_coregistered_directory(session_dir,satellites)
result_json_path = os.path.join(coregistered_dir, 'transformation_results.json')

# copy the config_gdf.geojson files to the coregistered directory
shutil.copy(os.path.join(session_dir, 'config_gdf.geojson'), os.path.join(coregistered_dir, 'config_gdf.geojson'))
print(f"Coregisted directory: {coregistered_dir}")

# get the filtered jpgs
# returns dictionary of satname : [date, date, date]
filtered_dates_by_sat = get_filtered_dates_dict(sorted_jpg_path, 'jpg')
# drop L7 since we can't coregister it
if 'L7' in filtered_dates_by_sat:
    filtered_dates_by_sat.pop('L7')

# list the directories in the session directory
# 2. loop through the directories
results = {}
for satellite in tqdm.tqdm(filtered_dates_by_sat.keys(),desc='Satellites'):
    print(f"Processing {satellite}")
    # 1. Get the satellite directory and its multispectral directory (ms)
    satellite_dir = os.path.join(session_dir,satellite)
    ms_dir = os.path.join(satellite_dir,'ms')
    # only read the ms files that match the dates in 
    tif_files = glob.glob(os.path.join(ms_dir, '*.tif')) 
    # convert each tif to a date and if its not in the filtered dates, remove it
    # example ms file : 2021-05-15-22-02-03_L8_ID_1_datetime11-04-24__04_30_52_ms.tif
    tif_filenames = [tif for tif in tif_files if extract_date_from_filename(os.path.basename(tif)) in filtered_dates_by_sat[satellite]]
    # create full paths again for the tif files
    tif_files = [os.path.join(ms_dir, tif) for tif in tif_filenames]
    
    output_dir = os.path.join(coregistered_dir,satellite,'ms') # directory to save coregistered files to
    # coregister all the tif files for this satellite
    results[satellite] = coregister_files(tif_files, template_path, output_dir, coregister_settings)
    # after each set of tif files are coregistered, save the results
    save_coregistered_results(results, satellite, WINDOW_SIZE, template_path, result_json_path, settings)

# this creates a csv file in the same directory as the json file that contains how each file should be filtered
coregister_settings.update(**filter_settings)
new_config_path = save_coregistered_config(config_path,coregistered_dir,coregister_settings)

# FILTER ANY OUTLIER SHIFTS
# this creates a csv file in the same directory as the json file that contains how each file should be filtered
output_csv_path = filters.apply_filtering(result_json_path, filter_settings)
df = pd.read_csv(output_csv_path)

# Move the files that failed the filtering to a new directory in coregistered directory
failed_coregs = df[~df['filter_passed']].groupby('satellite')['filename'].apply(list)
file_utils.process_failed_coregistrations(failed_coregs, coregistered_dir, session_dir,replace=replace_failed_files, copy_only=False, move_only=True, subfolder_name='ms')

# Copy remaining files (swir,pan,mask,meta) to the coregistered directory. If replace replace_failed_files = true copy the unregistered versions of these files
copy_remaining_tiffs(df,coregistered_dir,session_dir,satellites,replace_failed_files=replace_failed_files)

# Copy the files (meta, swir, pan ) that passed coregistration and apply the shifts to them
apply_shifts_to_tiffs(df,coregistered_dir,session_dir,satellites,apply_shifts_filter_passed=True)

# Create jpgs for the files which passed the filtering and copy the jpgs from the files that failed the filtering
# make sure to allow users to turn off the copying of the origianl files just in case they don't want them

# settings # read these from the settings section of config.json
inputs  = get_config(new_config_path,roi_id) # this gets the setting for this ROI ID
config = get_config(new_config_path)
# rename sat_list to satname to make it work with create_coregistered_jpgs
inputs['satname'] = inputs.pop('sat_list')
file_utils.create_coregistered_jpgs(inputs, settings = config['settings'])
