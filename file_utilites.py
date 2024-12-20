import os
import glob
import json
import shutil
import re
import numpy as np
from collections import OrderedDict
from enum import Enum

from coastsat import SDS_preprocess

class Satellite(Enum):
    L5 = 'L5'
    L7 = 'L7'
    L8 = 'L8'
    L9 = 'L9'
    S2 = 'S2'

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

def get_config(config_path,roi_id=None):
    with open(config_path, 'r') as f:
        config = json.load(f)
    if roi_id:
        if roi_id not in config:
            raise ValueError(f"ROI ID {roi_id} not found in config file.")
        config = config[roi_id]
    return config

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
        copy_files_for_satellites(filenames, coregistered_dir, session_dir, satellites,)
    else:
        # Only copy the meta directories to the coregistered directory for the files that passed the filtering
        filenames = df[df['filter_passed']==True]['filename']
        copy_meta_for_satellites(filenames, coregistered_dir, session_dir, satellites)


def save_coregistered_results(results, WINDOW_SIZE, template_path, result_json_path, settings,satellite:str="") -> OrderedDict:
    """
    Process and save coregistration results ensuring 'settings' is the last item in the dictionary.

    Args:
        results (dict): The coregistration results dictionary.
        WINDOW_SIZE (int): The window size setting.
        template_path (str): The template path for coregistration.
        result_json_path (str): Path to save the resulting JSON file.
        satellite (str): The satellite name.
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
        
    if satellite:
        # Merge results for the current satellite
        results[satellite] = merge_list_of_dicts(results[satellite])

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

def create_readme(coregistered_dir, results):
    # create a readme.txt file at the output_dir
    with open(os.path.join(coregistered_dir, 'readme.txt'), 'w') as f:
        # read the json data and count the number of successful coregistrations
        successful_coregistrations = 0
        qc_failed = 0
        improvements = []
        for key, value in results.items():
            if 'settings' in key:
                continue
            if value['success'] == 'True':
                successful_coregistrations += 1
                # create a list of the change in ssim score for each successful coregistration
                # then create an average improvement in ssim score
                improvements.append(value['change_ssim'])
            if value['qc'] == 0:
                qc_failed += 1
            if len(improvements) == 0:
                average_improvement = 0
            elif len(improvements) == 1:
                average_improvement = improvements[0]
            else:
                average_improvement = np.mean(improvements)
            f.write(f"Number of successful coregistrations: {successful_coregistrations}\n")
            f.write(f"Total number of coregistrations: {len(results) - 2}\n")
            f.write(f"Average improvement in SSIM score: {average_improvement}\n")
            f.write(f"Number of QC failed coregistrations: {qc_failed}\n")
            f.write(f"Settings: {results['settings']}\n")


def merge_list_of_dicts(list_of_dicts):
    merged_dict = {}
    for d in list_of_dicts:
        merged_dict.update(d)
    return merged_dict

def save_coregistered_results(results, WINDOW_SIZE, template_path, result_json_path, settings,satellite:str="") -> OrderedDict:
    """
    Process and save coregistration results ensuring 'settings' is the last item in the dictionary.

    Args:
        results (dict): The coregistration results dictionary.
        WINDOW_SIZE (int): The window size setting.
        template_path (str): The template path for coregistration.
        result_json_path (str): Path to save the resulting JSON file.
        satellite (str): The satellite name.
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
        
    if satellite:
        # Merge results for the current satellite
        results[satellite] = merge_list_of_dicts(results[satellite])

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


def find_satellite_in_filename(filename: str) -> str:
    """Use regex to find the satellite name in the filename.
    Satellite name is case-insensitive and can be separated by underscore (_) or period (.)"""
    for satellite in Satellite:
        # Adjusting the regex pattern to consider period (.) as a valid position after the satellite name
        if re.search(fr'(?<=[\b_]){satellite.value}(?=[\b_.]|$)', filename, re.IGNORECASE):
            return satellite.value
    return ""

def get_planet_dict(directory: str, file_type: str,contains=None ) -> dict:
    """

    
    Parameters:
    -----------
    directory : str
        The directory where the files are located.

    file_type : str
        The filetype of the files to be included.
        Ex. 'jpg'

    contains (str, optional): Substring that the filename must contain. Defaults to None.

    Returns:
    --------
    dict

    
    """

    satellites = {"planet": set(),}
    satellites['planet'] = get_matching_files(directory, file_type,contains)
    
    return satellites

def extract_date_from_filename(filename: str,pattern = r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}") -> str:
    """Extracts the first part of the filename that matches the pattern from a filename.

     - The default is that the filename is expected to have the date in format "YYYY-MM-DD-HH-MM-SS".
     - Example 2024-05-28-22-18-07 would be extracted from "2024-05-28-22-18-07_S2_ID_1_datetime11-04-24__04_30_52_ms.tif" for the default pattern

    Args:
        filename (str): The filename to extract the date from.
        pattern (str): The regex pattern to match the date string. Defaults to r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}". This matches "YYYY-MM-DD-HH-MM-SS"

    """
    match = re.match(pattern, filename)
    if match:
        return match.group(0)
    else:
        return ""

def get_matching_files(directory, file_type, contains=None):
    """
    Get files from a directory filtered by file type and an optional substring in the filename.

    Args:
        directory (str): Path to the directory to search for files.
        file_type (str): File extension to filter by (e.g., 'tif').
        contains (str, optional): Substring that the filename must contain.

    Returns:
        list: List of matching file paths.
    """
    # Adjust the pattern to include the 'contains' substring, if provided
    if contains:
        pattern = os.path.join(directory, f"*{contains}*.{file_type}")
    else:
        pattern = os.path.join(directory, f"*.{file_type}")
    
    # Use glob to find files directly matching the pattern
    return glob.glob(pattern)


def merge_list_of_dicts(list_of_dicts):
    merged_dict = {}
    for d in list_of_dicts:
        merged_dict.update(d)
    return merged_dict

def process_failed_coregistrations(failed_coregs, coregistered_dir, unregistered_dir,replace=False,
                                   copy_only=False, move_only=True, subfolder_name='ms'):
    """
    Replaces failed coregistration files with the original unregistered files if replace is trye

    Parameters:
        failed_coregs (dict): A dictionary where keys are satellite names and values are lists of filenames that failed coregistration.
        coregistered_dir (str): The base directory containing coregistered satellite data.
        unregistered_dir (str): The base directory containing the original unregistered files.
        copy_only (bool): If True, files will be copied to the failed directory. Defaults to False.
        move_only (bool): If True, files will be moved to the failed directory. Defaults to True.
        subfolder_name (str): The name of the subfolder containing the files to be moved. Defaults to 'ms'.
    """
    # copy and move cannot be both True
    if (copy_only and move_only) or (not copy_only and not move_only):
        raise ValueError("Exactly one of 'copy_only' or 'move_only' must be True. Both cannot be True or False.")

    # Moves (or copies) the failed coregistration files to a 'failed_coregistration' directory for each satellite
    handle_failed_coregs_per_satellite(failed_coregs, coregistered_dir, copy_only=copy_only, move_only=move_only)

    # Copy the original files to the coregistered directory (replaces the failed coregistrations with the original
    if replace:
        copy_original_to_coregistered_per_satellite(failed_coregs, unregistered_dir, coregistered_dir, subfolder_name)

def open_json_file(json_file):
    # Load JSON data
    with open(json_file, 'r') as json_file:
        results = json.load(json_file)
    return results

def moved_files(failed_coregs, coregistered_dir, copy_only=True, move_only=False):
    """
    Handles failed coregistration files by copying or moving them to specific directories.
    This is specifically designed to work with CoastSat and CoastSeg sessions.

    Parameters:
        failed_coregs (dict): A dictionary where keys are satellite names and values are lists of filenames that failed coregistration.
        coregistered_dir (str): The base directory containing coregistered satellite data.
        copy_only (bool): If True, files will be copied to the failed directory. Defaults to True.
        move_only (bool): If True, files will be moved to the failed directory. Defaults to False.
    """
    for satellite, filenames in failed_coregs.items():
        # Create a directory for the satellite's failed coregistration
        failed_dir = os.path.join(coregistered_dir, 'failed_coregistration')
        os.makedirs(failed_dir, exist_ok=True)

        # Define the satellite directory
        satellite_dir = os.path.join(coregistered_dir, satellite, 'ms')

        # Copy or move the failed coregistration files
        for filename in filenames:
            src = os.path.join(satellite_dir, filename)
            dst = os.path.join(failed_dir, filename)
            if os.path.exists(src):
                if copy_only:
                    shutil.copy(src, dst)
                    print(f"Copied {filename} to {dst}")
                elif move_only:
                    shutil.move(src, dst)
                    print(f"Moved {filename} to {dst}")

def move_failed_files(failed_coregs, coregistered_dir,source_dir:str):
    """
    Handles failed coregistration files by moving them to a 'failed_coregistration' directory within the coregistered directory.
    Parameters:
        failed_coregs (list): A list of filenames that failed coregistration.
        coregistered_dir (str): The base directory containing coregistered satellite data.
        source_dir (str): The directory containing the files to move.
    """
    if isinstance(failed_coregs, str):
        failed_coregs = [failed_coregs]

    failed_dir = os.path.join(coregistered_dir, 'failed_coregistration')
    os.makedirs(failed_dir, exist_ok=True)


    for filename in failed_coregs:
        print(f"filenames: {filename}")
        src = os.path.join(source_dir, filename)
        dst = os.path.join(failed_dir, filename)
        print(f"src: {src}")
        print(f"dst: {dst}")
        if os.path.exists(src):
            if not os.path.exists(dst):
                shutil.move(src, dst)
                print(f"Moved {filename} to {dst}")


def handle_failed_coregs_per_satellite(failed_coregs, coregistered_dir, copy_only=True, move_only=False):
    """
    Handles failed coregistration files by copying or moving them to specific directories.
    This is specifically designed to work with CoastSat and CoastSeg sessions.

    Parameters:
        failed_coregs (dict): A dictionary where keys are satellite names and values are lists of filenames that failed coregistration.
        coregistered_dir (str): The base directory containing coregistered satellite data.
        copy_only (bool): If True, files will be copied to the failed directory. Defaults to True.
        move_only (bool): If True, files will be moved to the failed directory. Defaults to False.
    """
    for satellite, filenames in failed_coregs.items():
        # Create a directory for the satellite's failed coregistration
        failed_dir = os.path.join(coregistered_dir, 'failed_coregistration', satellite)
        os.makedirs(failed_dir, exist_ok=True)

        # Define the satellite directory
        satellite_dir = os.path.join(coregistered_dir, satellite, 'ms')

        # Copy or move the failed coregistration files
        for filename in filenames:
            src = os.path.join(satellite_dir, filename)
            dst = os.path.join(failed_dir, filename)
            if os.path.exists(src):
                if copy_only:
                    shutil.copy(src, dst)
                    print(f"Copied {filename} to {dst}")
                elif move_only:
                    shutil.move(src, dst)
                    print(f"Moved {filename} to {dst}")

def copy_original_to_coregistered_per_satellite(failed_coregs, unregistered_dir, coregistered_dir,subfolder_name = 'ms'):
    """
    Copies the original files from the unregistered directory to the coregistered directory
    for the specified satellite.

    Parameters:
        failed_coregs (dict): A dictionary where keys are satellite names and values are lists of filenames that failed coregistration.
        unregistered_dir (str): The base directory containing the original unregistered files.
        coregistered_dir (str): The base directory containing coregistered satellite data.
        subfolder_name (str): The name of the subfolder containing the files to be moved. Defaults to 'ms'.
    """
    for satellite, filenames in failed_coregs.items():
        # Define the satellite directory
        satellite_dir = os.path.join(coregistered_dir, satellite, subfolder_name)
        os.makedirs(satellite_dir, exist_ok=True)

        # Copy the original files to the coregistered directory
        for filename in filenames:
            src = os.path.join(unregistered_dir, satellite, subfolder_name, filename)
            dst = os.path.join(satellite_dir, filename)
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"Copied {filename} from {src} to {dst}")
            else:
                print(f"Source file not found: {src}")



# def move_failed_coregs(coreg_dir, ms_dir, failed_coregs):
#     """
#     Moves files from ms_dir to a 'failed_coregs' directory within coreg_dir if they exist.

#     Parameters:
#         coreg_dir (str): The directory where 'failed_coregs' folder will be created.
#         ms_dir (str): The source directory containing the files to move.
#         failed_coregs (list): A list of filenames to move to 'failed_coregs'.

#     Returns:
#         None
#     """
#     # Create 'failed_coregs' directory if it doesn't exist
#     failed_coreg_dir = os.path.join(coreg_dir, 'failed_coregs')
#     os.makedirs(failed_coreg_dir, exist_ok=True)

#     # Move the files
#     for filename in failed_coregs:
#         # Get the full path of the source file
#         ms_file = os.path.join(ms_dir, filename)
#         # Move the file if it exists and isn't already in the target directory
#         if os.path.exists(ms_file) and not os.path.exists(os.path.join(failed_coreg_dir, filename)):
#             shutil.move(ms_file, failed_coreg_dir)
#             print(f"Moved {filename} to {failed_coreg_dir}")

def copy_files_if_not_exists(source_dir, destination_dir):
    """
    Copies files from the source directory to the destination directory
    if they do not already exist in the destination.

    Parameters:
        source_dir (str): Path to the source directory.
        destination_dir (str): Path to the destination directory.
    """
    for filename in os.listdir(source_dir):
        src = os.path.join(source_dir, filename)
        dst = os.path.join(destination_dir, filename)
        if os.path.exists(dst):
            continue
        if os.path.exists(src) and os.path.isfile(src):
            shutil.copy(src, dst)
            print(f"Copied {filename} to {dst}")

def read_json_file(json_file, raise_error=True):
    if not os.path.exists(json_file):
        if raise_error:
            raise FileNotFoundError(f"File not found: {json_file}")
        else:
            return None
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def get_valid_roi_id(config: dict, roi_id:str = None) -> str:
    """
    Get a valid ROI ID from the config dictionary.

    Args:
        config (dict): Configuration dictionary containing 'roi_ids'.
        roi_id (str, optional): A specific ROI ID to use. Defaults to None.

    Returns:
        str: A valid ROI ID.

    Raises:
        SystemExit: If no valid ROI ID is found in the config dictionary.
    """
    if roi_id is not None:
        if roi_id in config.keys():
            return roi_id
        else:
            print(f"ROI ID {roi_id} not found in the config file.")
            raise KeyError(f"ROI ID {roi_id} not found in the config file.")
    # otherwise get the first roi_id
    roi_id = config['roi_ids'][0]
    if roi_id not in config.keys():
        for roi_id in config['roi_ids']:
            if roi_id in config.keys():
                break
        
        if roi_id not in config.keys():
            print("No ROI ID found in the config file.")
            raise KeyError("No ROI ID found in the config file.")

    return roi_id

def get_satellites(config_file,roi_id:str=None):
    config = read_json_file(config_file)
    roi_id = get_valid_roi_id(config, roi_id)
    satellites = config[roi_id]['sat_list']
    return satellites

def create_coregistered_directory(session_dir,satellites:list[str]):
    """
    Creates a coregistered directory structure within the specified session directory.
    
    The function creates a top-level 'coregistered' directory and subdirectories for 
    each satellite specified in the satellites list. It also creates specific subdirectories 
    based on the satellite type.

    Parameters:
    session_dir (str): The path to the session directory where the coregistered directory will be created.
    satellites (list of str): A list of satellite names for which subdirectories will be created. 
                              Recognized satellite names are 'S2', 'L7', 'L8', 'L9', and 'planet'.
    Returns:
    str: The path to the top level coregistered directory.
    """
    # make the top level coregistered directory
    coregistered_directory = os.path.join(session_dir, 'coregistered')
    os.makedirs(coregistered_directory, exist_ok=True)
    # make a subdirectory to contain the jpg_files
    os.makedirs(os.path.join(coregistered_directory, 'jpg_files','preprocessed'),exist_ok=True)
    if not satellites:
        return
    # make a subdirectory for each satellite
    for satname in satellites:
        ms_dir = os.path.join(coregistered_directory, satname, 'ms')
        mask_dir = os.path.join(coregistered_directory, satname, 'mask')
        meta_dir = os.path.join(coregistered_directory, satname, 'meta')
        os.makedirs(ms_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        if satname == 'S2':
            swir_dir = os.path.join(coregistered_directory, satname, 'swir')
            os.makedirs(swir_dir, exist_ok=True)
        elif satname in ['L7','L8','L9']:
            pan_dir = os.path.join(coregistered_directory, satname, 'pan')
            os.makedirs(pan_dir, exist_ok=True)
        elif satname.lower() == 'planet':
            os.makedirs(os.path.join(coregistered_directory, satname, 'mask'), exist_ok=True)
            os.makedirs(os.path.join(coregistered_directory, satname, 'nir'), exist_ok=True)
        else:
            print(f"Satellite {satname} not recognized.")
            # remove the ms directory
            os.rmdir(ms_dir)
            os.rmdir(mask_dir)
            return 
    return coregistered_directory

def copy_files(filenames, folder_name: str, src_dir: str, dst_dir: str, satname: str):
    """
    Copy files from the source directory to the destination directory based on the DataFrame.

    Destination directory structure:
    dst_dir / satname / folder_name / filename

    Example 
    copy_files(['2023-12-09-18-40-32_L9_ID_ice1_datetime06-06-24__09_02_37_ms.tif'], 'mask', mask_dir, coreg_dir, 'L9')

    Args:
        filenames(list[str]): filenames to copy. filenames contain the filename 'ms' which is replaced by the folder_name
        folder_name (str): The folder name to replace in the filename (e.g., 'mask').
        src_dir (str): Directory path where the original files are located.
        dst_dir (str): Directory path where the files should be copied.
        satname (str): Satellite name to be used in the directory path.
    """
    for file in filenames:
        filename = file.replace('ms', folder_name)
        # Get the path for this file
        file_path = os.path.join(src_dir, filename)

        if os.path.exists(file_path):
            # Copy the file to the destination directory
            shutil.copy(file_path, os.path.join(dst_dir, satname, folder_name, filename))
            # print(f"Copied {filename} to {os.path.join(dst_dir, satname, folder_name)}")

def copy_filenames_to_dir(filenames, src_dir, dst_dir):
    """
    Copy files from the source directory to the destination directory based on the DataFrame.

    Args:
        filenames(list[str]): filenames to copy.
        src_dir (str): Directory path where the original files are located.
        dst_dir (str): Directory path where the files should be copied.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for file in filenames:
        # Get the path for this file
        file_path = os.path.join(src_dir, file)

        if os.path.exists(file_path):
            # Copy the file to the destination directory
            shutil.copy(file_path, os.path.join(dst_dir, file))

def copy_filepaths_to_dir(filepaths, dst_dir):
    """
    Copy filepaths  to the destination directory.

    Args:
        filepaths(list[str]): filepaths to copy.
        src_dir (str): Directory path where the original files are located.
        dst_dir (str): Directory path where the files should be copied.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for file_path in filepaths:
        file = os.path.basename(file_path)

        if os.path.exists(file_path):
            # Copy the file to the destination directory
            shutil.copy(file_path, os.path.join(dst_dir, file))


def copy_files_for_satellites(filenames:list[str],coreg_dir,unregistered_dir,satellites:list[str]):
    """
    Copies files for different satellite types into their respective directories.

    Parameters:

    filenames (list[str]): List of filenames to be copied.
    coreg_dir (str): Directory where coregistered files are stored.
    unregistered_dir (str): Directory where unregistered files are stored.
    satellites (list[str]): List of satellite names.

    The function creates subdirectories for each satellite in the unregistered directory.
    It copies files into 'mask' and 'meta' directories for all satellites.

    For 'S2' satellites, it also copies files into 'swir' directory.
    For 'L7', 'L8', and 'L9' satellites, it copies files into 'pan' directory.
    If the satellite is 'Planet', it prints a message indicating that Planet files are not yet supported.
    """
    # make a subdirectory for each satellite
    for satname in satellites:
        # all satellites have ms,mask and meta directories
        mask_dir = os.path.join(unregistered_dir, satname, 'mask')
        copy_files(filenames, 'mask', mask_dir, coreg_dir, satname)
        meta_dir = os.path.join(unregistered_dir, satname, 'meta')
        # example metadata file: 2022-05-17-22-08-11_L9_ID_1_datetime11-04-24__04_30_52.txt
        # example ms file : 2022-04-01-21-56-37_L9_ID_1_datetime11-04-24__04_30_52_ms.tif
        meta_filenames = [f.replace('_ms.tif', '.txt') for f in filenames]
        copy_filenames_to_dir(meta_filenames,meta_dir,os.path.join(coreg_dir, satname,'meta'))
        if satname == 'S2':
            swir_dir = os.path.join(unregistered_dir, satname, 'swir')
            copy_files(filenames, 'mask', mask_dir, coreg_dir, satname)
            copy_files(filenames, 'swir', swir_dir, coreg_dir, satname)
        elif satname in ['L7','L8','L9']:
            pan_dir = os.path.join(unregistered_dir, satname, 'pan')
            copy_files(filenames, 'pan', pan_dir, coreg_dir, satname)
        elif satname.lower() == 'planet':
            print("Planet files not yet supported.")

def copy_meta_for_satellites(filenames:list[str],coreg_dir,unregistered_dir,satellites:list[str]):
    """
    Copies meta files for different satellite types into their respective directories.

    Parameters:

    filenames (list[str]): List of filenames to be copied.
    coreg_dir (str): Directory where coregistered files are stored.
    unregistered_dir (str): Directory where unregistered files are stored.
    satellites (list[str]): List of satellite names.

    If the satellite is 'Planet', it prints a message indicating that Planet files are not yet supported.
    """
    # make a subdirectory for each satellite
    for satname in satellites:
        meta_dir = os.path.join(unregistered_dir, satname, 'meta')
        if not os.path.exists(meta_dir):
            continue
        # example metadata file: 2022-05-17-22-08-11_L9_ID_1_datetime11-04-24__04_30_52.txt
        # example ms file : 2022-04-01-21-56-37_L9_ID_1_datetime11-04-24__04_30_52_ms.tif
        meta_filenames = [f.replace('_ms.tif', '.txt') for f in filenames]
        copy_filenames_to_dir(meta_filenames,meta_dir,os.path.join(coreg_dir, satname,'meta'))


def get_filepaths_to_folders(inputs, satname,coregistered_name:str=None):
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


def create_coregistered_jpgs(inputs, settings: dict):
    """
    Creates coregistered JPEG images from multispectral (ms) TIFF file for each satellite in the given directory.

    Args:
        inputs (dict): A dictionary containing the following keys:
            - "satname" (list): List of satellite names.
            - "filepath" (str): Base file path where the data is stored.  (e.g., r"C:\\Users\\user\\CoastSeg\\data")
            - "sitename" (str): Name of the site. (Make sure to put 'coregistered' in the path to indicate that these files should be saved in the coregistered directory.)
        settings (dict): A dictionary containing the following settings:
            - "cloud_threshold" (float, optional): Threshold for cloud detection. Defaults to 0.99.
            - "cloud_mask_issue" (bool, optional): Flag indicating if there is an issue with the cloud mask. Defaults to False.
            - "apply_cloud_mask" (bool, optional): Flag indicating whether to apply the cloud mask. Defaults to True.

    Returns:
        None

    Example Inputs:
    """
    # Get the settings
    cloud_threshold = settings.get('cloud_threshold', 0.99)
    cloud_mask_issue = settings.get('cloud_mask_issue', False)
    apply_cloud_mask = settings.get('apply_cloud_mask', True)

    for satname in inputs["satname"]:
        # This gets the path to the ms, mask and swir/pan folders for each satellite
        tif_paths = get_filepaths_to_folders(inputs, satname)
        # this is the directory of the ms files which were coregistered
        ms_dir = os.path.join(inputs["filepath"], inputs["sitename"], satname, "ms")

        for filename in os.listdir(ms_dir):
            if filename.endswith("_ms.tif"):
                # make sure the tif file exists
                if not os.path.exists(os.path.join(ms_dir, filename)):
                    print(f"File not found: {filename}")
                    continue
                
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