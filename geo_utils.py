import pandas as pd
import numpy as np
import os
import shutil
import rasterio
from coregister_class import apply_shift_to_tiff

def apply_shifts_to_tiffs(df,coregistered_dir,session_dir,satellites:list=None,apply_shifts_filter_passed=True):
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
    if apply_shifts_filter_passed:
        # Apply the shifts to the other files if they passed the filtering
        filenames = df[df['filter_passed']==True]['filename']
    else: # get all the filenames whether they passed the filtering or not
        filenames = df['filename']

    if satellites:
        apply_shifts_for_satellites(df,filenames,coregistered_dir,session_dir,satellites)
    else:
        apply_shifts_to_files_planet(df,filenames,coregistered_dir,session_dir)


def apply_shifts_to_satellite_files(df: pd.DataFrame, valid_files: list, src_dir: str, dst_dir: str, satname: str,folder_name:str,verbose=False):
    """
    Apply shifts to the specified files based on the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'filename', 'shift_x', and 'shift_y' columns.
        valid_files (list): List of filenames to which the shifts should be applied. (not the full path)
        src_dir (str): Directory path where the original files are located.
        dst_dir (str): Directory path where the coregistered files should be saved.
        satname (str): Satellite name to be used in the directory path.
        folder_name (str): Folder name to be used in the directory path.
            Example: If you want to apply shifts to the 'mask' directory this should be 'mask'.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    # Get the pan/swir & mask path for each ms file
    for file in valid_files:
        # Apply the shifts to the files
        # 1. Get the shift from the DataFrame
        shift_x = df[df['filename'] == file]['shift_x'].values[0]
        shift_y = df[df['filename'] == file]['shift_y'].values[0]

        # 2. Apply the shift to the file
        src_path = os.path.join(src_dir, file.replace('ms', folder_name))
        dst_path = os.path.join(dst_dir, satname, folder_name, file.replace('ms', folder_name))
        if os.path.exists(src_path):
            apply_shift_to_tiff(src_path, dst_path, (shift_y, shift_x), verbose=verbose)
            # print(f"Applied shift to {os.path.basename(dst_path)}")

def apply_shifts_to_files_planet(df: pd.DataFrame, valid_files: list, src_dir: str, dst_dir: str,verbose=False):
    """
    Apply shifts to the specified files based on the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'filename', 'shift_x', and 'shift_y' columns.
        valid_files (list): List of filenames to which the shifts should be applied. (not the full path)
        src_dir (str): Directory path where the original files are located.
        dst_dir (str): Directory path where the coregistered files should be saved.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    # Get the pan/swir & mask path for each ms file
    for file in valid_files:
        # Apply the shifts to the files
        # 1. Get the shift from the DataFrame
        shift_x = df[df['filename'] == file]['shift_x'].values[0]
        shift_y = df[df['filename'] == file]['shift_y'].values[0]

        # udm file : 20200603_203636_82_1068_3B_udm2_clip.tif
        # ms file  : 20200603_203636_82_1068_3B_AnalyticMS_toar_clip.tif

        # 2. Apply the shift to the file
        src_path = os.path.join(src_dir, file.replace('AnalyticMS_toar_clip', 'udm2_clip'))
        dst_path = os.path.join(dst_dir, file.replace('AnalyticMS_toar_clip', 'udm2_clip'))
        if os.path.exists(src_path):
            apply_shift_to_tiff(src_path, dst_path, (shift_y, shift_x), verbose=verbose)


def apply_shifts_for_satellites(df,passed_coregs,coreg_dir,unregistered_dir,satellites:list[str],verbose:bool=False):
    # make a subdirectory for each satellite
    for satname in satellites:
        mask_dir = os.path.join(unregistered_dir, satname, 'mask')
        apply_shifts_to_satellite_files(df, passed_coregs, mask_dir, coreg_dir, satname, 'mask',verbose=verbose)
        if satname == 'S2':
            swir_dir = os.path.join(unregistered_dir, satname, 'swir')
            apply_shifts_to_satellite_files(df, passed_coregs, swir_dir, coreg_dir, satname, 'swir',verbose=verbose)
        elif satname in ['L7','L8','L9']:
            pan_dir = os.path.join(unregistered_dir, satname, 'pan')
            apply_shifts_to_satellite_files(df, passed_coregs, pan_dir, coreg_dir, satname, 'pan',verbose=verbose)