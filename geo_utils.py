import pandas as pd
import numpy as np
import os
import shutil
import rasterio
from coregister_class import apply_shift_to_tiff


def apply_shifts_to_files(df: pd.DataFrame, valid_files: list, src_dir: str, dst_dir: str, satname: str,folder_name:str,verbose=False):
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

def apply_shifts_for_satellites(df,passed_coregs,coreg_dir,unregistered_dir,satellites:list[str],verbose:bool=False):
    # make a subdirectory for each satellite
    for satname in satellites:
        mask_dir = os.path.join(unregistered_dir, satname, 'mask')
        apply_shifts_to_files(df, passed_coregs, mask_dir, coreg_dir, satname, 'mask',verbose=verbose)
        if satname == 'S2':
            swir_dir = os.path.join(unregistered_dir, satname, 'swir')
            apply_shifts_to_files(df, passed_coregs, swir_dir, coreg_dir, satname, 'swir',verbose=verbose)
        elif satname in ['L7','L8','L9']:
            pan_dir = os.path.join(unregistered_dir, satname, 'pan')
            apply_shifts_to_files(df, passed_coregs, pan_dir, coreg_dir, satname, 'pan',verbose=verbose)
        elif satname.lower() == 'planet':
            print("Planet files not yet supported.")