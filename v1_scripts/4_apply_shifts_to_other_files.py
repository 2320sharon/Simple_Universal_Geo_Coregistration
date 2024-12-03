import pandas as pd
import numpy as np
import os
import shutil
import rasterio
from coregister_class import apply_shift_to_tiff

def copy_files(df: pd.DataFrame, folder_name: str, src_dir: str, dst_dir: str, satname: str):
    """
    Copy files from the source directory to the destination directory based on the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'filename' column.
        folder_name (str): The folder name to replace in the filename (e.g., 'mask').
        src_dir (str): Directory path where the original files are located.
        dst_dir (str): Directory path where the files should be copied.
        satname (str): Satellite name to be used in the directory path.
    """
    for file in df['filename']:
        filename = file.replace('ms', folder_name)
        # Get the path for this file
        file_path = os.path.join(src_dir, filename)

        if os.path.exists(file_path):
            # Copy the file to the destination directory
            shutil.copy(file_path, os.path.join(dst_dir, satname, folder_name, filename))
            print(f"Copied {filename} to {os.path.join(dst_dir, satname, folder_name)}")


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
            print(f"Applied shift to {os.path.basename(dst_path)}")


satname  = 'S2'

base_dir = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52"

# coregistered ms directory
coreg_dir = os.path.join(base_dir, 'coregistered')
ms_dir = os.path.join(coreg_dir, satname, 'ms')

# unregistered ms_directory
ms_dir = os.path.join(base_dir, satname,'ms' )
# if s2 then get the swir directory
if satname == 'S2':
    swir_dir = os.path.join(base_dir, satname, 'swir')

mask_dir = os.path.join(base_dir, satname, 'mask')

os.makedirs(os.path.join(coreg_dir, satname, 'mask'), exist_ok=True)
os.makedirs(os.path.join(coreg_dir, satname, 'ms'), exist_ok=True)
os.makedirs(os.path.join(coreg_dir, satname, 'swir'), exist_ok=True)
os.makedirs(os.path.join(coreg_dir, satname, 'meta'), exist_ok=True)


# read the filenames from the filtered csv whose filter_passed is True
csv_path = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52\coregistered\S2\ms\transformation_results_filtered.csv"
df = pd.read_csv(csv_path)


# get the pan/swir & mask path for each ms file
copy_files(df, 'mask', mask_dir, coreg_dir, satname)
copy_files(df, 'swir', swir_dir, coreg_dir, satname)
# copy the entire meta directory to the coregistered directory
shutil.copytree(os.path.join(base_dir, satname, 'meta'), os.path.join(coreg_dir, satname, 'meta'),dirs_exist_ok=True)

# apply the shifts to mask, pan/swir files ( check the resolution of the files)
# pan resolution :
# L7: 30m
# L8: 30m
# L9: 30m
# swir resolution (s2)
# S2: 10m
# mask resolution (s2)
# S2: 10m

# get the filenames that kept (filter_passed == True)
passed_coregs = df[df['filter_passed']]['filename']
# get the pan/swir & mask path for each ms file
apply_shifts_to_files(df, passed_coregs, mask_dir, coreg_dir, satname, 'mask')
apply_shifts_to_files(df, passed_coregs, swir_dir, coreg_dir, satname, 'swir')


# Save these images to the coregistered directory under the respective folders