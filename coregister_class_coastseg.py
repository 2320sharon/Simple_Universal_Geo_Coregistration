import rasterio
import numpy as np
import os
import math
import json
import rasterio
import glob



from skimage.metrics import structural_similarity as ssim
from skimage import exposure
import tqdm
from skimage.registration import phase_cross_correlation
import threading
from scipy.ndimage import shift as scipy_shift
from rasterio.warp import reproject, Resampling  # Import directly from rasterio.warp
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering
import matplotlib.pyplot as plt

lock = threading.Lock()


from coregister_class import CoregisterInterface
import plotting


# Settings
# WINDOW_SIZE=(100,100)
WINDOW_SIZE=(256,256)
window_size_str= f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}"

session_dir = r'C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52'
template_path = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52\L9\ms\2023-06-30-22-01-55_L9_ID_1_datetime11-04-24__04_30_52_ms.tif"


# CoastSeg Session
# 1. read config.json file
# config_path = os.path.join(session_dir, 'config.json')
# with open(config_path, 'r') as f:
#     config = json.load(f)

# get the dirs of satellites
satellites = ['S2','L7', 'L8','L9']
satellites = ['S2']

# from os import walk,scandir
# def fast_scandir(dirname):
#     subfolders= [f.path for f in scandir(dirname) if f.is_dir()]
#     for dirname in list(subfolders):
#         subfolders.extend(fast_scandir(dirname))
#     return subfolders

# def process_L8():
#     ms_dir = os.path.join(session_dir,file,'ms')
#     mask_dir = os.path.join(session_dir,file,'mask')
#     pan_dir = os.path.join(session_dir,file,'pan')

# 2024-02-28-19-04-18_RGB_S2.jpg
settings = {
    'max_translation': 1000,
    'min_translation': -1000,
}

results = []

# list the directories in the session directory
# 2. loop through the directories
for file in os.listdir(session_dir):
    if os.path.isdir(os.path.join(session_dir,file)) and file in satellites:
        # read through ms directory and get base names
        ms_dir = os.path.join(session_dir,file,'ms')
        mask_dir = os.path.join(session_dir,file,'mask')
        # pan_dir = os.path.join(session_dir,file,'pan')
        swir_dir = os.path.join(session_dir,file,'swir')

        # get the base names of the files in the ms directory
        basenames = [os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(ms_dir, '*.tif'))]
        print(basenames)

        # get the files from the ms directory
        # coregister each ms file and save to a new directory called coregistered
        tif_files = glob.glob(os.path.join(ms_dir, '*.tif'))
        # coregistered_directory = os.path.join(ms_dir,'coregistered')
        # matching_window_strategy='use_predetermined_window_size'
        # coregistered_directory = os.path.join(ms_dir,'coregistered_max_overlap')

        # matching_window_strategy='max_overlap' # this strategy does not work...
        matching_window_strategy='max_center_size'
        coregistered_directory = os.path.join(ms_dir,matching_window_strategy+'_no_histogram_match'+f"_window_size_{window_size_str}")
        os.makedirs(coregistered_directory, exist_ok=True)

        # Start by looping through all the files in a directory
        if len(tif_files) < 1:
            print("No TIFF files found in the directory.")
            raise SystemExit
        for target_path in tqdm.tqdm(tif_files):
            # target_path = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52\S2\ms\2024-05-23-22-18-06_S2_ID_1_datetime11-04-24__04_30_52_ms.tif"
            # output_filename = os.path.basename(target_path).split('.')[0] + '_coregistered.tif'
            output_filename = os.path.basename(target_path)
            output_path = os.path.join(coregistered_directory, output_filename)
            try:
                coreg = CoregisterInterface(target_path=target_path, template_path=template_path, output_path=output_path,window_size=WINDOW_SIZE,settings=settings, verbose=False,matching_window_strategy=matching_window_strategy)
                if 'no valid matching window found' in coreg.get_coreg_info()['description']:
                    print(f"Skipping {os.path.basename(target_path)} due to no valid matching window found.")
                    new_result = {
                        os.path.basename(target_path): coreg.get_coreg_info()
                    }
                    results.append(new_result)
                    continue
                coreg.coregister()
                # remake jpg 
            except Exception as e:
                import traceback
                print(f"Error: {e}")
                traceback.print_exc()
                new_result = {
                    os.path.basename(target_path): coreg.get_coreg_info()
                }
            else:
                new_result = {
                    os.path.basename(target_path): coreg.get_coreg_info()
                }
            results.append(new_result)

print(f"len(results): {len(results)}")
# Save the results to the same coregistered directory
results = plotting.merge_list_of_dicts(results)

settings.update({'window_size': WINDOW_SIZE, 'template_path': template_path, })
results['settings'] = settings

# 5. set the output path to save the transformation results
result_json_path = os.path.join(coregistered_directory, 'transformation_results.json')
with open(result_json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

# @todo make sure to fix this so that it works even if the json contains only a single entry
plotting.create_readme(coregistered_directory, result_json_path)

plotting.plot_ssim_scores(result_json_path, coregistered_directory)
plotting.plot_ssim_scores_dev(result_json_path, coregistered_directory)
plotting.plot_shifts_by_month(result_json_path, coregistered_directory)
plotting.plot_shift_histogram(result_json_path, coregistered_directory)
plotting.plot_delta_ssim_scores(result_json_path, coregistered_directory)
plotting.plot_coregistration_success_by_month(result_json_path, coregistered_directory)