import rasterio
import numpy as np
import os
import math
import json
import rasterio
import glob


from skimage.metrics import structural_similarity as ssim
from skimage import exposure
from tqdm import tqdm
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
WINDOW_SIZE=(100,100)
window_size_str= f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}"

# Paths
unregistered_directory = r'C:\development\coastseg-planet\downloads\UNALAKLEET_pier_cloud_0.7_TOAR_enabled_2020-06-01_to_2023-08-01\e2821741-0677-435a-a93a-d4f060f3adf4\PSScene'
# C:\development\coastseg-planet\downloads\UNALAKLEET_pier_cloud_0.7_TOAR_enabled_2020-06-01_to_2023-08-01\e2821741-0677-435a-a93a-d4f060f3adf4\PSScene
template_path = r"C:\3_code_from_dan\6_coregister_implementation_coastseg\raw_coastseg_data\L9\ms\2023-08-01-22-02-10_L9_ID_uxk1_datetime11-04-24__05_08_02_ms.tif"

# get the basenames from the filtered planet imagery
filtered_planet_dir = r'C:\3_code_from_dan\2_coregistration_unalakleet\unalakeet\planet_imagery'
filtered_planet_files = glob.glob(os.path.join(filtered_planet_dir, '*TOAR_model_format.tif'))
print(f"len(filtered_planet_basenames): {len(filtered_planet_files)}")
# get the basenames by splitting at 3b
filtered_planet_basenames = [os.path.basename(f).split('3B')[0] for f in filtered_planet_files]

print(f"len(filtered_planet_basenames): {len(filtered_planet_basenames)}")
# print(f"filtered_planet_basenames: {filtered_planet_basenames}")

# Target path
all_tif_files = glob.glob(os.path.join(unregistered_directory, '*3B_AnalyticMS_toar_clip.tif'))

print(f"len(tif_files): {len(all_tif_files)}")

tif_files = []

for f in all_tif_files:
    # print((os.path.basename(f).split('3B')[0]))
    if os.path.basename(f).split('3B')[0] in filtered_planet_basenames:
        # print(f"Skipping {f}")
        tif_files.append(f)


print(f"len(tif_files): {len(tif_files)}")
# output directory
coregistered_directory = r"C:\3_code_from_dan\2_coregistration_unalakleet\unalakeet\2_coregistered_planet_new_shift_method_target_band_red"
os.makedirs(coregistered_directory, exist_ok=True)
print(f"Output directory: {coregistered_directory}")
settings = {
    'max_translation': 1000,
    'min_translation': -1000,
}

TARGET_BAND = 3
TEMPLATE_BAND = 1

# # I need to apply this in bulk to all images in the directory
# # I need to save the results in a json file
# # I need to plot the results
# # I need to create a readme file
results = []

# Start by looping through all the files in a director
if len(tif_files) < 1:
    print("No TIFF files found in the directory.")
    raise SystemExit
for target_path in tqdm(tif_files):
    # output_filename = os.path.basename(target_path).split('.')[0] + '_coregistered.tif'
    output_filename = os.path.basename(target_path)
    output_path = os.path.join(coregistered_directory, output_filename)
    try:
        coreg = CoregisterInterface(target_path=target_path, template_path=template_path, output_path=output_path,window_size=WINDOW_SIZE,settings=settings, verbose=False,target_band=TARGET_BAND ,template_band=TEMPLATE_BAND)
        coreg.coregister()
        # remake jpg 
    except Exception as e:
        print(f"Error: {e}")
    
    new_result = {
        os.path.basename(target_path): coreg.get_coreg_info()
    }
    results.append(new_result)

print(f"len(results): {len(results)}")
# Save the results to the same coregistered directory
results = plotting.merge_list_of_dicts(results)

settings.update({'window_size': WINDOW_SIZE, 'template_path': template_path,'target_band': TARGET_BAND, 'template_band': TEMPLATE_BAND})
results['settings'] = settings

# 5. set the output path to save the transformation results
result_json_path = os.path.join(coregistered_directory, 'transformation_results.json')
with open(result_json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Saved results to: {result_json_path}")
# @todo make sure to fix this so that it works even if the json contains only a single entry
plotting.create_readme(coregistered_directory, result_json_path)

plotting.plot_ssim_scores(result_json_path, coregistered_directory)
plotting.plot_ssim_scores_dev(result_json_path, coregistered_directory)
plotting.plot_shifts_by_month(result_json_path, coregistered_directory)
plotting.plot_shift_histogram(result_json_path, coregistered_directory)
plotting.plot_delta_ssim_scores(result_json_path, coregistered_directory)
plotting.plot_coregistration_success_by_month(result_json_path, coregistered_directory)