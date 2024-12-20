import numpy as np
import os
import json
import glob
import tqdm
import threading
import matplotlib
from coregister_class import CoregisterInterface
import plotting
import file_utilites  as file_utils
from collections import OrderedDict

matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering

lock = threading.Lock()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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
    results[satellite] = file_utils.merge_list_of_dicts(results[satellite])
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

def coregister_files(tif_files:list, template_path:str, coregistered_directory:str, satellite:str, WINDOW_SIZE:tuple, settings:dict, matching_window_strategy:str):
    results = []
    for target_path in tqdm.tqdm(tif_files[:2],desc=f"Coregistering files for {satellite}"):
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
    return results

# Settings
WINDOW_SIZE=(256,256)
window_size_str= f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}"

session_dir = r'C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52_original'
template_path = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52_original\L9\ms\2023-06-30-22-01-55_L9_ID_1_datetime11-04-24__04_30_52_ms.tif"

# get the dirs of satellites
satellites = ['S2', 'L8','L9']
settings = {
    'max_translation': 1000,
    'min_translation': -1000,
}
matching_window_strategy='max_center_size'

results = {}
coregistered_path = os.path.join(session_dir,matching_window_strategy+'_no_histogram_match'+f"_window_size_{window_size_str}")
os.makedirs(coregistered_path , exist_ok=True)

# list the directories in the session directory
# 2. loop through the directories
for folder in os.listdir(session_dir):
    if os.path.isdir(os.path.join(session_dir,folder)) and folder in satellites:
        # 3. get the satellite name
        satellite = folder
        # read through ms directory and get base names
        ms_dir = os.path.join(session_dir,satellite,'ms')
        # get the base names of the files in the ms directory
        basenames = [os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(ms_dir, '*.tif'))]
        # get the files from the ms directory
        # coregister each ms file and save to a new directory called coregistered
        tif_files = glob.glob(os.path.join(ms_dir, '*.tif'))

        coregistered_directory = os.path.join(coregistered_path, satellite)
        os.makedirs(coregistered_directory, exist_ok=True)

        # Start by looping through all the files in a directory
        if len(tif_files) < 1:
            print("No TIFF files found in the directory.")
            raise SystemExit
        results[satellite] = coregister_files(tif_files, template_path, coregistered_directory, satellite, WINDOW_SIZE, settings, matching_window_strategy)
        # Save the results to the same coregistered directory
        if len(results[satellite]) > 1:
            results[satellite] = file_utils.merge_list_of_dicts(results[satellite])
        

settings.update({'window_size': WINDOW_SIZE, 'template_path': template_path, })
results['settings'] = settings

# 5. set the output path to save the transformation results
result_json_path = os.path.join(coregistered_path, 'transformation_results.json')
with open(result_json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4, cls=NumpyEncoder)
# save_coregistered_results(results, satellite, WINDOW_SIZE, template_path, result_json_path, settings)
print(f"Saved results to: {result_json_path}")
with open(result_json_path, 'r') as json_file:
    results = json.load(json_file)

# print(f"results: {results}")
# # 6. plot the results

plotting.plot_ssim_scores(results, coregistered_directory)
plotting.plot_ssim_scores_dev(results, coregistered_directory)
plotting.plot_shifts_by_month(results, coregistered_directory)
plotting.plot_shift_histogram(results, coregistered_directory)
plotting.plot_delta_ssim_scores(results, coregistered_directory)
plotting.plot_coregistration_success_by_month(results, coregistered_directory)