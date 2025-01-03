import os
import json
import traceback

from coregister_class import CoregisterInterface
import file_utilites  as file_utils


# Step 1. Set the settings
#--------------------------
# Settings
WINDOW_SIZE=(256,256) 

settings = {
    'max_translation': 1000,  # max translation in meters
    'min_translation': -1000, # min translation in meters
}
VERBOSE = False # Set this to True to see the debug information during coregistration

# Choose  a matching window strategy
# 1. 'max_center_size' - This strategy will find the largest window that fits within the target image and the template image
# 2. 'optimal_centered_window' - This strategy will find the largest window that fits within the target image and the template image and is centered. Only downside is that this is slow for large images


# matching_window_strategy = 'max_center_size' #optimal_centered_window
matching_window_strategy = 'optimal_centered_window'

TARGET_BAND = 1
TEMPLATE_BAND = 1

# Step 2. Enter the template path and target path
#-------------------------------------------------
# template path 
# - This is the tiff file that you want to coregister the target file to
# - Ensure this file has accurate georeferencing (such as a Landsat scene) at the SAME location as the target image

# Target path
# - This is the tiff file that you want to coregister to the template

template_path = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime08-14-24__11_57_56\L9\ms\2024-04-06-18-46-04_L9_ID_1_datetime08-14-24__11_57_56_ms.tif"
target_path =r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime08-14-24__11_57_56\S2\ms\2023-07-13-19-04-28_S2_ID_1_datetime08-14-24__11_57_56_ms.tif"

# output directory to save the coregistered images
coregistered_directory = os.path.join(os.getcwd(), 'coregistered')
os.makedirs(coregistered_directory, exist_ok=True)
print(f"Coregistered imagery will be saved to : {coregistered_directory}")


# Step 3. Coregister the image
#-----------------------------

results = []

output_filename = os.path.basename(target_path)
output_path = os.path.join(coregistered_directory, output_filename)
try:
    coreg = CoregisterInterface(target_path=target_path, template_path=template_path, output_path=output_path,window_size=WINDOW_SIZE,settings=settings, verbose=VERBOSE,target_band=TARGET_BAND ,template_band=TEMPLATE_BAND, matching_window_strategy=matching_window_strategy)
    if coreg.bounds == (None, None, None, None):
        print(f"Skipping {os.path.basename(target_path)} due to no valid matching window found.")
        new_result = {
            os.path.basename(target_path): coreg.get_coreg_info()
        }
    else:
        coreg.coregister()
        print(f"Saved the coregistered image to: {output_path}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

new_result = {
    os.path.basename(target_path): coreg.get_coreg_info()
}
results.append(new_result)

# # Save the results to the same coregistered directory
results = file_utils.merge_list_of_dicts(results)

settings.update({'window_size': WINDOW_SIZE, 'template_path': template_path,'target_band': TARGET_BAND, 'template_band': TEMPLATE_BAND})
results['settings'] = settings

# 5. set the output path to save the transformation results
result_json_path = os.path.join(coregistered_directory, 'transformation_results.json')
with open(result_json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Saved the coregistration results to: {result_json_path}")
