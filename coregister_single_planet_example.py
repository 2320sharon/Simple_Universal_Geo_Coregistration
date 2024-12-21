import os
import json
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering
import traceback

from coregister_class import CoregisterInterface
import file_utilites  as file_utils


# Step 1. Set the settings
#--------------------------
# Settings
WINDOW_SIZE=(256,256) 
settings = {
    'max_translation': 1000,
    'min_translation': -1000,
}
VERBOSE = True
matching_window_strategy = 'optimal_centered_window'

# Pay attention to the band numbers
# For this example I am using a planet that has the band order of [blue, green, red, nir] while the landsat scene has the band order of [red,blue,green nir, swir,]
# For coregistration to be most effective the bands should be similar so I am using the red band from the planet image and the red band from the landsat image
# The red band is band 3 in the planet image (target) and band 1 in the landsat image (template)
TARGET_BAND = 3
TEMPLATE_BAND = 1

# Step 2. Enter the template path and target path
#-------------------------------------------------
# template path 
# - This is the tiff file that you want to coregister to
# - Ensure this file has accurate georeferencing (such as a Landsat scene) at the SAME location as the target image
template_path = r""
# Target path
# - This is the tiff file that you want to coregister to the template
target_path = r""

# output directory to save the coregistered images
coregistered_directory = os.path.join(os.getcwd(), 'coregistered_planet')
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
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

new_result = {
    os.path.basename(target_path): coreg.get_coreg_info()
}
results.append(new_result)

print(f"result : {new_result}")


print(f"len(results): {len(results)}")
# # Save the results to the same coregistered directory
results = file_utils.merge_list_of_dicts(results)

settings.update({'window_size': WINDOW_SIZE, 'template_path': template_path,'target_band': TARGET_BAND, 'template_band': TEMPLATE_BAND})
results['settings'] = settings

# 5. set the output path to save the transformation results
result_json_path = os.path.join(coregistered_directory, 'transformation_results.json')
with open(result_json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Saved results to: {result_json_path}")
