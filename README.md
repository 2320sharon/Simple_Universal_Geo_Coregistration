# Simple_Universal_Coregistration

⚠️This tool is still under active development and the documentation is unfinished. Not ready for operational usage.⚠️

A simple coregistration tool that uses phase cross correlation to determine the shift to match two geospatial images.

## How it Works
This method works by initially providing a template/reference tif that will be used as the source of truth to coregister the target images to. Make sure the template image does not contain excessive cloud cover, fog, or no data pixels as these will negatively impact the results.

1. **Overlap Detection:** Identify if the template image and target image overlap.
2. **Reprojection:** Reproject the target or template image to match the resolution of the tiff with the lowest resolution.
3. **Data Type Adjustment:** Modify the data type of the target or template to match the tiff with the largest data type.
4. **Matching Window:** If overlapping, create a window of `window_size` in the overlapping region. The location of the window will vary depending on the matching window strategy selected.
5. **Image Cropping:** Crop both the target and template to the matching window.
6. **Histogram Matching:** Adjust the target image's histogram to match that of the template.
7. **Shift Detection:** Run the phase_cross_correlation function to identify the shift within 1/100 pixels.
8. **Quality Control:** Apply filters to remove shifts exceeding predefined minimum and maximum translation limits.
9. **Image Shifting:** Copy and shift the target tiff by the detected shifts.

# Installation and Usage

### Basic Setup

1. **Clone the repository:** `git clone https://github.com/<your-username>/Simple_Universal_Coregistration.git`
2. **Navigate to the tool directory:** `cd Simple_Universal_Coregistration`
3. **Install dependencies:** `pip install -r requirements.txt`

## How to Use on a dataset
1. Initialize a `CoregisterInterface` class instance for each image.
2. Ensure template and target images from different sources have matching `target_band` and `template_band`.
3. Run coregistration in batches by satellite type.
4. Execute post-processing scripts to filter out outlier shifts.

## Settings

- **Window Size:**  Max size of the window to determine coregistration within. Default is `(256,256)`.
  - This is the size of the region that both the target image ( image to coregister) and the reference image ( image to use a reference to coregister) will be cropped to
  - Make sure that this value is even and ensure that it is large enough to capture important feature for coregistration.
- **Matching Window Strategy:** 
  - `max_center_size` (default): Finds the largest possible window at the center.
  - `use_predetermined_window_size`: Uses a predefined window size, sliding across the overlap until a fit is found.
  - `optimal_centered_window`: Finds all the possible windows in the region of overlap then selects the window that is cloest to center and the largest.
      This is a fallback method that is very slow but reliable. 
- **Minimum Window Size(min_window_size):** Smallest permissible window size `(64,64)`.
- **Gaussian Weights(gaussian_weights):** Applies more weight to central features, improving focus during matching.
  - Whether to use Gaussian weights for SSIM calculations
- **target_band:** The target's band number that should be used to coregister. Default is 1
    -   Make sure the target band and template band point to the same type of band. For example target band 1 is red and template band 3 is red.
- **template_band:** The template's band number that should be used to coregister. Default is 1
- **settings:**
  -   1. max_translation (float): Maximum translation (in meters) allowed for coregistration. Defaults to 1000m.
  -   2. min_translation (float): Minimum translation (in meters) allowed for coregistration. Defaults to -1000m.
  

# Coregistration Result
- Result of each individual coregistraion is stored in `CoregisterInterface().coreg_info`
- 
```
    {
        'shift_x': 0.0,
        'shift_y': 0.0,
        'shift_x_meters': 0.0,   # inital shift in x direction before any quality control in meters
        'shift_y_meters': 0.0,   # inital shift in y direction before any quality control in meters
        'initial_shift_x': 0.0,
        'initial_shift_y': 0.0, # inital shift before any quality control in pixels ( pixels resolution is that of the image with the lowest resolution)
        'error': 0.0,
        'shift_reliability': 0.0, 
        'qc': 0.0,
        'description': '',  # did coregistration succeed, if not why it failed 
        'success': 'False', # coregistered_ssim > original_ssim
        'original_ssim': 0.0,
        'coregistered_ssim': 0.0,
        'change_ssim': 0.0, # change in the ssim value after coregistration (coregistered_ssim - original_ssim)
        'window_size': (256, 256), # window size used to coregister this image
    }
```
- 


## Example
<Show place>
<Show Settings>
<Show results and json file>
<Show Before and After>

## How to Use with CoastSeg
`coregister_class_coastseg_zoo_implementation.py`
1. Select a session from the `/data` folder within CoastSeg
2. Select the ROI ID for the ROI you wish to register
3. Select a template image either a landsat 8 or landsat 9
4. Adjust the settings
<Make an example>

# Outlier Filtering
Outlier filtering is performed by reading the results of coregistering all the images in a dataset and filtering out bad shifts relative to the other files in the dataset.

Functions to apply outlier filtering are available in the `filters` module. Currently two methods of outlier filtering are available.
1. Filters out shifts that exceed the minimum and maximum (x,y) shifts in pixels
2. Filters out shifts by z score. The z score is the combined z score of the x and y pixel shifts for each image.
   - The combined z score = $$\sqrt{x_{\text{zscore}} + y_{\text{zscore}}}$$
   - If the combined z score exceeds the z_threshold then it is flagged as an outlier in the CSV generated by `apply_outliers`



```
filter_settings = {
    'min_shift_meters': (-200,-200),
    'max_shift_meters': (200,200),
    'z_threshold': 1.5,         # Set to None to disable outlier detection using the z score
}

output_csv_path = os.path.join(coregistered_dir, 'filtered_files.csv')
output_csv_path = filters.apply_filtering(results,output_csv_path, filter_settings)
df = pd.read_csv(output_csv_path)

# Move the files that failed the filtering to a new directory in coregistered directory
failed_coregs = df[~df['filter_passed']].groupby('satellite')['filename'].apply(list)
file_utils.process_failed_coregistrations(failed_coregs, coregistered_dir, session_dir,replace=replace_failed_files, copy_only=False, move_only=True, subfolder_name='ms')


```



<Explain two method of outlier filtering>
![plot_outlier_shifts](https://github.com/user-attachments/assets/69f21601-4c22-4b75-ab98-bddc19e1b614)
![combined_z_scores](https://github.com/user-attachments/assets/fe6d3ce3-a080-41ed-b6ef-a6dcca0702e2)


