# Simple_Universal_Coregistration

⚠️This tool is still under active development and the documentation is unfinished. Not ready for operational usage.⚠️

A simple coregistration tool that uses phase cross correlation to determine the shift to match two geospatial images. 

## This method works by:
1. Taking a template image and a target image and determining if they overlap
2. The target or template is reprojected to the tiff with the lowest resolution.
3. The target or template has its data type modified to match the tiff with the largest data type.
4. If these do overlap and `matching_window_strategy` is set to `max_center_size` a window is created of `window_size` at the center of the overlap
5. Both the target and template are cropped to this matching window
6. The target is histogram matched to the template
7. The phase_cross_correlation function is run to identify the shift within 1/100 pixels
8. Quality control is applied to filter out shifts larger than the min and max translation 
9. The target tiff is copied and shifted by the detected shifts

## How to use on a dataset
1. Create a coregisterinterface class for each image
3. Correct coregister on batches of one satellite type at a time
- Make sure if your template and target come from different sources that the `target_band` and `template_band` match
3. Run a post processing script to filter out any outlier shifts (function coming soon)

## Settings
- Window Size: Max size of the window to determine coregistration within. Defaults to 100,100
  - This is the size of the region that both the target image ( image to coregister) and the reference image ( image to use a reference to coregister) will be cropped to
  - Make sure that this value is even and ensure that it is large enough to capture important feature for coregistration.
- Matching Window Strategy
  -   1.  `max_center_size` : Finds the largest possible window at the center of the overlap. (default)
  -   2. `use_predetermined_window_size`; Finds the window of window size at the first avaiable location by sliding a window across the overlap region.
- min_window_size : Smallest window size that can be used to perform coregistration default to (64,64)
- gaussian_weights : Whether to use Gaussian weights for SSIM
   - This puts more weight on the features at the center of the image
- target_band :
- template_band: 
- settings:
  -   1. max_translation (float): Maximum translation (in meters) allowed for coregistration. Defaults to 1000m.
  -   2. min_translation (float): Minimum translation (in meters) allowed for coregistration. Defaults to -1000m.
            

- <todo explain rest of settings>

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
<Explain two method of outlier filtering>
![plot_outlier_shifts](https://github.com/user-attachments/assets/69f21601-4c22-4b75-ab98-bddc19e1b614)
![combined_z_scores](https://github.com/user-attachments/assets/fe6d3ce3-a080-41ed-b6ef-a6dcca0702e2)


