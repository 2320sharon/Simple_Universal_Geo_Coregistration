# Simple_Universal_Coregistration
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
- Window Size:
-   1. 
-   2. 
- min_window_size : Smallest window size
-

# Outlier Filtering
![plot_outlier_shifts](https://github.com/user-attachments/assets/69f21601-4c22-4b75-ab98-bddc19e1b614)
![combined_z_scores](https://github.com/user-attachments/assets/fe6d3ce3-a080-41ed-b6ef-a6dcca0702e2)


