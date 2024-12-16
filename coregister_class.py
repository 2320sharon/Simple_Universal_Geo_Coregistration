import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from scipy.spatial.distance import cdist
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from numpy.fft import fft2, ifft2, fftshift
from skimage.registration import phase_cross_correlation
from math import sqrt
import hashlib

# matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering
# plt.ion()  # Turn on interactive mode
lock = threading.Lock()


#-----------
# Created by Sharon Fitzpatrick Batiste
# Date: 2024 - 11 -10
#-----------
# Remaining tasks
# - Determine if the scale factor is needed
# - Make the sure the temporary files that are created to reproject the template are deleted (they aren't right now)
# - Make sure it works even if the template is reprojected
# - Make sure it works even if the template and target are the same resolution

# 12/2/24
# remove scale factor (not used since we determine the shift in meters)
# remove the temporary files that are created to reproject the template
# works even if the template is reprojected
# works even if the template and target are the same resolution


def find_best_window_in_combined_mask(tiff1_path, tiff2_path, min_height=16, min_width=16, max_height=256, max_width=256, alpha=0.6, beta=0.4):
    """
    Finds the best window within the combined mask using constraints on size, center proximity, and max dimensions.

    Args:
        combined_mask (numpy.ndarray): Boolean mask where True indicates NoData pixels.
        min_height (int): Minimum allowed height of the window.
        min_width (int): Minimum allowed width of the window.
        max_height (int, optional): Maximum allowed height of the window.
        max_width (int, optional): Maximum allowed width of the window.
        alpha (float): Weight for the area in the scoring function.
        beta (float): Weight for the distance to the center in the scoring function.

    Returns:
        tuple: Coordinates of the best window as (row_start, col_start, row_end, col_end).
               If no valid window is found, raises a ValueError.
    """
    combined_mask = get_combined_mask(tiff1_path, tiff2_path)

    # Create a unique hash for the combined_mask
    def mask_to_hash(mask):
        return hashlib.md5(mask.tobytes()).hexdigest()

    # Cache dictionary
    if not hasattr(find_best_window_in_combined_mask, "_cache"):
        find_best_window_in_combined_mask._cache = {}

    cache = find_best_window_in_combined_mask._cache

    # Generate hash of the combined_mask
    mask_hash = mask_to_hash(combined_mask)

    # If result is cached, return it
    if mask_hash in cache:
        return cache[mask_hash]

    # Ensure mask is inverted (0 = NoData, 1 = Valid Data)
    valid_mask = ~combined_mask

    rows, cols = valid_mask.shape
    histogram = np.zeros(cols, dtype=int)

    # Center of the mask
    center_row, center_col = rows / 2, cols / 2
    max_area = valid_mask.size
    max_center_dist = sqrt(center_row**2 + center_col**2)

    all_rectangles = []

    # Process each row to build histograms and collect all valid rectangles
    for r in range(rows):
        for c in range(cols):
            histogram[c] = histogram[c] + 1 if valid_mask[r, c] else 0

            # Limit the histogram height to max_height, if specified
            if max_height and histogram[c] > max_height:
                histogram[c] = max_height

        # Generate rectangles from the histogram
        for i in range(cols):
            min_height = histogram[i]
            for j in range(i, min(i + (max_width or cols), cols)):
                min_height = min(min_height, histogram[j])
                width = j - i + 1

                # Skip rectangles smaller than min dimensions
                if min_height < min_height or width < min_width:
                    continue

                # Skip rectangles with odd dimensions
                if min_height % 2 != 0 or width % 2 != 0:
                    continue

                # Add rectangle (row_end, col_start, col_end, height)
                all_rectangles.append((r, i, j, min_height))

    # Find the best rectangle based on scoring
    best_score = float('-inf')
    best_coords = None

    for rect in all_rectangles:
        rect_row_end, rect_col_start, rect_col_end, rect_height = rect
        rect_width = rect_col_end - rect_col_start + 1
        rect_row_start = rect_row_end - rect_height + 1

        # Calculate area and center distance
        rect_area = rect_width * rect_height
        rect_center_row = (rect_row_start + rect_row_end) / 2
        rect_center_col = (rect_col_start + rect_col_end) / 2
        center_dist = sqrt((center_row - rect_center_row)**2 + (center_col - rect_center_col)**2)

        # Normalize metrics
        normalized_area = rect_area / max_area
        normalized_center_dist = center_dist / max_center_dist

        # Calculate score
        score = alpha * normalized_area - beta * normalized_center_dist

        if score > best_score:
            best_score = score
            best_coords = (rect_row_start, rect_col_start, rect_row_end, rect_col_end)

    if not best_coords:
        raise ValueError("No valid window found in the combined mask.")

    row_start, col_start, row_end, col_end = best_coords
    window_size_x = col_end - col_start + 1
    window_size_y = row_end - row_start + 1


    #row_start, col_start, row_end, col_end = best_coords
    # best coords are  xmin : coords[1] , ymin : coords[0] , xmax : coords[3] , ymax : coords[2]
    cache[mask_hash] = (window_size_x, window_size_y),best_coords

    return (int(window_size_x), int(window_size_y)),best_coords

def calculate_shift_reliability(template, target,):
    # Step 2: Compute the Cross Power Spectrum (CPS)
    fft_template = fft2(template)
    fft_target = fft2(target)
    eps = 1e-15  # Avoid division by zero
    cps = fft_template * np.conjugate(fft_target) / (np.abs(fft_template) * np.abs(fft_target) + eps)
    correlation_map = fftshift(np.abs(ifft2(cps)))

    # Step 3: Locate the peak in the correlation map
    peak_row, peak_col = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
    peak_strength = np.mean(correlation_map[peak_row - 1:peak_row + 2, peak_col - 1:peak_col + 2])

    # Step 4: Mask the peak and calculate background statistics
    correlation_map_masked = correlation_map.copy()
    correlation_map_masked[peak_row - 1:peak_row + 2, peak_col - 1:peak_col + 2] = -9999
    background_values = correlation_map_masked[correlation_map_masked != -9999]
    background_mean = np.mean(background_values)
    background_std = np.std(background_values)

    # Step 5: Calculate reliability
    reliability = 100 - ((background_mean + 2 * background_std) / peak_strength * 100)
    reliability = max(0, min(100, reliability))  # Clamp between 0 and 100

    return  reliability

def read_resolutions(tiff_path):
    """
    Reads the spatial resolution of a TIFF file.
    
    Parameters:
    tiff_path (str): Path to the TIFF file.
    
    Returns:
    tuple: The spatial resolution of the TIFF file as (x_resolution, y_resolution).
    """
    with rasterio.open(tiff_path) as src:
        return src.res

def get_max_overlap_window_size_in_pixels(tiff1_path, tiff2_path):
    """
    Determines the largest possible square window size (in pixels) 
    that can fit entirely within the overlapping region of two TIFF files.

    Args:
        tiff1_path (str): Path to the first TIFF file.
        tiff2_path (str): Path to the second TIFF file.

    Returns:
        int: The largest possible square window size (in pixels) 
             that fits within the overlapping region.
    """
    import rasterio
    from shapely.geometry import box

    with rasterio.open(tiff1_path) as src1, rasterio.open(tiff2_path) as src2:
        # Get the bounding boxes of both rasters
        bbox1 = box(*src1.bounds)
        bbox2 = box(*src2.bounds)
        
        # Find the intersection of the bounding boxes
        intersection = bbox1.intersection(bbox2)
        if intersection.is_empty:
            raise ValueError("The two TIFF files do not overlap.")
        
        # Get the bounds of the intersection
        intersection_bounds = intersection.bounds
        left, bottom, right, top = intersection_bounds
        
        # Calculate the width and height of the intersection in pixels
        # Use the resolution of the first raster to compute pixel dimensions
        pixel_width = (right - left) / src1.res[0]
        pixel_height = (top - bottom) / src1.res[1]
        
        # Convert to integer pixel values
        intersection_width = int(pixel_width)
        intersection_height = int(pixel_height)
        
        # Determine the maximum square box size that can fit (in pixels)
        max_window_size = min(intersection_width, intersection_height)
        
        return max_window_size

def find_best_starting_point(tiff1_path, tiff2_path):
    """
    Finds the best starting point near the center of the overlapping region 
    of two TIFF files. If the center is a NoData point, searches for the closest 
    valid point with data in both TIFFs.

    Args:
        tiff1_path (str): Path to the first TIFF file.
        tiff2_path (str): Path to the second TIFF file.

    Returns:
        tuple: (row, col) pixel coordinates of the best starting point.
    """


    with rasterio.open(tiff1_path) as src1, rasterio.open(tiff2_path) as src2:
        # Define overlapping window between the two TIFFs
        overlap_window = src1.window(*src1.bounds).intersection(src2.window(*src2.bounds))
        # Convert to row/col coordinates for consistent box size
        row_start, col_start, row_end, col_end = map(int, overlap_window.flatten())

        # Mask out the valid data regions for both TIFFs
        valid_data_mask1 = src1.read(1, masked=True).mask[row_start:row_end, col_start:col_end]
        valid_data_mask2 = src2.read(1, masked=True).mask[row_start:row_end, col_start:col_end]

        # Combine masks to find invalid data regions
        combined_mask = valid_data_mask1 | valid_data_mask2  

        # Determine the central point
        row_center = combined_mask.shape[0] // 2
        col_center = combined_mask.shape[1] // 2

        # plot the combined mask on top of the target image band 1
        # plt.imshow(src1.read(1, window=overlap_window), cmap='gray')
        # # view of the overlap region for src2
        # plt.imshow(src2.read(1, window=overlap_window), cmap='gray', alpha=0.5)
        # # plt.imshow(combined_mask, cmap='gray', alpha=0.5)
        # plt.title("Combined Mask (True = NoData, False = Valid)")
        # plt.show()

        if not combined_mask[row_center, col_center]:
            return (row_center, col_center)  # Center point is valid

        # If the center is invalid, find the nearest valid point
        valid_points = np.column_stack(np.where(~combined_mask))  # Get all valid points
        if valid_points.size == 0:
            raise ValueError("No valid data points in the overlapping region.")

        # Calculate distances from the center to all valid points
        center_point = np.array([[row_center, col_center]])
        distances = cdist(center_point, valid_points, metric='euclidean')
        closest_index = np.argmin(distances)

        # Return the closest valid point
        best_point = tuple(valid_points[closest_index])
        return best_point

def get_combined_mask(tiff1_path, tiff2_path):
    with rasterio.open(tiff1_path) as src1, rasterio.open(tiff2_path) as src2:
        # Define overlapping window between the two TIFFs
        overlap_window = src1.window(*src1.bounds).intersection(src2.window(*src2.bounds))
        # Convert to row/col coordinates for consistent box size
        row_start, col_start, row_end, col_end = map(int, overlap_window.flatten())

        # Mask out the valid data regions for both TIFFs
        valid_data_mask1 = src1.read(1, masked=True).mask[row_start:row_end, col_start:col_end]
        valid_data_mask2 = src2.read(1, masked=True).mask[row_start:row_end, col_start:col_end]

        # Combine masks to find invalid data regions
        combined_mask = valid_data_mask1 | valid_data_mask2 

        return combined_mask

def get_valid_window_with_fallback(tiff1_path, tiff2_path, start_point=None, 
                                   initial_window_size=None, min_window_size=(16, 16)):
    """
    Determines the largest possible square window size (in pixels) 
    that can fit entirely within the overlapping region of two TIFF files,
    starting from a specified point. If an initial window size is provided, 
    it attempts to use that size first and shrinks the window until a valid 
    window is found or the minimum window size is reached. Ensures the window 
    size is always even.

    Args:
        tiff1_path (str): Path to the first TIFF file.
        tiff2_path (str): Path to the second TIFF file.
        start_point (tuple, optional): (row, col) pixel coordinates to start 
                                       creating the window from. Defaults to 
                                       the center of the overlapping region.
        initial_window_size (tuple, optional): (rows, cols) dimensions of the 
                                               initial window size to attempt.
        min_window_size (tuple): Minimum window size (rows, cols) allowed. 
                                 Defaults to (16, 16).

    Returns:
        tuple: Bounds of the valid window as (row_start, row_end, col_start, col_end).
               If no valid window is found, raises a ValueError.
    """
    combined_mask = get_combined_mask(tiff1_path, tiff2_path)


    with rasterio.open(tiff1_path) as src1, rasterio.open(tiff2_path) as src2:
        # Determine the starting point
        if start_point is None:
            row_center = (combined_mask.shape[0]) // 2
            col_center = (combined_mask.shape[1]) // 2
        else:
            row_center, col_center = start_point

        if not (0 <= row_center < combined_mask.shape[0] and 0 <= col_center < combined_mask.shape[1]):
            raise ValueError("Start point is out of bounds of the overlapping region.")

        if combined_mask[row_center, col_center]:
            raise ValueError("Start point is located in a NoData region.")

        # Initialize window dimensions
        if initial_window_size is None:
            initial_window_size = (min_window_size[0], min_window_size[1])

        # Ensure window size is even
        def make_even(size):
            return size if size % 2 == 0 else size - 1

        rows, cols = map(make_even, initial_window_size)
        min_rows, min_cols = map(make_even, min_window_size)

        # Attempt to create a valid window
        def is_valid_window(window_size):
            rows, cols = window_size
            half_rows, half_cols = rows // 2, cols // 2
            row_start = row_center - half_rows
            row_end = row_center + half_rows
            col_start = col_center - half_cols
            col_end = col_center + half_cols
            
            # Check if the window exceeds bounds
            if (row_start < 0 or col_start < 0 or 
                row_end > combined_mask.shape[0] or col_end > combined_mask.shape[1]):
                return False, None
            
            # Check if the window contains any NoData pixels
            # plot the combined mask
            # plt.imshow(combined_mask[row_start:row_end, col_start:col_end].astype(int), cmap='gray')
            # plt.colorbar()
            # plt.title("Combined Mask (True = NoData, False = Valid)")
            # plt.show()
            if not combined_mask[row_start:row_end, col_start:col_end].any():
                # return True, (row_start, row_end, col_start, col_end)
                return True, (row_start,col_start,row_end,col_end)
            
            return False, None

        # Start with the initial window size and shrink if necessary
        while rows >= min_rows and cols >= min_cols:
            # print(f"rows: {rows}, cols: {cols}")
            valid, bounds = is_valid_window((rows, cols))
            if valid:
                return (int(rows), int(cols)), bounds
            
            # Shrink window size, keeping it even
            rows -= 2
            cols -= 2

        # If no valid window is found
        return (0,0),(None, None, None, None)

def get_max_valid_window_size_and_bounds(tiff1_path, tiff2_path):
    """
    Determines the largest possible square window size (in pixels) 
    that can fit entirely within the overlapping region of two TIFF files,
    ensuring the window contains no NoData pixels in both TIFFs.

    Args:
        tiff1_path (str): Path to the first TIFF file.
        tiff2_path (str): Path to the second TIFF file.

    Returns:
        int: The largest possible square window size (in pixels) 
             that contains no NoData pixels in both TIFF files.
        tuple: The pixel bounds of the largest valid window as 
               (row_start, row_end, col_start, col_end).
    """
    combined_mask = get_combined_mask(tiff1_path, tiff2_path)
    with rasterio.open(tiff1_path) as src1, rasterio.open(tiff2_path) as src2:
        # # Define overlapping window between the two TIFFs
        # overlap_window = src1.window(*src1.bounds).intersection(src2.window(*src2.bounds))
        # # Convert to row/col coordinates for consistent box size
        # row_start, col_start, row_end, col_end = map(int, overlap_window.flatten())

        # # Mask out the valid data regions for both TIFFs
        # valid_data_mask1 = src1.read(1, masked=True).mask[row_start:row_end, col_start:col_end]
        # valid_data_mask2 = src2.read(1, masked=True).mask[row_start:row_end, col_start:col_end]

        # # Combine masks to find invalid data regions
        # combined_mask = valid_data_mask1 | valid_data_mask2  

        max_window_size = 0
        best_bounds = None, None, None, None

        # Find the largest square window that contains no NoData pixels
        # Start with the largest even window size and decrement by 2 (to ensure even sizes)
        for window_size in range(min(combined_mask.shape) - (min(combined_mask.shape) % 2), 0, -2):
            found = False  # Flag to stop further iteration once a valid window is found
            for row in range(combined_mask.shape[0] - window_size + 1):  # Loop through rows
                for col in range(combined_mask.shape[1] - window_size + 1):  # Loop through columns
                    # Check if the window does not contain any `True` values
                    if not combined_mask[row:row + window_size, col:col + window_size].any():
                        max_window_size = window_size
                        best_bounds = (row, row + window_size, col, col + window_size)
                        found = True  # Mark as found
                        break  # Break the column loop
                if found:
                    break  # Break the row loop
            if found:
                break  # Break the window_size loop
            
        return max_window_size, best_bounds

def normalize(array: np.ndarray) -> np.ndarray:
    minval = np.min(array)
    maxval = np.max(array)
    # avoid zerodivision
    if maxval == minval:
        maxval += 1e-5
    return ((array - minval) / (maxval - minval)).astype(np.float64)

def apply_shift_to_tiff(target_path:str, output_path:str, shift:np.ndarray,verbose=False):
    """
    Applies a spatial shift to a GeoTIFF file and writes the result to a new file.
    Parameters:
    target_path (str): The file path to the input GeoTIFF file.
    output_path (str): The file path to save the output GeoTIFF file with the applied shift.
    shift (np.ndarray): A numpy array containing the shift values [y_shift, x_shift].
    verbose (bool, optional): If True, prints detailed information about the process. Default is False.
    Returns:
    None
    """
    if verbose:
        print(f"Applying shift {shift}")
    with rasterio.open(target_path) as src:
        meta = src.meta.copy()

        transform_shift = Affine.translation(shift[1], shift[0])  # x=shift[1], y=shift[0]
        meta['transform'] = src.transform * transform_shift

        # Ensure no changes to image data
        meta.update({
            'compress': src.compression if src.compression else 'lzw',  # Preserve compression
            'dtype': src.dtypes[0],  # Preserve data type
        })

        if src.nodata is not None:
            meta.update({'nodata': src.nodata})

        if verbose:
            print(f"Original transform:\n{src.transform}")
            print(f"Updated transform:\n{meta['transform']}")

        # Write a new file with updated metadata
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(src.read())  # Writes all bands directly

def masked_ssim(reference, target, ref_no_data_value=0,target_no_data_value=0,gaussian_weights=True):
    """
    Compute the Structural Similarity Index (SSIM) between two images, masking out no-data regions.
    Parameters:
    reference (ndarray): The reference image.
    target (ndarray): The target image to compare against the reference.
    ref_no_data_value (int or float, optional): The no-data value in the reference image. Default is 0.
    target_no_data_value (int or float, optional): The no-data value in the target image. Default is 0.
    gaussian_weights (bool, optional): Whether to use Gaussian weights for windowing. Default is True.
    Returns:
    float: The mean SSIM score over valid (non-no-data) regions.
    """
    # Create a mask for valid (non-no-data) pixels in both images
    mask = (reference != ref_no_data_value) & (target != target_no_data_value)
    

    reference_masked = np.where(mask, reference, 0)
    target_masked = np.where(mask, target, 0)
    
    # Compute SSIM only over the masked region (valid data)
    ssim_score, ssim_image = ssim(
        normalize(reference_masked), # could also try np.ma.masked_equal(reference, ref_no_data_value)
        normalize(target_masked),   # could also try np.ma.masked_equal(target, target_no_data_value)
        data_range=1,               # data was normalized to [0, 1]
        gaussian_weights=gaussian_weights, # Use Gaussian weights for windowing (gives more importance to center pixels)
        use_sample_covariance=False, # Use population covariance for SSIM
        full=True
    )

    # Mask out the no-data regions in the SSIM image
    ssim_image[~mask] = np.nan  # Set no-data regions to NaN

    # Compute the mean SSIM score over valid regions only
    mean_ssim = np.nanmean(ssim_image)  # Use np.nanmean to ignore NaN values


    return mean_ssim

def calc_ssim_in_bounds(template_path, target_path,template_nodata,target_nodata,row_start,col_start,row_end,col_end,gaussian_weights=True,target_band_number=1,template_band_number=1):
    """
    Calculate the Structural Similarity Index (SSIM) within specified bounds between two images.
    Parameters:
    template_path (str): Path to the template image file.
    target_path (str): Path to the target image file.
    template_nodata (float): No-data value for the template image.
    target_nodata (float): No-data value for the target image.
    row_start (int): Starting row index for the bounds.
    col_start (int): Starting column index for the bounds.
    row_end (int): Ending row index for the bounds.
    col_end (int): Ending column index for the bounds.
    gaussian_weights (bool, optional): Whether to use Gaussian weights for SSIM calculation. Default is True.
    target_band_number (int, optional): Band number to read from the target image. Default is 1.
    template_band_number (int, optional): Band number to read from the template image. Default is 1.
    Returns:
    float: SSIM score between the specified bounds of the template and target images.
    """
    # read the bands from the matching bounds
    template_band = read_bounds(template_path,row_start,col_start,row_end,col_end,band_number=template_band_number)
    target_band = read_bounds(target_path,row_start,col_start,row_end,col_end,band_number=target_band_number)
     
    # Compute SSIM score
    ssim_score = masked_ssim(template_band, target_band, template_nodata, target_nodata, gaussian_weights)
    return ssim_score

def read_bounds(tif_path, row_start, col_start, row_end, col_end,band_number=1):
    """
    Reads a specific window of data from a given TIFF file.
    Parameters:
    tif_path (str): Path to the TIFF file.
    row_start (int): Starting row index for the window.
    col_start (int): Starting column index for the window.
    row_end (int): Ending row index for the window.
    col_end (int): Ending column index for the window.
    band_number (int, optional): The band number to read from the TIFF file. Defaults to 1.
    Returns:
    numpy.ndarray: The data read from the specified window of the TIFF file.
    """
    with rasterio.open(tif_path) as src:
        # Create a window using the bounds
        window = rasterio.windows.Window.from_slices((row_start, row_end), (col_start, col_end))
        # Read only the first band within the window
        data_box = src.read(band_number, window=window)
        return data_box


def reproject_to_image(template_path, target_path, output_path):
    """
    Reprojects a target image to match the spatial resolution, transform, and CRS of a template image.
    Parameters:
    template_path (str): Path to the template image file which provides the reference resolution, transform, CRS, and nodata values.
    target_path (str): Path to the target image file that needs to be reprojected.
    output_path (str): Path where the reprojected image will be saved.
    Returns:
    str: The path to the reprojected image file.
    """
    
    # Open the reference image to get resolution, transform, CRS, and nodata values
    with rasterio.open(template_path) as ref:
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_crs = ref.crs

    # Open the target image and update metadata to match reference
    with rasterio.open(target_path) as target:
        target_meta = target.meta.copy()
        target_meta.update({
            'driver': 'GTiff',
            'height': ref_height,
            'width': ref_width,
            'transform': ref_transform,
            'crs': ref_crs,
        })

        # Create output file and perform resampling
        with rasterio.open(output_path, 'w', **target_meta) as dst:
            for i in range(1, target.count + 1):  # Iterate over each band
                target_band = target.read(i)
                reproject(
                    source=target_band,
                    destination=rasterio.band(dst, i),
                    src_transform=target.transform,
                    src_crs=target.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.cubic,
                    src_nodata=target.nodata,
                    dst_nodata=target.nodata,  # keep the original no data value
                    dst_resolution=(ref_transform[0], ref_transform[4])
                )

    return output_path

def write_window_to_tiff(window,src_path, output_path):
    print(f"Window dimensions: {window.width}, {window.height}")
    with rasterio.open(src_path) as src:
        data = src.read(window=window)
        print(f"Data shape: {data.shape}")
        # Update the transform to reflect the window's position
        transform = src.window_transform(window)
        # Define metadata for the new file
        out_meta = src.meta.copy()
        out_meta.update({
            "height": window.height,
            "width": window.width,
            "transform": transform
        })
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(data)
    return output_path

def find_shift(template: np.ndarray, target: np.ndarray) -> tuple:
    """
        Calculate the shift required to align the target image with the template image using phase cross-correlation.
        Parameters:
        template (np.ndarray): The reference image to which the target image will be aligned.
        target (np.ndarray): The image that needs to be aligned to the template.
        Returns:
        tuple: A tuple containing:
            - shift (np.ndarray): The estimated pixel shift required to align the target image with the template.
            - error (float): The error of the cross-correlation.
            - diffphase (float): The global phase difference between the images.
    """
    # Perform cross-correlation with masks
    shift, error, diffphase = phase_cross_correlation(
        template, target, upsample_factor=100,
    )

    return shift, error, diffphase

class CoregisterInterface:
    DEFAULT_COREG_INFO ={
        'shift_x': 0.0,
        'shift_y': 0.0,
        'shift_x_meters': 0.0,
        'shift_y_meters': 0.0,
        'initial_shift_x': 0.0,
        'initial_shift_y': 0.0,
        'error': 0.0,
        'shift_reliability': 0.0,
        'qc': 0.0,
        'description': '',
        'success': 'False',
        'original_ssim': 0.0,
        'coregistered_ssim': 0.0,
        'change_ssim': 0.0,
        'window_size': (256, 256), # default window size
        'min_window_size': (24, 24), # default min window size
    }

    def __init__(self, 
             target_path: str, 
             template_path: str, 
             output_path: str, 
             window_size: tuple = (100, 100), 
             settings: dict = {}, 
             gaussian_weights: bool = True, 
             verbose: bool = False, 
             target_band: int = 1, 
             template_band: int = 1, 
             matching_window_strategy: str = 'max_center_size',
             min_window_size: tuple = (24, 24),
             ):
        """
        Initialize the Coregistration class.

        Args:
            target_path (str): Path to the target image.
            template_path (str): Path to the template image.
            output_path (str): Path to save the output image.
            window_size (tuple, optional): Size of the window for coregistration. Defaults to (100, 100).
            settings (dict, optional): Additional settings for coregistration. Defaults to {}.
                Available Settings:
                -------------------
                    max_translation (float): Maximum translation (in meters) allowed for coregistration. Defaults to 1000m.
                    min_translation (float): Minimum translation (in meters) allowed for coregistration. Defaults to -1000m.
            
            gaussian_weights (bool, optional): Whether to use Gaussian weights for SSIM. Defaults to True.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            target_band (int, optional): Band number to use from the target image. Defaults to 1.
            template_band (int, optional): Band number to use from the template image. Defaults to 1.
            matching_window_strategy (str, optional): Strategy to use for finding the matching window. Defaults to 'max_center_size'.
                Two options available:
                ----------------------
                1. max_center_size (default)
                   - It starts with the initial window size at the center of the overlap and shrinks until it finds a matching window or the minimum window size is reached.
                2. use_predetermined_window_size: 
                    uses the window size provided in the window_size parameter. Starts from the corner of the image until it finds a matching window of the specified size
            min_window_size (tuple, optional): Minimum window size for coregistration. Defaults to (24, 24).
            
        """
        self.verbose = verbose

        self.target_band = target_band
        self.template_band = template_band
        
        # Initialize paths
        self.target_path = target_path
        self.original_target_path = target_path # save this location in case we need to reproject the target
        self.original_template_path = template_path
        self.template_path = template_path
        self.output_path = output_path

        # Initialize window size, settings, and gaussian flag
        self.window_size = window_size
        self.settings = settings
        self.gaussian_weights = gaussian_weights  # set this to True if you want to use Gaussian weights for SSIM (good for important pixels at the center of the image)

        # Initialize properties
        self.matching_window_strategy = matching_window_strategy # this is the startegy to use to find the matching window
        # Options for the matching window strategy are:
        # 1. max_center_size: finds the largest window centered at the center of the overlap
        # 2. use_predetermined_window_size: uses the window size provided in the window_size parameter. Starts from the corner of the image until it finds a matching window of the specified size

        # Properties with default values
        self.bounds = (None, None, None, None)

        # track files that need to be removed after coregistration is complete
        self.updated_dtype_file = ""
        self.reprojected_file = ""


        # Window size
        self.min_window_size= min_window_size   # if the window size is less than this, it will not be used

        # Track resolutions for target, template, and current resolution
        self.target_resolution = None
        self.template_resolution = None
        self.current_resolution = None
        self.original_target_resolution = None

        # Track SSIM values
        self.original_ssim = 0
        self.coregistered_ssim = 0

        # Track no data values
        self.target_nodata = self.read_no_data(self.target_path)
        self.template_nodata = self.read_no_data(self.template_path)

        # Track whether the template has to be reprojected
        self.template_reprojected = False

        # Coregistration information
        self.coreg_info = {}
        self.get_coreg_info()

        # Initialize any additional parameters as needed
        self._initialize_resolutions()
        self.set_current_resolution()

    
        # make both the target and template the same data type (needed for histogram matching) (changes them to be the largest data type possible to avoid loss of information)
        self.target_path, self.template_path = self.update_tiff_match_largest_dtype(self.target_path, self.template_path, os.path.join(os.path.dirname(self.output_path), 'changed_dtype.tif'))

        # in this case the target should now have a dtype that matches the template

        if verbose:
            print(f"self.template_reprojected: {self.template_reprojected}")

        # reproject either the target or the template to match the other
        if self.template_reprojected: # if this runs then the template was reprojected
            self.template_path = reproject_to_image(self.target_path, self.template_path, 'reprojected_template.tif')
            self.reprojected_file = self.template_path
        else: # this is genrally what will run and it will reproject the target to match the template
            reprojected_path = output_path.replace('.tif','_reprojected.tif')
            self.target_path = reproject_to_image(self.template_path, self.target_path, reprojected_path)
            self.reprojected_file = reprojected_path  # this file will be deleted after coregistration is complete

        with rasterio.open(self.target_path) as src:
            self.target_dtype = src.dtypes[0]

        with rasterio.open(self.template_path) as src:
            self.template_dtype = src.dtypes[0]

        if self.verbose:
            print(f"Reprojected image saved to: {output_path}")

        max_window_size=get_max_overlap_window_size_in_pixels(self.target_path, self.template_path)
        if self.verbose:
            print(f"Max window size: {max_window_size}")
        if max_window_size <16:
            self.clear_temp_files()
            raise ValueError(f"The overlapping region was smaller than 16 pixels. Coregistraion is not possible. ")
            
        initial_window_size = self.window_size
        if self.matching_window_strategy == 'max_center_size': # finds the largest window starting with the initital window size at the center of the overlap
            try:
                best_point = find_best_starting_point(self.target_path, self.template_path)
                # the bounds are (row_start,col_start,row_end,col_end)
                window_size, best_bounds = get_valid_window_with_fallback(self.target_path, self.template_path, start_point=best_point,initial_window_size=self.window_size,min_window_size=self.min_window_size)
                # if this method fails implement the VERY SLOW but reliable fallback method which has a cache in case its seen it before
                if self.bounds == (None, None, None, None):
                    window_size, best_bounds = find_best_window_in_combined_mask(self.target_path, self.template_path, self.min_window_size[0],self.min_window_size[1], self.window_size[0],self.window_size[1], )
                if verbose:
                    print(f"best_point: {best_point}, window_size: {window_size}, best_bounds: {best_bounds}")
            except Exception as e:
                import traceback
                print(f"Error: {e}")
                traceback.print_exc()
                self.best_bounds = (None, None, None, None)
                if self.verbose:
                    print(f"Error finding the best window starting point. Using the max overlap window size.")
                self.clear_temp_files()
            else: # if the window is found
                self.window_size = window_size
                self.bounds = best_bounds
        elif self.matching_window_strategy == 'use_predetermined_window_size': # finds the first window of the specified size within the overlap
            self.bounds = self.find_matching_bounds(self.target_path, self.template_path)
        else:
            raise ValueError(f"Invalid matching window strategy: {self.matching_window_strategy}")
        
        self.coreg_info.update({'window_size': self.window_size})

        if self.bounds == (None, None, None, None):
            self.coreg_info.update({'description': "No valid matching window was found within the range of {self.min_window_size} to " + str(initial_window_size) + " pixels."})
            self.coreg_info.update({'success': 'False'})
            self.clear_temp_files()
        else:
            # save a figure of the matching region
            if verbose:
                self.write_matching_window_tiffs()
                self.save_matching_region_figure()

    @property
    def default_coreg_info(self):
        """
        Provides the default coregistration information.
        
        Returns:
            dict: A dictionary with default coregistration information values.
        """
        return {
            'shift_x': 0.0,
            'shift_y': 0.0,
            'shift_x_meters': 0.0,
            'shift_y_meters': 0.0,
            'initial_shift_x': 0.0,
            'initial_shift_y': 0.0,
            'error': 0.0,
            'shift_reliability': 0.0,
            'qc': 0,
            'description': '',
            'success': 'False',
            'original_ssim': 0.0,
            'coregistered_ssim': 0.0,
            'change_ssim': 0.0,
            'window_size': (256, 256), # default window size
        }

    def clear_temp_files(self):
        if self.reprojected_file:
            if self.verbose:
                print(f"Removing reprojected file: {self.reprojected_file}")
            os.remove(self.reprojected_file)
        if self.updated_dtype_file:
            if self.verbose:
                print(f"Removing updated dtype file: {self.updated_dtype_file}")
            os.remove(self.updated_dtype_file)

    def update_tiff_match_largest_dtype(self,tiff1_path, tiff2_path, output_path):
        """
        Changes the data type of the smaller data type TIFF to match the larger data type TIFF
        without scaling any values and returns the original paths in the input order.
        
        Parameters:
        tiff1_path (str): Path to the first TIFF file.
        tiff2_path (str): Path to the second TIFF file.
        output_path (str): Path where the TIFF file with updated dtype will be saved.
        
        Returns:
        tuple: Paths to the TIFFs in the order they were passed, with the smaller TIFF's
            path replaced by the updated output path.
        """
        # Open both TIFFs and extract their data types and metadata
        with rasterio.open(tiff1_path) as tiff1, rasterio.open(tiff2_path) as tiff2:
            dtype1 = tiff1.dtypes[0]
            dtype2 = tiff2.dtypes[0]

            # Determine which TIFF has the smaller data type
            dtype1_size = np.dtype(dtype1).itemsize
            dtype2_size = np.dtype(dtype2).itemsize
            
            if dtype1_size > dtype2_size:
                smaller_tiff = tiff2
                larger_dtype = dtype1
                updated_tiff_path = output_path
                unchanged_tiff_path = tiff1_path
            elif dtype1_size < dtype2_size:
                smaller_tiff = tiff1
                larger_dtype = dtype2
                updated_tiff_path = output_path
                unchanged_tiff_path = tiff2_path
            else:
                # If the data types are the same, no changes are needed
                return tiff1_path, tiff2_path

            # Update metadata to match the larger data type
            updated_meta = smaller_tiff.meta.copy()
            updated_meta.update(dtype=larger_dtype)

            # Read the smaller TIFF's data and write it with the new dtype
            with rasterio.open(output_path, 'w', **updated_meta) as dst:
                for i in range(1, smaller_tiff.count + 1):
                    band = smaller_tiff.read(i)
                    dst.write(band.astype(larger_dtype), i)

        # Save the updated path so it can be deleted later
        self.updated_dtype_file = updated_tiff_path

        # Return the paths in the original order
        if dtype1_size > dtype2_size:
            # print(f"tif1 > tif2: {updated_tiff_path}")
            return unchanged_tiff_path, updated_tiff_path
        else:
            # print(f"tif1 <= tif2: {updated_tiff_path}")
            return updated_tiff_path, unchanged_tiff_path

    def write_matching_window_tiffs(self):
        row_start, col_start, row_end, col_end = self.bounds
        print(f"bounds: {row_start, col_start, row_end, col_end}")
        window = rasterio.windows.Window.from_slices((row_start, row_end), (col_start, col_end))
        print(f"window: {window}")
        # save the cropped tiffs
        filename = os.path.basename(self.template_path).split('.')[0] + "_cropped.tif"
        output_dir = os.path.dirname(self.output_path)
        output_path = os.path.join(output_dir, filename)
        print(f"cropped template path : {output_path}")
        write_window_to_tiff(window,self.template_path, output_path)

        filename = os.path.basename(self.target_path).split('.')[0] + "_cropped.tif"
        output_path = os.path.join(output_dir, filename)
        print(f"cropped target path : {output_path}")
        write_window_to_tiff(window,self.target_path, output_path)

    def save_matching_region_figure(self):
        output_dir = os.path.dirname(self.output_path)
        # Save a figure of the matching region
        row_start, col_start, row_end, col_end = self.bounds
        template_band = read_bounds(self.template_path,row_start,col_start,row_end,col_end,band_number=self.template_band)
        target_band = read_bounds(self.target_path,row_start,col_start,row_end,col_end,band_number=self.target_band)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # add the window size to the title
        fig.suptitle(f'Matching Region ({self.window_size[0]}x{self.window_size[1]})\n bounds: {row_start, col_start, row_end, col_end}')

        ax[0].imshow(template_band, cmap='gray')
        ax[0].set_title('Template Image')
        ax[1].imshow(target_band, cmap='gray')
        ax[1].set_title('Target Image')
        plt.tight_layout()
        # get the filename of the output path
        filename = os.path.basename(self.output_path).split('.')[0]
        plt.savefig(os.path.join(output_dir, f'matching_region_({self.window_size}){filename}.png'))
        plt.close()


    def get_coreg_info(self):
        """
        Retrieves and initializes coregistration information.
        This method checks if the coreg_info attribute is an empty dictionary. If it is, it initializes it with default values.
        It then ensures that certain fields within coreg_info are JSON serializable by converting them to float.
        Returns:
            dict: A dictionary containing coregistration information with the following keys:
            - 'shift_x' (float): Shift in the x direction, after being scaled to the target resolution.
            - 'shift_y' (float): Shift in the y direction, after being scaled to the target resolution.
            - 'shift_x_meters' (float): Shift in the x direction in meters.
            - 'shift_y_meters' (float): Shift in the y direction in meters.
            - 'initial_shift_x' (float): Initial shift in the x direction.
            - 'initial_shift_y' (float): Initial shift in the y direction.
            - 'error' (float): Error value.
            - 'qc' (int): Quality control value.
            - 'description' (str): Description of the coregistration.
            - 'success' (str): Success status of the coregistration.
            - 'original_ssim' (float): Original Structural Similarity Index (SSIM).
            - 'coregistered_ssim' (float): Coregistered Structural Similarity Index (SSIM).
            - 'change_ssim' (float): Change in Structural Similarity Index (SSIM).
            - 'window_size' (int): Window size used for coregistration.
        """
        if self.coreg_info == {}:
            self.coreg_info = CoregisterInterface.DEFAULT_COREG_INFO.copy()
            self.coreg_info.update({'window_size': self.window_size})
            self.coreg_info.update({'min_window_size': self.min_window_size})


        # make the shifts json serializeable
        self.coreg_info.update({'shift_x': float(self.coreg_info['shift_x'])})
        self.coreg_info.update({'shift_y': float(self.coreg_info['shift_y'])})
        self.coreg_info.update({'initial_shift_x': float(self.coreg_info['initial_shift_x'])})
        self.coreg_info.update({'initial_shift_y': float(self.coreg_info['initial_shift_y'])})
        self.coreg_info.update({'error': float(self.coreg_info['error'])})
        self.coreg_info.update({'shift_reliability': float(self.coreg_info['shift_reliability'])})
        self.coreg_info.update({'original_ssim': float(self.coreg_info['original_ssim'])})
        self.coreg_info.update({'coregistered_ssim': float(self.coreg_info['coregistered_ssim'])})
        self.coreg_info.update({'change_ssim': float(self.coreg_info['change_ssim'])})
        self.coreg_info.update({'shift_y_meters': float(self.coreg_info['shift_y_meters'])})
        self.coreg_info.update({'shift_x_meters': float(self.coreg_info['shift_x_meters'])})
        self.coreg_info.update({'min_window_size': (int(self.min_window_size[0]),int(self.min_window_size[1]) )})
        self.coreg_info.update({'window_size': (int(self.window_size[0]),int(self.window_size[1]) )})

        return self.coreg_info

    def quality_control_shift(self,shift:tuple,):
        """
        Check if the shift provided in meters is lower than the maximum translation and higher than the minimum translation.
        Returns 1 if the shift passes the quality control, otherwise 0.
        Args:
            shift (tuple)(in meters): A tuple containing the shift values (x, y).
        Returns:
            int: Returns 1 if the shift values pass the quality control, otherwise 0.
        """
        qc = 1

        if 'max_translation' in self.settings:
            if shift[0] > self.settings.get('max_translation') or shift[1] > self.settings.get('max_translation'):
                qc = 0
        if 'min_translation'  in self.settings:
            if shift[0] < self.settings.get('min_translation') or shift[1] < self.settings.get('min_translation'):
                qc = 0

        return qc

    def _initialize_resolutions(self):
        """
        Initializes or calculates the resolutions for the target and template images.

        This method sets the following attributes:
        - target_resolution: The resolution of the target image.
        - original_target_resolution: The original resolution of the target image, 
          which is needed later to scale the shifts in meters back to pixels.
        - template_resolution: The resolution of the template image.

        It uses the `get_resolution` method to obtain the resolutions based on the 
        provided file paths (`target_path` and `template_path`).
        """
        # Method to initialize or calculate the target and template resolutions
        self.target_resolution = self.get_resolution(self.target_path)
        self.original_target_resolution = self.target_resolution   # this is needed later to scale the shifts in meters back to pixels
        self.template_resolution = self.get_resolution(self.template_path)

    def set_current_resolution(self):
        if self.target_resolution is None or self.template_resolution is None:
            self._initialize_resolutions()
        
        # set the current resolution to whichever is lower (aka worse) (REMEMBER THIS MEANS THE VALUE IS HIGHER)
        if self.target_resolution[0] > self.template_resolution[0]:
            if self.verbose:
                print(f"target res {self.target_resolution} >  template {self.template_resolution}")
            self.current_resolution = self.target_resolution  # target resolution is worse
            self.template_reprojected = True # the template has to be reprojected to match the target resolution
        elif self.target_resolution[0] == self.template_resolution[0]:
            if self.verbose:
                print(f"target res {self.target_resolution} ==  template {self.template_resolution}")
            self.current_resolution = self.target_resolution
        else:
            if self.verbose:
                print(f"target res {self.target_resolution} <  template {self.template_resolution}")
            self.current_resolution = self.template_resolution
            
        self.get_coreg_info()

        if self.verbose:
            print(f"Current resolution: {self.current_resolution}")


    def identify_shifts(self):
        """
        Identifies the shifts between the target and template images.
        This method performs the following steps:
        1. Retrieves the bounds of the reprojected target and template images.
        2. Reads the bands from the matching bounds for both the target and template images.
        3. Applies histogram matching to the target image using the template image.
        4. Finds the shift between the template and target images using phase cross-correlation.
        5. Estimates the reliability of the calculated shift.
        6. Converts the shift from pixels to meters.
        7. Performs quality control on the calculated shift.
        8. Updates the coregistration information with the calculated shifts and related data.
        Returns:
            None
        """
        # 1. Get the bounds of the reprojected target and template
        row_start, col_start, row_end, col_end = self.bounds
        if self.verbose:
            print(f"Bounds (row_start,col_start, row_end, col_end  ): {row_start, col_start, row_end, col_end}")
        # read the bands from the matching bounds for the target and template
        template_window = read_bounds(self.template_path,row_start,col_start,row_end,col_end,band_number=self.template_band)
        target_window = read_bounds(self.target_path,row_start,col_start,row_end,col_end,band_number=self.target_band)
        # 2. apply histogram matching
        target_window = exposure.match_histograms(target_window, template_window)
        # 3. Find the shift between the template and target images
        initial_shift, error, diffphase = phase_cross_correlation(
            template_window, target_window, upsample_factor=100,
        )
        # turn the initial shift to float so that it can be saved to file
        initial_shift = float(initial_shift[0]), float(initial_shift[1])
        # estimate shift reliability
        shift_reliability = calculate_shift_reliability(template_window, target_window)
        # convert the shift to meters ( shift is in Y X format in pixels)
        shift_meters = (initial_shift[0]*self.current_resolution[1], initial_shift[1]*self.current_resolution[0])

        # 4. This is where quality control would go @ todo
        shift_qc = self.quality_control_shift(shift_meters)
        if not shift_qc:
            shift = (0,0)
        else:
            shift = initial_shift

        # convert the meters to the target resolution in pixels
        shift = (shift_meters[0]/self.original_target_resolution[1], shift_meters[1]/self.original_target_resolution[0])

        self.coreg_info.update({        
                'shift_x': shift[1],
                'shift_y': shift[0],
                'shift_x_meters': shift_meters[1],
                'shift_y_meters': shift_meters[0],
                'initial_shift_x':initial_shift[1],
                'initial_shift_y': initial_shift[0],
                'current_resolution': self.current_resolution,
                'target_resolution': self.original_target_resolution,
                'error': error,
                'shift_reliability': shift_reliability, 
                'qc': shift_qc,
                'description':'successfully coregistered' if shift_qc else 'failed : shift exceeded max or min translation',
        })


    def find_matching_bounds(self,target_path, template_path):
        # Reworked function to find matching bounds without NoData pixels
        with rasterio.open(target_path) as src1, rasterio.open( template_path) as src2:
            # Read data and metadata
            data1 = src1.read(1)
            data2 = src2.read(1)
            nodata1 = src1.nodata
            nodata2 = src2.nodata

            # Define overlapping window between the two TIFFs
            overlap_window = src1.window(*src1.bounds).intersection(src2.window(*src2.bounds))

            # Convert to row/col coordinates for consistent box size
            row_start, col_start, row_end, col_end = map(int, overlap_window.flatten())

            # Check for valid window_size regions within the overlap
            for row in range(row_start, row_end - self.window_size[0] + 1):
                for col in range(col_start, col_end - self.window_size[1] + 1):
                    # Read the corresponding data for both TIFFs
                    data1_box = data1[row:row + self.window_size[0], col:col + self.window_size[1]]
                    data2_box = data2[row:row + self.window_size[0], col:col + self.window_size[1]]
                    
                    # Check for NoData in both TIFFs
                    if (data1_box != nodata1).all() and (data2_box != nodata2).all():
                        if self.verbose:
                            print(f"Found valid region at row {row}, col {col} , row + self.window_size[0] : {row + self.window_size[0]} , col + self.window_size[1] : {col + self.window_size[1]}")
                        return row, col, row + self.window_size[0], col + self.window_size[1]

        return None, None, None, None  


    def apply_shift(self):
        # Apply the calculated shifts to the target image
        apply_shift_to_tiff(self.original_target_path, self.output_path, (self.coreg_info['shift_y'], self.coreg_info['shift_x']),verbose=self.verbose)


    def get_resolution(self, image_path):
        """REturns the resolution of the image"""
        # Use rasterio to read the resolution of the image
        with rasterio.open(image_path) as src:
            resolution = src.res  # Returns a tuple (pixel size in x, pixel size in y)
        return resolution

    def read_no_data(self, image_path):
        # Read the no data value of the image
        with rasterio.open(image_path) as src:
            no_data = src.nodata
        return no_data

    def histogram_match_to_image(self,target_band, template_band):
        return exposure.match_histograms(target_band, template_band)

    def calc_original_ssim(self):
        # Calculate the SSIM of the original target image
        # read the bounds of the matching region
        row_start, col_start, row_end, col_end = self.bounds
        self.original_ssim = calc_ssim_in_bounds(self.template_path, self.target_path,self.template_nodata,self.target_nodata,row_start,col_start,row_end,col_end,self.gaussian_weights,target_band_number=self.target_band,template_band_number=self.template_band)
        
        self.coreg_info.update({'original_ssim': self.original_ssim})
        
        return self.original_ssim
    
    def calc_coregistered_ssim(self):
        # Calculate the SSIM of the coregistered target image
        if not os.path.exists(self.output_path):
            raise Exception("Coregistered image does not exist. Run coregister() first.")
        
        output_path = self.output_path
        
        # reproject the target to match the template ( if the target was reprojected)
        if not self.template_reprojected:
            output_path = reproject_to_image(self.template_path, output_path, 'reprojected_target.tif')
        
        # read the bounds of the matching region
        row_start, col_start, row_end, col_end = self.bounds
        target_no_data = self.read_no_data(output_path)
        self.coregistered_ssim = calc_ssim_in_bounds(self.template_path, output_path,self.template_nodata,target_no_data,row_start,col_start,row_end,col_end,self.gaussian_weights,target_band_number=self.target_band,template_band_number=self.template_band)
        
        self.coreg_info.update({'coregistered_ssim': self.coregistered_ssim})

        # remove the reprojected coregistered target
        os.remove('reprojected_target.tif')
        
        return self.coregistered_ssim

    def calc_improvement(self):
        # Calculate the improvement in SSIM after coregistration
        improvement = self.coregistered_ssim - self.original_ssim
        self.coreg_info.update({'change_ssim': improvement})
        self.coreg_info.update({'success': "True" if improvement>0 else "False"}) 
        return improvement

    def calc_ssim(self):
        # Calculate the SSIM of the original target image
        self.calc_original_ssim()
        # Calculate the SSIM of the coregistered target image
        self.calc_coregistered_ssim()

    def coregister(self):
        # Coregistration logic
        self.identify_shifts()
        self.calc_original_ssim()
        self.apply_shift()        # if the shift exceeded the min or max translation then the shift will be 0,0
        if self.coreg_info['qc']:
            self.calc_coregistered_ssim()
            self.calc_improvement()
        
        self.clear_temp_files()
        #@todo make sure to remove the reprojected template and target images including the file that had its dtype changed

def coregister_single(target_path, template_path, output_path, **kwargs):
    """
    Coregisters a single image with a template image.

    Parameters:
    - target_path (str): Path to the target image to be coregistered.
    - template_path (str): Path to the template image used for coregistration.
    - output_path (str): Path where the coregistered image will be saved.
    
    Keyword Arguments (kwargs):
    - WINDOW_SIZE (int): Size of the window used for matching.
    - settings (dict): A dictionary containing settings such as 'max_translation' and 'min_translation'.
    - matching_window_strategy (str): Strategy for window matching (e.g., 'max_overlap' or 'max_center_size').
    - verbose (bool): If True, prints detailed logs of the coregistration process.
    - Any other settings can also be passed and will be processed dynamically.

    Returns:
    None
    """
    # make sure the target and template path are not the same
    if target_path == template_path:
        print(f"Skipping {os.path.basename(target_path)} because the target and template paths are the same.")
        new_result = {
            os.path.basename(target_path): {CoregisterInterface.DEFAULT_COREG_INFO}
        }
        return new_result

    # Extract specific parameters with defaults
    window_size = kwargs.get('window_size', 100)  # Default value as an example
    settings = kwargs.get('settings', {})
    matching_window_strategy = kwargs.get('matching_window_strategy', 'max_center_size')
    verbose = kwargs.get('verbose', False)
    try:
        coreg = CoregisterInterface(target_path=target_path, template_path=template_path, output_path=output_path,window_size=window_size,settings=settings, verbose=verbose,matching_window_strategy=matching_window_strategy)
        if 'no valid matching window found' in coreg.get_coreg_info()['description'] or coreg.bounds == (None, None, None, None):
            print(f"Skipping {os.path.basename(target_path)} due to no valid matching window found.")
            new_result = {
                os.path.basename(target_path): coreg.get_coreg_info()
            }
            return new_result
        coreg.coregister()
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
    return new_result

# Dev notes: See coregister_class_tester.py for a test of this class

# Settings
# WINDOW_SIZE=(100,100)
# window_size_str= f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}"

# # Paths
# unregistered_directory = r'C:\3_code_from_dan\6_coregister_implementation_coastseg\3_processed_coastseg\RGB\S2'
# template_path = r"C:\3_code_from_dan\2_coregistration_unalakleet\unalakeet\template\L9\regular\2023-08-01-22-02-10_L9_ID_vaa1_datetime05-29-24__04_35_57_processed_model_format.tif"

# # Target path
# tif_files = glob.glob(os.path.join(unregistered_directory, '*.tif'))
# # Ensure there is at least one TIFF file
# # if tif_files:
# #     target_path = tif_files[0]
# #     print(f"Target path: {target_path}")
# # else:
# #     print("No TIFF files found in the directory.")
# filename = r'2021-05-19-22-18-07_S2_TOAR_model_format.tif'
# target_path = os.path.join(unregistered_directory, filename)


# # output directory
# coregistered_directory = r"C:\3_code_from_dan\2_coregistration_unalakleet\unalakeet\1_coregistered_S2_images_with_window_"+window_size_str+"new_mean_ssim_scores_fix_bounds"
# os.makedirs(coregistered_directory, exist_ok=True)

# output_filename = os.path.basename(target_path).split('.')[0] + '_coregistered.tif'
# output_path = os.path.join(coregistered_directory, output_filename)
# print(f"Output path: {output_path}")

# settings = {
#     'max_translation': 15,
#     'min_translation': -15,
# }

# # begin coregistration
# coreg = CoregisterInterface(target_path=target_path, template_path=template_path, output_path='output.tif',window_size=WINDOW_SIZE,settings=settings, verbose=True)
# print(coreg.scaling_factor)
# print(coreg.current_resolution)
# print(coreg.template_reprojected)
# print(coreg.bounds)

# coreg.coregister()
# print(coreg.get_coreg_info())
# assert coreg.get_coreg_info()['qc'] == 1
# assert coreg.get_coreg_info()['initial_shift_x'] == -0.55
# assert coreg.get_coreg_info()['initial_shift_y'] == -0.29
# assert coreg.get_coreg_info()['shift_x'] == -0.8250000000000001
# assert coreg.get_coreg_info()['shift_y'] == -0.43499999999999994
# assert coreg.get_coreg_info()['original_ssim'] == 0.6001147326519192
# # apply shift failed coreg score does not match
# assert coreg.get_coreg_info()['coregistered_ssim'] == 0.6041479874381226
# assert coreg.get_coreg_info()['change_ssim'] == 0.004033254786203422
# assert coreg.get_coreg_info()['success'] == "True"