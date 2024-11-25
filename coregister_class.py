import rasterio
import numpy as np
import os
import math
import json
import rasterio
import glob


from rasterio.transform import Affine
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

def normalize(array: np.ndarray) -> np.ndarray:
    minval = np.min(array)
    maxval = np.max(array)
    # avoid zerodivision
    if maxval == minval:
        maxval += 1e-5
    return ((array - minval) / (maxval - minval)).astype(np.float64)

def apply_shift_to_tiff(target_path, output_path, shift:np.ndarray,verbose=False):
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

def apply_shift_to_tiff_old_method(target_path, output_path, shift:np.ndarray,verbose=False):
    if verbose:
        print(f"Applying shift {shift}")
    with rasterio.open(target_path) as src:
        meta = src.meta.copy()

        # transform_shift = Affine.translation(shift[1], shift[0])  # x=shift[1], y=shift[0]
        # meta['transform'] = src.transform * transform_shift

        # if verbose:
        #     print(f"Original transform:\n{src.transform}")
        #     print(f"Updated transform:\n{meta['transform']}")


        meta.update({
            'driver': 'GTiff',
            'height': src.height,
            'width': src.width,
            'transform': src.transform,
            'crs': src.crs
        })
        with rasterio.open(output_path, 'w', **meta) as dst:
            for i in range(1, src.count + 1):
                band = src.read(i)
                
                transformed_band = scipy_shift(band, shift=shift, mode='reflect', cval=0.0, order=3)
                # get the data type of the original band
                dtype = band.dtype
                dst.write(transformed_band.astype(dtype), i)
                # dst.write(band.astype(dtype), i)



def masked_ssim(reference, target, ref_no_data_value=0,target_no_data_value=0,gaussian_weights=True):
    # Create a mask for valid (non-no-data) pixels in both images
    mask = (reference != ref_no_data_value) & (target != target_no_data_value)
    

    reference_masked = np.where(mask, reference, 0)
    target_masked = np.where(mask, target, 0)
    
    # Compute SSIM only over the masked region (valid data)
    ssim_score, ssim_image = ssim(
        normalize(reference_masked), # could also try np.ma.masked_equal(reference, ref_no_data_value)
        normalize(target_masked),   # could also try np.ma.masked_equal(target, target_no_data_value)
        data_range=1, # our pixels are in the range of 0-255
        gaussian_weights=True, # Use Gaussian weights for windowing (gives more importance to center pixels)
        use_sample_covariance=False, # Use population covariance for SSIM
        full=True
    )

    # Mask out the no-data regions in the SSIM image
    ssim_image[~mask] = np.nan  # Set no-data regions to NaN

    # Compute the mean SSIM score over valid regions only
    mean_ssim = np.nanmean(ssim_image)  # Use np.nanmean to ignore NaN values


    return mean_ssim

# ssim(normalize(np.ma.masked_equal(self.matchWin[:],
#                                                            self.matchWin.nodata)),
#                               normalize(np.ma.masked_equal(self.otherWin[:],
#                                                            self.otherWin.nodata)),
#                               data_range=1)

def calc_ssim_in_bounds(template_path, target_path,template_nodata,target_nodata,row_start,col_start,row_end,col_end,gaussian_weights=True,target_band_number=1,template_band_number=1):
    # read the bands from the matching bounds
    template_band = read_bounds(template_path,row_start,col_start,row_end,col_end,band_number=template_band_number)
    target_band = read_bounds(target_path,row_start,col_start,row_end,col_end,band_number=target_band_number)
     
    # Compute SSIM score
    ssim_score = masked_ssim(template_band, target_band, template_nodata, target_nodata, gaussian_weights)
    return ssim_score

def read_bounds(tif_path, row_start, col_start, row_end, col_end,band_number=1):
    with rasterio.open(tif_path) as src:
        # Create a window using the bounds
        window = rasterio.windows.Window.from_slices((row_start, row_end), (col_start, col_end))
        # Read only the first band within the window
        data_box = src.read(band_number, window=window)
        return data_box

def reproject_to_image(template_path, target_path, output_path):
    """
    Reprojects a target image to match the spatial resolution, transform, CRS, and nodata values of a template image.
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
        ref_nodata = ref.nodata
        ref_dtype = ref.dtypes[0]  # Get the data type of the reference image

    # Open the target image and update metadata to match reference
    with rasterio.open(target_path) as target:
        target_meta = target.meta.copy()
        target_meta.update({
            'driver': 'GTiff',
            'height': ref_height,
            'width': ref_width,
            'transform': ref_transform,
            'crs': ref_crs,
            'nodata': ref_nodata,
            'dtype': ref_dtype  # Set the data type to match the reference image
        })

        # Check and replace invalid nodata value
        if ref_nodata is None or ref_nodata == float('-inf'):
            ref_nodata = 0  # or another valid nodata value for uint16

        target_meta['nodata'] = ref_nodata

        # Create output file and perform resampling
        with rasterio.open(output_path, 'w', **target_meta) as dst:
            for i in range(1, target.count + 1):  # Iterate over each band
                target_band = target.read(i)
                # replace the nodata value in the target image
                target_band[target_band == target.nodata] = ref_nodata
                reproject(
                    source=target_band,
                    destination=rasterio.band(dst, i),
                    src_transform=target.transform,
                    src_crs=target.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.cubic,
                    src_nodata=ref_nodata,
                    dst_nodata=ref_nodata
                )

    return output_path

def write_window_to_tiff(window,src_path, output_path):
    with rasterio.open(src_path) as src:
        data = src.read(window=window)
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
    def __init__(self, target_path, template_path, output_path, window_size: tuple=100, settings: dict={}, gaussian_weights: bool=True,verbose: bool = False,target_band: int = 1, template_band: int = 1):
        self.verbose = verbose

        self.target_band = target_band
        self.template_band = template_band
        
        # Initialize paths
        self.target_path = target_path
        self.original_target_path = target_path # save this location in case we need to reproject the target
        self.template_path = template_path
        self.output_path = output_path

        # Initialize window size, settings, and gaussian flag
        self.window_size = window_size
        self.settings = settings
        self.gaussian_weights = gaussian_weights  # set this to True if you want to use Gaussian weights for SSIM (good for important pixels at the center of the image)

        # Properties with default values
        self.scaling_factor: float = 1
        self.bounds: tuple = None

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
        self.set_scaling_factor()

        # reproject either the target or the template to match the other
        if self.template_reprojected:
            self.template_path = reproject_to_image(self.target_path, self.template_path, 'reprojected_template.tif')
        else: # this is genrally what will run
            reprojected_path = output_path.replace('.tif','_reprojected.tif')
            self.target_path = reproject_to_image(self.template_path, self.target_path, reprojected_path)

        if self.verbose:
            print(f"Reprojected image saved to: {output_path}")

        self.template_nodata = self.read_no_data(self.template_path)
        self.target_nodata = self.read_no_data(self.target_path)

        # get the bounds of the matching region
        self.bounds = self.find_matching_bounds(self.target_path, self.template_path)
        if self.bounds == (None, None, None, None):
            self.coreg_info.update({'description': 'no valid matching window found of size '+str(self.window_size)})
            self.coreg_info.update({'success': 'False'})
        else:
            # save a figure of the matching region
            self.write_matching_window_tiffs()
            self.save_matching_region_figure()

        # self.bounds = self.get_valid_window(self.target_path, self.template_path, self.window_size)



    def write_matching_window_tiffs(self):
        row_start, col_start, row_end, col_end = self.bounds
        window = rasterio.windows.Window.from_slices((row_start, row_end), (col_start, col_end))
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
        if self.coreg_info == {}:
            self.coreg_info = {
                'shift_x': 0,
                'shift_y': 0,
                'shift_x_meters': 0,
                'shift_y_meters': 0,
                'initial_shift_x':0,
                'initial_shift_y': 0,
                'error': 0,
                'qc': 0,
                'description':'',
                'success': 'False',
                'original_ssim': 0,
                'coregistered_ssim': 0,
                "change_ssim": 0,
            }

        # make the shifts json serializeable
        self.coreg_info.update({'shift_x': float(self.coreg_info['shift_x'])})
        self.coreg_info.update({'shift_y': float(self.coreg_info['shift_y'])})
        self.coreg_info.update({'initial_shift_x': float(self.coreg_info['initial_shift_x'])})
        self.coreg_info.update({'initial_shift_y': float(self.coreg_info['initial_shift_y'])})
        self.coreg_info.update({'error': float(self.coreg_info['error'])})
        self.coreg_info.update({'original_ssim': float(self.coreg_info['original_ssim'])})
        self.coreg_info.update({'coregistered_ssim': float(self.coreg_info['coregistered_ssim'])})
        self.coreg_info.update({'change_ssim': float(self.coreg_info['change_ssim'])})
        self.coreg_info.update({'shift_y_meters': float(self.coreg_info['shift_y_meters'])})
        self.coreg_info.update({'shift_x_meters': float(self.coreg_info['shift_x_meters'])})

        return self.coreg_info

    def quality_control_shift(self,shift):
        qc = 1

        if 'max_translation' in self.settings:
            if shift[0] > self.settings.get('max_translation') or shift[1] > self.settings.get('max_translation'):
                qc = 0
        if 'min_translation'  in self.settings:
            if shift[0] < self.settings.get('min_translation') or shift[1] < self.settings.get('min_translation'):
                qc = 0

        return qc

    def _initialize_resolutions(self):
        # Method to initialize or calculate the target and template resolutions
        self.target_resolution = self.get_resolution(self.target_path)
        self.original_target_resolution = self.target_resolution   # this is needed later to scale the shifts in meters back to pixels
        self.template_resolution = self.get_resolution(self.template_path)

    def set_scaling_factor(self):
        if self.target_resolution is None or self.template_resolution is None:
            self._initialize_resolutions()
        
        # set the current resolution to whichever is lower (aka worse)
        if self.target_resolution[0] > self.template_resolution[0]:
            if self.verbose:
                print(f"target res {self.target_resolution} >  template {self.template_resolution}")
            self.current_resolution = self.target_resolution  # target resolution is worse
            self.scaling_factor =  1
            self.template_reprojected = True # the template has to be reprojected to match the target resolution
        elif self.target_resolution[0] == self.template_resolution[0]:
            if self.verbose:
                print(f"target res {self.target_resolution} ==  template {self.template_resolution}")
            self.current_resolution = self.target_resolution
            self.scaling_factor = 1
        else:
            if self.verbose:
                print(f"target res {self.target_resolution} <  template {self.template_resolution}")
            self.current_resolution = self.template_resolution
            self.scaling_factor = self.template_resolution[0] / self.target_resolution[0]
            
        self.get_coreg_info()
        self.coreg_info.update({'scaling_factor': self.scaling_factor})

        if self.verbose:
            print(f"Scaling factor: {self.scaling_factor} and current resolution: {self.current_resolution}")


    def identify_shifts(self):
        # 1. Get the bounds of the reprojected target and template
        row_start, col_start, row_end, col_end = self.bounds
        # read the bands from the matching bounds for the target and template
        template_window = read_bounds(self.template_path,row_start,col_start,row_end,col_end,band_number=self.template_band)
        target_window = read_bounds(self.target_path,row_start,col_start,row_end,col_end,band_number=self.target_band)
        # 2. apply histogram matching
        target_window = exposure.match_histograms(target_window, template_window)
        # 3. Find the shift between the template and target images
        initial_shift, error, diffphase = shift, error, diffphase = phase_cross_correlation(
            template_window, target_window, upsample_factor=100,
        )



        # 4. This is where quality control would go @ todo
        shift_qc = self.quality_control_shift(initial_shift)

        if not shift_qc:
            shift = (0,0)
        else:
            shift = initial_shift

        # convert the shift to meters ( shift is in Y X format in pixels)
        # shift_meters = (shift[1]*self.current_resolution[0], shift[0]*self.current_resolution[1])

        shift_meters = (shift[0]*self.current_resolution[1], shift[1]*self.current_resolution[0])

        # convert the meters to the target resolution in pixels
        shift = (shift_meters[0]/self.original_target_resolution[1], shift_meters[1]/self.original_target_resolution[0])

        # get the shift_x and shift_y scaled if the target was reprojected
        # if not self.template_reprojected:
        #     shift = shift[0]*self.scaling_factor, shift[1]*self.scaling_factor

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
                    # print(f"Checking row {row}, col {col}")
                    # print(f"row + self.window_size[0] : {row + self.window_size[0]}")
                    # print(f"col + self.window_size[1] : {col + self.window_size[1]}")

                    # Read the corresponding data for both TIFFs
                    data1_box = data1[row:row + self.window_size[0], col:col + self.window_size[1]]
                    data2_box = data2[row:row + self.window_size[0], col:col + self.window_size[1]]
                    
                    # Check for NoData in both TIFFs
                    if (data1_box != nodata1).all() and (data2_box != nodata2).all():
                        if self.verbose:
                            print(f"Found valid region at row {row}, col {col} , row + self.window_size[0] : {row + self.window_size[0]} , col + self.window_size[1] : {col + self.window_size[1]}")
                        return row, col, row + self.window_size[0], col + self.window_size[1]

        print(f"No valid {self.window_size} region found without NoData pixels.")
        self.coreg_info.update({'description': f'no valid matching window found of size '+str(self.window_size)})
        # raise Exception(f"No valid region found without NoData pixels of the specified window size {self.window_size}.")
        return None, None, None, None  


    def apply_shift(self):
        # Apply the calculated shifts to the target image
        apply_shift_to_tiff(self.original_target_path, self.output_path, (self.coreg_info['shift_y'], self.coreg_info['shift_x']),verbose=self.verbose)
        # apply_shift_to_tiff_old_method(self.original_target_path, self.output_path, (self.coreg_info['shift_y'], self.coreg_info['shift_x']),verbose=self.verbose)


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