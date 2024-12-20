import json
import pandas as pd
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt

# Filter functions
# 1. Filer

def plot_shifts_with_outliers(df: pd.DataFrame, filename: str, z_threshold: float = 1.5):
    """
    Create a scatter plot of shifts with outliers highlighted and save it to a file.

    Args:
        df (pd.DataFrame): DataFrame containing 'shift_x' and 'shift_y' columns.
        filename (str): The name of the file to save the plot.
        z_threshold (float, optional): The z-score threshold to identify outliers. Defaults to 1.5.
    """

    # Extract shifts
    x_shifts = df['shift_x']
    y_shifts = df['shift_y']

    # get only the values not the row index
    x_shifts = x_shifts.values
    y_shifts = y_shifts.values


    # Calculate mean and standard deviation
    mu_x, sigma_x = np.mean(x_shifts), np.std(x_shifts)
    mu_y, sigma_y = np.mean(y_shifts), np.std(y_shifts)


    # Compute z-scores
    z_x = (x_shifts - mu_x) / sigma_x
    z_y = (y_shifts - mu_y) / sigma_y

    # Compute combined z-score

    z_combined = np.sqrt(z_x**2 + z_y**2)

    # Identify outliers
    outliers = z_combined > z_threshold

    # Create the scatter plot
    plt.figure(figsize=(10, 6))

    # Plot non-outliers in green
    plt.scatter(
        x_shifts[~outliers],
        y_shifts[~outliers],
        color='green',
        label='Non-Outliers',
        alpha=0.7
    )

    # Plot outliers in red
    plt.scatter(
        x_shifts[outliers],
        y_shifts[outliers],
        color='red',
        label='Outliers',
        alpha=0.7
    )

    # Add labels, legend, and grid
    plt.title("Scatter Plot of Shifts with Outliers Highlighted")
    plt.xlabel("Shift X")
    plt.ylabel("Shift Y")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid(alpha=0.3)

    # Save the plot to a file
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

    # Close the plot to free memory
    plt.close()

def identify_and_plot_outliers(df: pd.DataFrame, z_threshold: float = 2, plot: bool = True, plot_filename: str = None):
    """
    Identify outliers based on combined z-scores and optionally plot the z-scores.

    Args:
        df (pd.DataFrame): DataFrame containing 'shift_x' and 'shift_y' columns.
        z_threshold (float, optional): The z-score threshold to identify outliers. Defaults to 2.
        plot (bool, optional): Whether to plot the combined z-scores. Defaults to True.
        plot_filename (str, optional): The filename to save the plot. If None, the plot will not be saved.

    Returns:
        np.ndarray: Boolean array indicating which points are outliers.
        list: List of filenames (row indices) of the outliers.
        np.ndarray: Combined z-scores for each point.
    """
    # Extract shifts
    x_shifts = df['shift_x']
    y_shifts = df['shift_y']

    # Calculate mean and standard deviation
    mu_x, sigma_x = np.mean(x_shifts), np.std(x_shifts)
    mu_y, sigma_y = np.mean(y_shifts), np.std(y_shifts)

    # Compute z-scores
    z_x = (x_shifts - mu_x) / sigma_x
    z_y = (y_shifts - mu_y) / sigma_y

    # Compute combined z-score
    z_combined = np.sqrt(z_x**2 + z_y**2)

    # Identify outliers
    outliers = z_combined > z_threshold

    # Get filenames of outliers
    outlier_filenames = df.index[outliers].tolist()

    # Print results
    print(f"Number of Outliers: {np.sum(outliers)}")
    print("Outlier Indices:", np.where(outliers)[0])
    print("Outlier Filenames:", outlier_filenames)

    # Optional: Plot z_combined
    if plot:
        plt.hist(z_combined, bins=20, alpha=0.7)
        plt.axvline(z_threshold, color='red', linestyle='--', label=f'Outlier Threshold (z={z_threshold})')
        plt.title("Combined Z-Scores")
        plt.xlabel("z_combined")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.3)
        
        if plot_filename:
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
        else:
            plt.show()

        # Close the plot to free memory
        plt.close()

    return outliers, outlier_filenames, z_combined

def filter_by_z_score(df: pd.DataFrame, z_threshold: float = 2, combined_z_plot_filename: str = 'combined_z_scores.png', shifts_plot_filename: str = 'plot_outlier_shifts.png') -> pd.DataFrame:
    """
    Identify and plot outliers, add combined z-scores to the DataFrame, and plot the shifts.

    Args:
        df (pd.DataFrame): DataFrame containing 'shift_x' and 'shift_y' columns.
        z_threshold (float, optional): The z-score threshold to identify outliers. Defaults to 2.
        combined_z_plot_filename (str, optional): The filename to save the combined z-scores plot. Defaults to 'combined_z_scores.png'.
        shifts_plot_filename (str, optional): The filename to save the shifts plot. Defaults to 'plot_outlier_shifts.png'.

    Returns:
        pd.DataFrame: The updated DataFrame with combined z-scores added as a column.
    """
    # Initialize filter_passed and filter_description columns if they don't exist
    if 'filter_passed' not in df.columns:
        df['filter_passed'] = True
    if 'filter_description' not in df.columns:
        df['filter_description'] = ''

    # Create a mask for rows where filter_passed is True
    filter_mask = df['filter_passed']

    # Identify and plot outliers only for rows where filter_passed is True
    outliers, outlier_filenames, z_combined = identify_and_plot_outliers(df, z_threshold=z_threshold, plot=True, plot_filename=combined_z_plot_filename)
    
    print(f"outliers: {outliers}")

    # Add the combined z-scores as a column to the DataFrame
    df.loc[filter_mask, 'z_combined'] = z_combined
    
    # Update filter_passed and filter_description based on z_combined
    outlier_indices = df[filter_mask].index[outliers]
    df.loc[outlier_indices, 'filter_passed'] = False
    df.loc[outlier_indices, 'filter_description'] += f'z score exceeded z threshold ({z_threshold}); '

    print("Outliers identified:")
    print(df.loc[outlier_indices, ['z_combined', 'filter_passed', 'filter_description']])


    # Plot the shifts and color them red/green based on whether they are outliers
    plot_shifts_with_outliers(df, shifts_plot_filename, z_threshold=z_threshold)
    
    return df

def filter_shifts_by_range(df: pd.DataFrame, min_shift_meters: tuple, max_shift_meters: tuple) -> pd.DataFrame:
    """
    Filter the DataFrame based on the given ranges for shift_x_meters and shift_y_meters.

    Args:
        df (pd.DataFrame): DataFrame containing 'shift_x_meters' and 'shift_y_meters' columns.
        min_shift_meters (tuple): Minimum allowed values for shift_x_meters and shift_y_meters.
        max_shift_meters (tuple): Maximum allowed values for shift_x_meters and shift_y_meters.

    Returns:
        pd.DataFrame: The updated DataFrame with 'filter_passed' and 'filter_description' columns.
    """
    # Initialize filter_passed and filter_description columns if they don't exist
    if 'filter_passed' not in df.columns:
        df['filter_passed'] = True
    if 'filter_description' not in df.columns:
        df['filter_description'] = ''

    # Filter only the rows where filter_passed is True
    filter_mask = df['filter_passed']

    # Filter based on shift_x_meters
    df.loc[filter_mask & (df['shift_x_meters'] < min_shift_meters[0]), 'filter_passed'] = False
    df.loc[filter_mask & (df['shift_x_meters'] < min_shift_meters[0]), 'filter_description'] += f'shift_x_meters below {min_shift_meters[0]}; '
    df.loc[filter_mask & (df['shift_x_meters'] > max_shift_meters[0]), 'filter_passed'] = False
    df.loc[filter_mask & (df['shift_x_meters'] > max_shift_meters[0]), 'filter_description'] += f'shift_x_meters exceeded {max_shift_meters[0]}; '

    # Filter based on shift_y_meters
    df.loc[filter_mask & (df['shift_y_meters'] < min_shift_meters[1]), 'filter_passed'] = False
    df.loc[filter_mask & (df['shift_y_meters'] < min_shift_meters[1]), 'filter_description'] += f'shift_y_meters below {min_shift_meters[1]}; '
    df.loc[filter_mask & (df['shift_y_meters'] > max_shift_meters[1]), 'filter_passed'] = False
    df.loc[filter_mask & (df['shift_y_meters'] > max_shift_meters[1]), 'filter_description'] += f'shift_y_meters exceeded {max_shift_meters[1]}; '

    return df

def create_dataframe_with_satellites(results:dict):
    """
    Create a DataFrame from a dictionary containing transformation results.
    
    Expected JSON format:
    {
        'L8': { 
                "filename1.tif": {
                "shift_x": 10.0,
                "shift_y": 20.0,
                "shift_x_meters": 100.0,
                "shift_y_meters": 200.0,
                "ssim": 0.95
                },
                "filename2.tif": {
                    "shift_x": -5.0,
                    "shift_y": 15.0,
                    "shift_x_meters": -50.0,
                    "shift_y_meters": 150.0,
                    "ssim": 0.92
                },
        },
        'L9': { 
                "filename1.tif": {
                "shift_x": 10.0,
                "shift_y": 20.0,
                "shift_x_meters": 100.0,
                "shift_y_meters": 200.0,
                "ssim": 0.95
                },
                "filename2.tif": {
                    "shift_x": -5.0,
                    "shift_y": 15.0,
                    "shift_x_meters": -50.0,
                    "shift_y_meters": 150.0,
                    "ssim": 0.92
                },
        }
        'settings':{
            'max_translation': 1000,
            'min_translation': -1000,
            'window_size': [256, 256],}
        ...
    }

    Args:
       results(dict)

    Returns:
        pd.DataFrame: The DataFrame created from the JSON file.
    
    """

    data = []
    for satellite, files in results.items():
        if satellite == "settings":
            continue  # Skip the 'settings' section
        for filename, attributes in files.items():
            print(f"filename: {filename}")
            print(f"attributes: {attributes}")
            attributes['filename'] = filename
            attributes['satellite'] = satellite
            data.append(attributes)

    # Create the DataFrame columns should be
    #  shift_x, shift_y, shift_x_meters, shift_y_meters, ssim, satellite
    # with filename as the index
    df = pd.DataFrame(data)
    # set the file name as the index
    df.set_index('filename', inplace=True)
    return df

def apply_filtering(results, output_path:str, settings:dict):
    """
    Processes transformation results from a JSON file, filters and identifies outliers,
    and saves the filtered DataFrame to a new CSV file.

    Args:
        results(dict): Dictionary containing transformation results.
        Either in the format
        {
            'L8': {
                'filename1.tif': {'shift_x': 10.0, 'shift_y': 20.0, 'shift_x_meters': 100.0, 'shift_y_meters': 200.0, 'ssim': 0.95},
                'filename2.tif': {'shift_x': -5.0, 'shift_y': 15.0, 'shift_x_meters': -50.0, 'shift_y_meters': 150.0, 'ssim': 0.92}
            },
            'L9': {
                'filename1.tif': {'shift_x': 10.0, 'shift_y': 20.0, 'shift_x_meters': 100.0, 'shift_y_meters': 200.0, 'ssim': 0.95},
            },
            'settings': {'max_translation': 1000, 'min_translation': -1000, 'window_size': [256, 256]}
        }
        or
        {
            'filename1.tif': {'shift_x': 10.0, 'shift_y': 20.0, 'shift_x_meters': 100.0, 'shift_y_meters': 200.0, 'ssim': 0.95},
            'filename2.tif': {'shift_x': -5.0, 'shift_y': 15.0, 'shift_x_meters': -50.0, 'shift_y_meters': 150.0, 'ssim': 0.92},
            'settings': {'max_translation': 1000, 'min_translation': -1000, 'window_size': [256, 256]}
        }   
        output_path (str): Path to the transformation results JSON file.
        settings (dict): A dictionary with the following keys:
            - 'min_shift_meters' (tuple(float)): Minimum shift distance for filtering for x and y
                Example: (-200, -200) for x and y respectively.
            - 'max_shift_meters' (tuple(float)): Maximum shift distance for filtering for x and y
                Example: (200, 200) for x and y respectively.
            - 'z_threshold' (float): Z-score threshold for outlier detection. Set to None to disable.
                This z score is the combined z score of shift_x and shift_y.
                combined_z_score = sqrt(z_x^2 + z_y^2)

    Returns:
        str: Path to the saved filtered CSV file.
    """
    # Unpack settings
    min_shift_meters = settings.get('min_shift_meters', float('-inf'))
    max_shift_meters = settings.get('max_shift_meters', float('inf'))
    z_threshold = settings.get('z_threshold', None)


    try:
        df = create_dataframe_with_satellites(results)
    except Exception as e:
        # check if the results is in the format {'filename': {'shift_x': 10.0, 'shift_y': 20.0, ..},'filename1': {'shift_x': 10.0, 'shift_y': 20.0, ..}}
        df = pd.DataFrame(results).transpose()

    df.fillna(0.0, inplace=True)

    # Process and plot outliers (dummy placeholder, replace with your actual implementation)
    if z_threshold:
        df = filter_by_z_score(
            df,
            z_threshold=z_threshold,
            combined_z_plot_filename='combined_z_scores.png',
            shifts_plot_filename='plot_outlier_shifts.png'
        )

    # Filter shifts based on ranges (dummy placeholder, replace with your actual implementation)
    df = filter_shifts_by_range(df, min_shift_meters, max_shift_meters)

    # Modify DataFrame structure
    df['filename'] = df.index
    df.reset_index(drop=True, inplace=True)
    df = df[['filename'] + [col for col in df.columns if col != 'filename']]

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Filtered DataFrame saved to {output_path}")

    return output_path
