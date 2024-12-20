import os
import numpy as np
import matplotlib.patches as mpatches

import matplotlib
# matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering
import matplotlib.pyplot as plt

def contains_subdictionaries(data_dict):
    """Check if the provided dictionary contains multiple sub-dictionaries."""
    subdict_count = sum(1 for value in data_dict.values() if isinstance(value, dict))
    return subdict_count > 1  # Return True if there are multiple sub-dictionaries


def get_date_from_filename(filename):
    """
    Extracts the date from the filename in the format YYYY-MM-DD.
    Assumes the date is the first part of the filename.
    """
    return filename.split('_')[0]

def plot_ssim_scores_dev(results, output_dir):
    counts = {
        'no_valid_window': 0,
        'shift_exceeded': 0,
        'no_shift': 0,
        'success': 0,
        'failed': 0
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, key_data in results.items():
        if key == 'file_inputs' or 'settings' in key:
            continue
        if not key_data:
            continue
        if contains_subdictionaries(key_data):
            for sub_key, value in key_data.items():
                plot_ssim_point(ax, value, counts)
        else:
            plot_ssim_point(ax, key_data, counts)

    yellow_patch = mpatches.Patch(color='yellow', label=f'No valid matching window found ({counts["no_valid_window"]})')
    orange_patch = mpatches.Patch(color='orange', label=f'Shift exceeded ({counts["shift_exceeded"]})')
    purple_patch = mpatches.Patch(color='purple', label=f'No shift ({counts["no_shift"]})')
    green_patch = mpatches.Patch(color='green', label=f'Success ({counts["success"]})')
    red_patch = mpatches.Patch(color='red', label=f'Shift made it worse ({counts["failed"]})')

    ax.legend(handles=[yellow_patch, orange_patch, purple_patch, green_patch, red_patch], loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Original SSIM')
    ax.set_ylabel('Coregistered SSIM')
    ax.set_title('Original SSIM vs Coregistered SSIM')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dev_ssim_scatter_plot.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'dev_ssim_scatter_plot.png')}")

def plot_ssim_point(ax, value, counts):
    if "no valid matching window found" in value['description']:
        ax.scatter(value['original_ssim'], value['coregistered_ssim'], color='yellow')  # no valid matching window found
        counts['no_valid_window'] += 1
    elif "shift exceeded" in value['description']:
        ax.scatter(value['original_ssim'], value['coregistered_ssim'], color='orange')  # bad shift
        counts['shift_exceeded'] += 1
    elif value['initial_shift_x'] == 0 and value['initial_shift_y'] == 0:
        ax.scatter(value['original_ssim'], value['coregistered_ssim'], color='purple')  # no shift
        counts['no_shift'] += 1
    elif value['success'] == 'True':
        ax.scatter(value['original_ssim'], value['coregistered_ssim'], color='green')
        counts['success'] += 1
    else:
        ax.scatter(value['original_ssim'], value['coregistered_ssim'], color='red')  # the shift made the coregistered image worse
        counts['failed'] += 1

def plot_delta_ssim_scores(results, output_dir):
    counts = {
        'shift_exceeded': 0,
        'success': 0,
        'failed': 0
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, key_data in results.items():
        if key == 'file_inputs' or 'settings' in key:
            continue
        if not key_data:
            continue
        if contains_subdictionaries(key_data):
            for sub_key, value in key_data.items():
                plot_delta_ssim_point(ax, value, counts)
        else:
            plot_delta_ssim_point(ax, key_data, counts)

    orange_patch = mpatches.Patch(color='orange', label=f'Shift exceeded ({counts["shift_exceeded"]})')
    green_patch = mpatches.Patch(color='green', label=f'Success ({counts["success"]})')
    red_patch = mpatches.Patch(color='red', label=f'Shift made it worse ({counts["failed"]})')

    ax.legend(handles=[orange_patch, green_patch, red_patch], loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Coregistered SSIM')
    ax.set_ylabel('delta SSIM')
    ax.set_title('Delta SSIM vs Coregistered SSIM')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delta_ssim_scatter_plot.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'delta_ssim_scatter_plot.png')}")

def plot_delta_ssim_point(ax, value, counts):
    if "shift exceeded" in value['description']:
        ax.scatter(value['coregistered_ssim'], value['change_ssim'], color='orange')  # bad shift
        counts['shift_exceeded'] += 1
    elif value['success'] == 'True':
        ax.scatter(value['coregistered_ssim'], value['change_ssim'], color='green')
        counts['success'] += 1
    else:
        ax.scatter(value['coregistered_ssim'], value['change_ssim'], color='red')  # the shift made the coregistered image worse
        counts['failed'] += 1

def plot_ssim_scores(results, output_dir):
    counts = {
        'no_valid_window': 0,
        'shift_exceeded': 0,
        'no_shift': 0,
        'success': 0,
        'failed': 0
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, key_data in results.items():
        if key == 'file_inputs' or 'settings' in key:
            continue
        if not key_data:
            continue
        # check if key_data contains multiple dictionaries if it does loop through them
        if contains_subdictionaries(key_data):
            for key, value in key_data.items():
                plot_ssim_point(ax, value, counts)
        else:
            plot_ssim_point(ax, key_data, counts)

    orange_patch = mpatches.Patch(color='orange', label=f'Shift exceeded ({counts["shift_exceeded"]})')
    green_patch = mpatches.Patch(color='green', label=f'Success ({counts["success"]})')
    red_patch = mpatches.Patch(color='red', label=f'Shift made it worse ({counts["failed"]})')

    ax.legend(handles=[orange_patch, green_patch, red_patch], loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Original SSIM')
    ax.set_ylabel('Coregistered SSIM')
    ax.set_title('Original SSIM vs Coregistered SSIM')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_scatter_plot.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'ssim_scatter_plot.png')}")

def plot_x_y_delta_ssim_scatter(results, output_dir):
    counts = {
        'no_valid_window': 0,
        'shift_exceeded': 0,
        'no_shift': 0,
        'success': 0,
        'failed': 0
    }

    shifts_x = []
    shifts_y = []
    delta_ssim = []
    colors = []

    for key, key_data in results.items():
        if key == 'file_inputs' or 'settings' in key:
            continue
        if not key_data:
            continue
        if contains_subdictionaries(key_data):
            for sub_key, value in key_data.items():
                append_shift_data(value, shifts_x, shifts_y, delta_ssim, colors, counts)
        else:
            append_shift_data(key_data, shifts_x, shifts_y, delta_ssim, colors, counts)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(shifts_x, shifts_y, delta_ssim, c=colors)

    orange_patch = mpatches.Patch(color='orange', label=f'Shift exceeded ({counts["shift_exceeded"]})')
    green_patch = mpatches.Patch(color='green', label=f'Success ({counts["success"]})')
    red_patch = mpatches.Patch(color='red', label=f'Shift made it worse ({counts["failed"]})')

    ax.legend(handles=[orange_patch, green_patch, red_patch], loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('X shift in pixels')
    ax.set_ylabel('Y shift in pixels')
    ax.set_zlabel('Delta SSIM')
    ax.set_title('X Y Delta SSIM Scatter Plot')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'x_y_delta_ssim_scatter_plot.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'x_y_delta_ssim_scatter_plot.png')}")

def append_shift_data(value, shifts_x, shifts_y, delta_ssim, colors, counts):
    shifts_x.append(value['shift_x'])
    shifts_y.append(value['shift_y'])
    delta_ssim.append(value['change_ssim'])
    if "shift exceeded" in value['description']:
        colors.append('orange')  # bad shift
        counts['shift_exceeded'] += 1
    elif value['success'] == 'True':
        colors.append('green')
        counts['success'] += 1
    else:
        colors.append('red')  # the shift made the coregistered image worse
        counts['failed'] += 1

def plot_shift_histogram(results, output_dir):

    shifts_x = []
    shifts_y = []

    for key, key_data in results.items():
        if key == 'file_inputs' or 'settings' in key:
            continue
        if not key_data:
            continue
        if contains_subdictionaries(key_data):
            for sub_key, value in key_data.items():
                shifts_x.append(value['initial_shift_x'])
                shifts_y.append(value['initial_shift_y'])
        else:
            shifts_x.append(key_data['initial_shift_x'])
            shifts_y.append(key_data['initial_shift_y'])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].hist(shifts_x, bins=30, color='blue', edgecolor='black')
    ax[0].set_title('Histogram of Shifts in X')
    ax[0].set_xlabel('Shift in X')
    ax[0].set_ylabel('Frequency')

    ax[1].hist(shifts_y, bins=30, color='blue', edgecolor='black')
    ax[1].set_title('Histogram of Shifts in Y')
    ax[1].set_xlabel('Shift in Y')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shift_histogram.png'))
    print(f"Histogram saved to {os.path.join(output_dir, 'shift_histogram.png')}")


def plot_shifts_by_month(results, output_dir):

    shifts_by_month = {}

    for key, key_data in results.items():
        if key == 'file_inputs' or 'settings' in key:
            continue
        if not key_data:
            continue
        if contains_subdictionaries(key_data):
            for sub_key, value in key_data.items():
                append_shift_by_month_data(sub_key, value, shifts_by_month)
        else:
            append_shift_by_month_data(key, key_data, shifts_by_month)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    for month, data in shifts_by_month.items():
        ax.scatter(data['x'], data['y'], color=data['colors'], label=month, alpha=0.6, edgecolors='w', linewidth=0.5)

    ax.set_xlabel('Shift in X')
    ax.set_ylabel('Shift in Y')
    ax.set_title('Shifts by Month')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shifts_by_month.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'shifts_by_month.png')}")

def append_shift_by_month_data(key, value, shifts_by_month):
    date = get_date_from_filename(key)
    month = date[:7]  # Extract YYYY-MM
    if month not in shifts_by_month:
        shifts_by_month[month] = {'x': [], 'y': [], 'colors': []}

    shifts_by_month[month]['x'].append(value['initial_shift_x'])
    shifts_by_month[month]['y'].append(value['initial_shift_y'])

    if value['success'] == 'True':
        shifts_by_month[month]['colors'].append('green')
    elif value['qc'] == 0:
        shifts_by_month[month]['colors'].append('orange')
    else:
        shifts_by_month[month]['colors'].append('red')

def plot_coregistration_success_by_month(results, output_dir):

    success_by_month = {}
    failed_by_month = {}

    for key, key_data in results.items():
        if key == 'file_inputs' or 'settings' in key:
            continue
        if not key_data:
            continue
        if contains_subdictionaries(key_data):
            for sub_key, value in key_data.items():
                append_success_by_month_data(sub_key, value, success_by_month, failed_by_month)
        else:
            append_success_by_month_data(key, key_data, success_by_month, failed_by_month)

    months = sorted(success_by_month.keys())
    success_counts = [success_by_month[month] for month in months]
    failed_counts = [failed_by_month[month] for month in months]

    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.bar(months, success_counts, color='green', label='Success')
    ax.bar(months, failed_counts, bottom=success_counts, color='red', label='Failed')

    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Coregistrations')
    ax.set_title('Number of Successful and Unsuccessful Coregistrations per Month')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coregistration_success_by_month.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'coregistration_success_by_month.png')}")

def append_success_by_month_data(key, value, success_by_month, failed_by_month):
    date = get_date_from_filename(key)
    month = date[:7]  # Extract YYYY-MM
    if month not in success_by_month:
        success_by_month[month] = 0
        failed_by_month[month] = 0

    if value['success'] == 'True':
        success_by_month[month] += 1
    else:
        failed_by_month[month] += 1


def create_readme(coregistered_dir, results):
    # create a readme.txt file at the output_dir
    with open(os.path.join(coregistered_dir, 'readme.txt'), 'w') as f:
        # read the json data and count the number of successful coregistrations
        successful_coregistrations = 0
        qc_failed = 0
        improvements = []
        for key, value in results.items():
            if 'settings' in key:
                continue
            if value['success'] == 'True':
                successful_coregistrations += 1
                # create a list of the change in ssim score for each successful coregistration
                # then create an average improvement in ssim score
                improvements.append(value['change_ssim'])
            if value['qc'] == 0:
                qc_failed += 1
            if len(improvements) == 0:
                average_improvement = 0
            elif len(improvements) == 1:
                average_improvement = improvements[0]
            else:
                average_improvement = np.mean(improvements)
            f.write(f"Number of successful coregistrations: {successful_coregistrations}\n")
            f.write(f"Total number of coregistrations: {len(results) - 2}\n")
            f.write(f"Average improvement in SSIM score: {average_improvement}\n")
            f.write(f"Number of QC failed coregistrations: {qc_failed}\n")
            f.write(f"Settings: {results['settings']}\n")

