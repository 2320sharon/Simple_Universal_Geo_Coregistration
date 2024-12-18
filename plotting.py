import numpy as np
import os
import numpy as np
import matplotlib.patches as mpatches

import matplotlib
# matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering
import matplotlib.pyplot as plt


def plot_ssim_scores_dev(results, output_dir):

         # Initialize counters for each category
        counts = {
            'no_valid_window': 0,
            'shift_exceeded': 0,
            'no_shift': 0,
            'success': 0,
            'failed': 0
        }

        # make a scatter plot of coregistered ssim vs original ssim scores and color those whose 'success' is True green and those whose 'success' is False red
        fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size if necessary
        for key, value in results.items():
            if key == 'file_inputs' or 'settings' in key:
                continue
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

        # Create custom legend with counts
        yellow_patch = mpatches.Patch(color='yellow', label=f'No valid matching window found ({counts["no_valid_window"]})')
        orange_patch = mpatches.Patch(color='orange', label=f'Shift exceeded ({counts["shift_exceeded"]})')
        purple_patch = mpatches.Patch(color='purple', label=f'No shift ({counts["no_shift"]})')
        green_patch = mpatches.Patch(color='green', label=f'Success ({counts["success"]})')
        red_patch = mpatches.Patch(color='red', label=f'Shift made it worse ({counts["failed"]})')

        ax.legend(handles=[yellow_patch, orange_patch, purple_patch, green_patch, red_patch], loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_xlabel('Original SSIM')
        ax.set_ylabel('Coregistered SSIM')
        ax.set_title('Original SSIM vs Coregistered SSIM')

        # plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to make space for the legend
        plt.tight_layout()  # Adjust the layout to make space for the legend
        plt.savefig(os.path.join(output_dir, 'dev_ssim_scatter_plot.png'))
        print(f"Plot saved to {os.path.join(output_dir, 'dev_ssim_scatter_plot.png')}")



def plot_delta_ssim_scores(results, output_dir):

         # Initialize counters for each category
        counts = {
            'shift_exceeded': 0,
            'success': 0,
            'failed': 0
        }

        # make a scatter plot of coregistered ssim vs original ssim scores and color those whose 'success' is True green and those whose 'success' is False red
        fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size if necessary
        for key, value in results.items():
            if key == 'file_inputs' or 'settings' in key:
                continue
            if "shift exceeded" in value['description']:
                ax.scatter(value['coregistered_ssim'],value['change_ssim'], color='orange')  # bad shift
                counts['shift_exceeded'] += 1
            elif value['success'] == 'True':
                ax.scatter(value['coregistered_ssim'],value['change_ssim'], color='green')
                counts['success'] += 1
            else:
                ax.scatter(value['coregistered_ssim'],value['change_ssim'], color='red')  # the shift made the coregistered image worse
                counts['failed'] += 1

        # Create custom legend with counts
        orange_patch = mpatches.Patch(color='orange', label=f'Shift exceeded ({counts["shift_exceeded"]})')
        green_patch = mpatches.Patch(color='green', label=f'Success ({counts["success"]})')
        red_patch = mpatches.Patch(color='red', label=f'Shift made it worse ({counts["failed"]})')

        ax.legend(handles=[ orange_patch,  green_patch, red_patch], loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_xlabel('Coregistered SSIM')
        ax.set_ylabel('delta SSIM')
        ax.set_title('Delta SSIM vs Coregistered SSIM')

        # plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to make space for the legend
        plt.tight_layout()  # Adjust the layout to make space for the legend
        plt.savefig(os.path.join(output_dir, 'delta_ssim_scatter_plot.png'))
        print(f"Plot saved to {os.path.join(output_dir, 'delta_ssim_scatter_plot.png')}")


def plot_ssim_scores(results, output_dir):

     # Initialize counters for each category
    counts = {
        'no_valid_window': 0,
        'shift_exceeded': 0,
        'no_shift': 0,
        'success': 0,
        'failed': 0
    }
    # make a scatter plot of coregistered ssim vs original ssim scores and color those whose 'success' is True green and those whose 'success' is False red
    fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size if necessary
    for key in results.keys():
        if key == 'file_inputs' or 'settings' in key:
            continue
        print(f"key: {key}")
        print(f"results[key]: {results[key]}")
        for key,value in results[key].items():
            if "shift exceeded" in value['description']:
                ax.scatter(value['original_ssim'], value['coregistered_ssim'], color='orange')  # bad shift
                counts['shift_exceeded'] += 1
            elif value['success'] == 'True':
                ax.scatter(value['original_ssim'], value['coregistered_ssim'], color='green')
                counts['success'] += 1
            else:
                ax.scatter(value['original_ssim'], value['coregistered_ssim'], color='red')  # the shift made the coregistered image worse
                counts['failed'] += 1
    # Create custom legend with counts
    orange_patch = mpatches.Patch(color='orange', label=f'Shift exceeded ({counts["shift_exceeded"]})')
    green_patch = mpatches.Patch(color='green', label=f'Success ({counts["success"]})')
    red_patch = mpatches.Patch(color='red', label=f'Shift made it worse ({counts["failed"]})')
    ax.legend(handles=[ orange_patch,  green_patch, red_patch], loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Original SSIM')
    ax.set_ylabel('Coregistered SSIM')
    ax.set_title('Original SSIM vs Coregistered SSIM')
    # plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to make space for the legend
    plt.tight_layout()  # Adjust the layout to make space for the legend
    plt.savefig(os.path.join(output_dir, 'ssim_scatter_plot.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'ssim_scatter_plot.png')}")



def plot_x_y_delta_ssim_scatter(results, output_dir):
    # Initialize counters for each category
    counts = {
        'no_valid_window': 0,
        'shift_exceeded': 0,
        'no_shift': 0,
        'success': 0,
        'failed': 0
    }

    # Initialize lists for shifts and delta SSIM
    shifts_x = []
    shifts_y = []
    delta_ssim = []
    colors = []

    for key, value in results.items():
        if key == 'file_inputs' or 'settings' in key:
            continue
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

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(shifts_x, shifts_y, delta_ssim, c=colors)

    # Create custom legend with counts
    orange_patch = mpatches.Patch(color='orange', label=f'Shift exceeded ({counts["shift_exceeded"]})')
    green_patch = mpatches.Patch(color='green', label=f'Success ({counts["success"]})')
    red_patch = mpatches.Patch(color='red', label=f'Shift made it worse ({counts["failed"]})')

    ax.legend(handles=[orange_patch, green_patch, red_patch], loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_xlabel('X shift in pixels')
    ax.set_ylabel('Y shift in pixels')
    ax.set_zlabel('Delta SSIM')
    ax.set_title('X Y Delta SSIM Scatter Plot')

    plt.tight_layout()  # Adjust the layout to make space for the legend
    plt.savefig(os.path.join(output_dir, 'x_y_delta_ssim_scatter_plot.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'x_y_delta_ssim_scatter_plot.png')}")



# def plot_x_y_delta_ssim_scatter(result_json_path, output_dir):
#     with open(result_json_path, 'r') as json_file:
#         results = json.load(json_file)

#          # Initialize counters for each category
#         counts = {
#             'no_valid_window': 0,
#             'shift_exceeded': 0,
#             'no_shift': 0,
#             'success': 0,
#             'failed': 0
#         }

#         # make a scatter plot of coregistered ssim vs original ssim scores and color those whose 'success' is True green and those whose 'success' is False red
#         fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size if necessary
#         for key, value in results.items():
#             if key == 'file_inputs' or 'settings' in key:
#                 continue
#             if "shift exceeded" in value['description']:
#                 ax.scatter(value['shift_x'], value['shift_y'],value['change_ssim'], color='orange')  # bad shift
#                 counts['shift_exceeded'] += 1
#             elif value['success'] == 'True':
#                 ax.scatter(value['shift_x'], value['shift_y'],value['change_ssim'], color='green')
#                 counts['success'] += 1
#             else:
#                 ax.scatter(value['shift_x'], value['shift_y'],value['change_ssim'], color='red')  # the shift made the coregistered image worse
#                 counts['failed'] += 1

#         # Create custom legend with counts
#         orange_patch = mpatches.Patch(color='orange', label=f'Shift exceeded ({counts["shift_exceeded"]})')
#         green_patch = mpatches.Patch(color='green', label=f'Success ({counts["success"]})')
#         red_patch = mpatches.Patch(color='red', label=f'Shift made it worse ({counts["failed"]})')

#         ax.legend(handles=[ orange_patch,  green_patch, red_patch], loc='center left', bbox_to_anchor=(1, 0.5))

#         ax.set_xlabel('X shift in pixels')
#         ax.set_ylabel('Y shift in pixels')
#         # ax.set_zlabel('delta SSIM')
#         ax.set_title(' X Y delta SSIM scatter plot')

#         # plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to make space for the legend
#         plt.tight_layout()  # Adjust the layout to make space for the legend
#         plt.savefig(os.path.join(output_dir, 'x_y_delta_ssim_scatter_plot.png'))
#         print(f"Plot saved to {os.path.join(output_dir, 'x_y_delta_ssim_scatter_plot.png')}")


def plot_shift_histogram(results, output_dir):

    shifts_x = []
    shifts_y = []

    for key, value in results.items():
        if key == 'file_inputs' or 'settings' in key:
            continue
        shifts_x.append(value['initial_shift_x'])
        shifts_y.append(value['initial_shift_y'])

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

    for key, value in results.items():
        if key == 'file_inputs' or 'settings' in key:
            continue
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


def plot_coregistration_success_by_month(results, output_dir):

    success_by_month = {}
    failed_by_month = {}

    for key, value in results.items():
        if key == 'file_inputs' or 'settings' in key:
            continue
        date = get_date_from_filename(key)
        month = date[:7]  # Extract YYYY-MM
        if month not in success_by_month:
            success_by_month[month] = 0
            failed_by_month[month] = 0
        
        if value['success'] == 'True':
            success_by_month[month] += 1
        else:
            failed_by_month[month] += 1

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


def merge_list_of_dicts(list_of_dicts):
    merged_dict = {}
    for d in list_of_dicts:
        merged_dict.update(d)
    return merged_dict


def get_date_from_filename(filename):
    """
    Extracts the date from the filename in the format YYYY-MM-DD.
    Assumes the date is the first part of the filename.
    """
    return filename.split('_')[0]