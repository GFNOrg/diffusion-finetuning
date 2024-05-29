import math
import matplotlib.pyplot as plt


def plot_samples(samples, sample_logs=None, title='', show=False, save=False, filename=''):
    num_samples = len(samples)
    grid_size = math.ceil(math.sqrt(num_samples))

    fig, axs = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(10, 10))
    axs = axs.flatten()

    for i in range(grid_size * grid_size):
        axs[i].axis('off')
        if i < num_samples:
            axs[i].imshow(samples[i], vmin=-1, vmax=1, cmap='gray')
            if sample_logs is not None:
                axs[i].set_title(sample_logs[i], fontsize=26)

    fig.suptitle(title, fontsize=32)

    if save and filename:
        plt.savefig(filename)

    if show:
        plt.show()

    if save or show:
        plt.close(fig)  # Close the figure after saving

    return fig, axs  # Return the figure and axes for further use


def compare_samples(samples1, samples2, sample1_title, sample2_title, sample1_logs=None, sample2_logs=None, dpi=300,
                    show=False, save=False, filename=''):
    num_samples1 = len(samples1)
    num_samples2 = len(samples2)
    grid_size1 = math.ceil(math.sqrt(num_samples1))
    grid_size2 = math.ceil(math.sqrt(num_samples2))

    # Calculate the total grid size needed
    total_cols = grid_size1 + grid_size2 + 1  # +1 for space between sets
    total_rows = max(grid_size1, grid_size2)

    fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols, figsize=(40, 20))

    # Hide all axes initially
    for ax in axs.flat:
        ax.axis('off')

    # Function to display samples in the grid
    def display_samples(samples, axs, start_row, start_col, num_samples, grid_size, sample_logs=None):
        for i in range(num_samples):
            row_idx = (i // grid_size) + start_row
            col_idx = (i % grid_size) + start_col
            ax = axs[row_idx, col_idx]
            ax.imshow((samples[i]+1)/2, cmap='gray')
            ax.axis('off')  # Only turn on the axis for images
            if sample_logs is not None:
                ax.set_title(sample_logs[i], fontsize=24)

    # Display first set of samples
    display_samples(samples1, axs, 0, 0, num_samples1, grid_size1, sample1_logs)

    # Display second set of samples, offset by the first set plus an additional column for spacing
    start_col_for_samples2 = grid_size1 + 1  # +1 for space
    display_samples(samples2, axs, 0, start_col_for_samples2, num_samples2, grid_size2, sample2_logs)

    # Set super title for the entire figure
    fig.suptitle(f"{sample1_title} vs {sample2_title}", fontsize=32)

    if save and filename:
        plt.savefig(filename, dpi=dpi)

    if show:
        plt.show()

    plt.close(fig)  # Close the figure to clean up


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def plot_exp_logs(exp_logs, exp_args, smoothing=.9, path='', filter='', show=True, save=False):
    # Extract unique keys from the experiments (assuming all experiments have the same keys)
    keys = next(iter(exp_logs.values())).keys()
    num_keys = len(keys)

    # Determine the layout for subplots (as square as possible)
    rows = cols = math.ceil(math.sqrt(num_keys))

    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    if not isinstance(axs, plt.Axes):
        axs = axs.flatten()  # Flatten the array for easy indexing
    else:
        axs = [axs]

    handles, labels = [], []

    # Loop over each key and plot the corresponding data for each experiment
    for idx, key in enumerate(keys):
        for exp_name, exp_data in exp_logs.items():
            if filter in exp_name:
                if key in exp_data.keys() and len(exp_data[key]) > 0:
                    line, = axs[idx].plot(exp_data[key], alpha=.1)
                    line, = axs[idx].plot(smooth(exp_data[key], smoothing), color=line.get_color())
                    axs[idx].tick_params(axis='both', which='major', labelsize=16)
                    if idx == 0:  # Only add to legend once
                        handles.append(line)
                        # labels.append(f"lr: {exp_args[exp_name]['lr']}, "
                        #               f"lr logZ: {exp_args[exp_name]['lr_logZ']}, "
                        #               f"ct: {exp_args[exp_name]['learning_cutoff']}")

                    labels.append(exp_name)
        axs[idx].set_title(key, fontsize=22)

        # axs[idx].legend(fontsize=14)
    fig.legend(handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.3),
               ncol=2,
               fontsize=18)

    # Hide any unused subplots
    for i in range(num_keys, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(path+"/experiment_run.png")
    if show:
        plt.show()


def plot_separate_exp_logs(exp_logs, exp_args, smoothing=.9, path='', filter='', show=True, save=False):
    # Extract unique keys from the experiments (assuming all experiments have the same keys)
    keys = next(iter(exp_logs.values())).keys()
    num_keys = len(keys)

    # Determine the layout for subplots (as square as possible)
    cols = 5
    rows = num_keys // cols + (1 if num_keys % cols != 0 else 0)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if not isinstance(axs, plt.Axes):
        axs = axs.flatten()  # Flatten the array for easy indexing
    else:
        axs = [axs]

    # Loop over each key and plot the corresponding data for each experiment
    for exp_name, exp_data in exp_logs.items():
        for idx, key in enumerate(keys):
            if filter in exp_name:
                if key in exp_data.keys() and len(exp_data[key]) > 0:
                    line, = axs[idx].plot(exp_data[key], alpha=.1)
                    axs[idx].plot(smooth(exp_data[key], smoothing), color=line.get_color())
                    axs[idx].tick_params(axis='both', which='major', labelsize=16)

            axs[idx].set_title(key, fontsize=22)

        # Hide any unused subplots
        for i in range(num_keys, len(axs)):
            axs[i].axis('off')

        plt.suptitle(exp_name, fontsize=32)
        plt.tight_layout()
        if save:
            plt.savefig(path+"/experiment_run.png")
        if show:
            plt.show()

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if not isinstance(axs, plt.Axes):
            axs = axs.flatten()  # Flatten the array for easy indexing
        else:
            axs = [axs]