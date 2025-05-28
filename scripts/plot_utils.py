import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import mode

def plot_tsne(embeddings, labels, legends=False, exclude_class_0=True):
    # sns.set(style="whitegrid")

    # Filter out class 0 if required
    if exclude_class_0:
        mask = labels != 0
        embeddings = embeddings[mask]
        labels = labels[mask]
        # Recalculate unique classes after filtering
        classes = np.unique(labels)
        # Map labels to a continuous range starting from 0
        label_to_id = {label: id for id, label in enumerate(classes)}
        mapped_labels = np.array([label_to_id[label] for label in labels])
    else:
        classes = np.unique(labels)
        mapped_labels = labels  # Use original labels if not excluding class 0

    # Create a scatter plot
    # plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=mapped_labels, cmap='Set1', s=3)
    plt.gca().set_aspect('equal', 'datalim')

    # Create a colorbar with correct ticks and labels
    cbar = plt.colorbar(boundaries=np.arange(len(classes)+1)-0.5, ticks=np.arange(len(classes)))
    cbar.set_ticklabels(classes)
    
    # Optionally add legends
    if legends:
        # Create custom legends with class labels
        legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
        plt.gca().add_artist(legend1)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    plt.tight_layout()  # Adjust layout to not cut off elements
    plt.show()
#%%
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions_over_labels(predictions, labels, cols=2):
    assert len(predictions) == len(labels), "Predictions and labels must have the same length."

    # Calculate the number of rows needed based on the number of columns
    num_pairs = len(predictions)
    rows = np.ceil(num_pairs / cols).astype(int)
    
    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 2.5))
    
    # Flatten the axes array for easy indexing
    axs = axs.flatten()
    
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if rows * cols != 1:
            ax = axs[i]
        else:
            ax = axs
            
        # Plot predictions and labels
        ax.plot(pred, label='Prediction', marker='o', linestyle='-', color='blue')
        ax.plot(label, label='Label', marker='x', linestyle='--', color='red')
        
        # Labeling the subplot
        ax.set_title(f'Pair {i+1}')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Only show plots in the axes that have data
        ax.axis('on')
    
    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axs[j])
    
    fig.tight_layout()
    plt.show()

# # Example usage
# predictions = [np.random.rand(10) for _ in range(5)]  # 5 random prediction arrays
# labels = [np.random.rand(10) for _ in range(5)]       # 5 random label arrays

# plot_predictions_over_labels(predictions, labels, cols=3)  # Adjust cols as needed
#%%
import matplotlib.pyplot as plt
import numpy as np

def plot_compact_horizontal_bars(predictions, labels):
    assert len(predictions) == len(labels), "Predictions and labels lists must be of the same length."
    
    num_samples = len(predictions)
    fig, ax = plt.subplots(figsize=(12, num_samples * 0.5 + 2))  # Adjust figure size as needed

    for i, (pred, label) in enumerate(zip(predictions, labels)):
        # Create a continuous bar for each prediction, with color indicating the prediction state
        for j, p in enumerate(pred):
            color = 'green' if p == 0 else 'red'
            ax.barh(y=i, width=1, left=j, color=color, edgecolor='gray', height=0.8)
        
        # Find indices where label changes from 0 to 1 and plot markers
        change_indices = np.where(np.diff(label) == 1)[0] + 1
        for idx in change_indices:
            ax.plot([idx, idx], [i-0.4, i+0.4], color='k', linestyle='--', linewidth=2, marker='*', markersize=5)

    ax.set_yticks(range(num_samples))
    ax.set_yticklabels([f'Pat {i+1}' for i in range(num_samples)])
    ax.set_xlabel('Index')
    ax.set_title('Predictions and Label Changes for Each Sample')
    
    # Optional: Add custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor='green', edgecolor='gray', label='No Seizure'),
                       Patch(facecolor='red', edgecolor='gray', label='Seizure'),
                       Line2D([0], [0], color='k', linestyle='--', linewidth=2, marker='*', markersize=5, label='Onset')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

# Example usage
# predictions = [[0, 0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1]]
# labels = [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]]

# plot_compact_horizontal_bars(predictions, labels)

#%%
def calculate_latency_directly(predictions, labels, consecutive_ones=1, clip_duration=10, overlap=3):
    # Define a helper function to find the first occurrence of `consecutive_ones` in the predictions
    def find_first_seizure_window(arr, threshold):
        consecutive_count = 0
        for i, val in enumerate(arr):
            if val == 1:
                consecutive_count += 1
                if consecutive_count >= threshold:
                    return i - (threshold - 1)  # Return the index of the first window in the sequence
            else:
                consecutive_count = 0
        return np.nan  # No seizure detected

    # Find the first window index where the seizure is labeled
    actual_onset_idx = find_first_seizure_window(labels, 1)  # Looking for the first '1'

    # Find the first window index where the seizure is predicted according to the consecutive ones criterion
    predicted_onset_idx = find_first_seizure_window(predictions, consecutive_ones)

    # If no seizure was detected in predictions or labels do not contain a seizure
    if predicted_onset_idx is None or actual_onset_idx is None:
        return None

    # Calculate detection latency in terms of window index difference
    latency = (predicted_onset_idx - actual_onset_idx) * (clip_duration-overlap)  # Multiply by 7 because each step is 10-3=7 seconds apart

    return latency

def find_split_indices(array):
    # Find indices where zeros start after non-zero sequences for splitting
    # return [i for i, (x, y) in enumerate(zip(array[:-1], array[1:]), start=1) if x == 1 and y == 0]
    return [i for i, (x, y) in enumerate(zip(array[:-1], array[1:]), start=1) if (x==1 or x==2) and y == 0]

def split_array_at_indices(array, indices):
    split_arrays = []
    start_idx = 0
    for idx in indices:
        split_arrays.append(array[start_idx:idx])
        start_idx = idx
    split_arrays.append(array[start_idx:])  # Add the last segment
    return split_arrays

def stabilize_predictions(predictions, window_size):
    """
    Stabilize predictions using a sliding window approach.
    """
    stabilized = np.copy(predictions)
    for i in range(len(predictions) - window_size + 1):
        window = predictions[i:i+window_size]
        majority_label = mode(window, keepdims=False, nan_policy='omit')[0]
        stabilized[i:i+window_size] = majority_label
    return stabilized