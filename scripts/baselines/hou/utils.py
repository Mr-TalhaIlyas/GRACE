import numpy as np
import scipy.signal
import scipy.stats
from scipy.fft import fft
import random
import os

def seed_torch(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def compute_kinematic_features(trajectory):
    """
    Computes the kinematic features for a single trajectory.
    trajectory: np.array of shape (T,)
    Returns a list of 17 features.
    """
    # Compute velocity, acceleration, and jerk
    velocity = np.gradient(trajectory)
    acceleration = np.gradient(velocity)
    jerk = np.gradient(acceleration)
    
    # For each signal, compute statistical features
    features = []
    for signal in [velocity, acceleration, jerk]:
        features.extend([
            np.std(signal),
            np.median(signal),
            np.mean(signal),
            np.max(signal),
            np.min(signal)
        ])
    
    return features  # List of 15 features

def compute_distance_features(x_coords, y_coords):
    """
    Computes the total covered distance and movement displacement.
    x_coords, y_coords: np.array of shape (T,)
    Returns a list of 2 features.
    """
    # Compute Euclidean distances between consecutive points
    diffs = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    total_distance = np.sum(diffs)
    
    # Compute displacement from initial to final position
    displacement = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
    
    return [total_distance, displacement]

def compute_spectral_features(signal, fs=1.0):
    """
    Computes spectral features for a given signal.
    signal: np.array of shape (T,)
    fs: Sampling frequency (default 1.0)
    Returns a list of 4 features.
    """
    # Remove NaNs
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    # Compute Power Spectral Density (PSD)
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=len(signal))
    
    # Normalize PSD
    # try:
    psd /= np.sum(psd)
    # except RuntimeError:
    #     psd = 0.0
    
    # Entropy
    entropy = -np.sum(psd * np.log2(psd + 1e-12))
    
    # Peak Magnitude
    peak_magnitude = np.max(psd)
    
    # Sum of Spectrum
    sum_spectrum = np.sum(psd)
    
    # Spectral Half Point
    cumulative_sum = np.cumsum(psd)
    half_point_idx = np.searchsorted(cumulative_sum, 0.5)
    spectral_half_point = freqs[half_point_idx] if half_point_idx < len(freqs) else freqs[-1]
    
    return [entropy, peak_magnitude, sum_spectrum, spectral_half_point]

#%%
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
def display_video(video):
    fig = plt.figure(figsize=(3,3))  #Display size specification

    mov = []
    for i in range(len(video)):  #Append videos one by one to mov
        img = plt.imshow(video[i], animated=True)
        plt.axis('off')
        mov.append([img])

    #Animation creation
    anime = animation.ArtistAnimation(fig, mov, interval=50, repeat_delay=1000)

    plt.close()
    return anime


'''
DATA UTILS
'''
import numpy as np
import torch
from tabulate import tabulate

def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip

class NormalizeVideo(object):
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(
            self.mean, self.std, self.inplace)

apply_norm = NormalizeVideo(mean=(128,128,128), std=(128,128,128))

def to_norm_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    # return clip.float().permute(3, 0, 1, 2) / 255.0 # range [0, 1]
    return apply_norm(clip.float().permute(3, 0, 1, 2)) # range [-1, 1]

def video_transform(videos):
    '''
    videos: list of PIL videos
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (B, T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (B, C, T, H, W)
    '''
    inputs = []
    for vid in videos:
        inputs.append(to_norm_tensor(vid))
    inputs = torch.stack(inputs, dim=0).float().to('cuda' if torch.cuda.is_available() else 'cpu')
    return inputs

def collate(batch):
    '''
    custom Collat funciton for collating individual fetched data samples into batches.
    '''
    frames = [b['frames'] for b in batch]
    
    body = [b['body'] for b in batch]
    face = [b['face'] for b in batch]
    rh = [b['rh'] for b in batch]
    lh = [b['lh'] for b in batch]
    
    ecg = [b['ecg'] for b in batch]

    sub_lbls = [b['sub_lbls'] for b in batch]
    super_lbls = [b['super_lbls'] for b in batch]    

    filename = [b['filename'] for b in batch]
    return {'frames': frames, 'body': body, 'face': face,
            'rh': rh, 'lh': lh, 'ecg': ecg, 'sub_lbls': sub_lbls,
            'super_lbls': super_lbls, 'filename': filename}

def generate_soft_labels(sample, num_classes):
    label = np.zeros(num_classes)
    for element in sample:
        label[element] += 1
    label /= len(sample)  # Normalize to get proportions
    return label

def get_of_indices(seconds_to_sample):
    
    max_samples_per_interval = 50
    frames_per_second = 30
    
    frames_per_second_to_sample = int(np.ceil(max_samples_per_interval/seconds_to_sample))
    
    # Generate random indices for frame sampling
    sampled_indices = []
    for second in range(1, seconds_to_sample + 1):
        start_frame = int((second - 1) * frames_per_second)
        end_frame = int(second * frames_per_second)
        sampled_indices.extend(np.random.choice(range(start_frame, end_frame),
                                                frames_per_second_to_sample,
                                                replace=False))
    # this 48 is because slowfast network's input
    x = np.random.choice(sampled_indices, 48, replace=False) 
    x.sort()
    return x

def get_pose_indices(seconds_to_sample):
    
    max_samples_per_interval = 150
    frames_per_second_to_sample = int(np.ceil(max_samples_per_interval/seconds_to_sample))
    
    frames_per_second = 30
    
    # Generate random indices for frame sampling
    sampled_indices = []
    for second in range(1, seconds_to_sample + 1):
        start_frame = int((second - 1) * frames_per_second)
        end_frame = int(second * frames_per_second)
        sampled_indices.extend(np.random.choice(range(start_frame, end_frame),
                                                frames_per_second_to_sample,
                                                replace=False))
    x = np.random.choice(sampled_indices, max_samples_per_interval, replace=False) 
    x.sort()
    return x

# x = get_of_indices(10)
# y = get_pose_indices(10)

def values_fromreport(report):
    p = report['weighted avg']['precision']
    r = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    return p,r, f1

def print_formatted_table(data):
    """
    Prints a formatted table of model performance metrics.

    Parameters:
    - data (dict): A dictionary containing model names as keys and lists of metrics as values.
                   Each list should contain floating-point numbers.
    - metrics (list): A list of strings representing the names of the metrics.
    """
    metrics = ['Preictal', 'Ictal', 'Avg Acc', 'Precision', 'Recall', 'F1']
    rows = []

    # Prepare data for tabulation
    for model, values in data.items():
        row = [model] + [f"{value:.4f}" for value in values]
        rows.append(row)

    # Define the table headers
    headers = ['Modality'] + metrics

    # Print the table
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    return None