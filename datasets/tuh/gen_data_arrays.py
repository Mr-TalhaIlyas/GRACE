# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 17:02:59 2025

@author: talha
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 23:10:26 2024

@author: talha
"""

import numpy as np
from pathlib import Path
import os
import numpy as np
from fmutils import fmutils as fmu
from pathlib import Path
import joblib
import glob

def extract_patches(EEG_data, labels, x, y, frequency):
    """
    Extract overlapping patches from EEG data based on labels.

    Parameters:
    - EEG_data: np.ndarray of shape (C, T)
    - labels: np.ndarray of shape (T,)
    - x: Window length in seconds
    - y: Overlap length in seconds
    - frequency: Sampling frequency in Hz

    Returns:
    - data_array: np.ndarray of shape (N, C, W)
    - labels_array: np.ndarray of shape (N,)
    """
    wlen = int(x * frequency)        # Window length in samples
    woverlap = int(y * frequency)    # Overlap length in samples
    step = wlen - woverlap           # Step size between windows

    data_patches = []
    label_patches = []

    # For each label (0 and 1)
    for label_value in np.unique(labels):
        # Identify indices where labels match the current label_value
        label_indices = np.where(labels == label_value)[0]

        # If no data for this label, skip
        if len(label_indices) == 0:
            continue

        # Split indices into continuous segments
        segments = np.split(label_indices, np.where(np.diff(label_indices) != 1)[0] + 1)

        for segment_indices in segments:
            segment_start = segment_indices[0]
            segment_end = segment_indices[-1] + 1  # +1 because end index is exclusive
            segment = EEG_data[:, segment_start:segment_end]
            T_segment = segment.shape[1]

            if T_segment < wlen:
                # If segment is shorter than window length, use negative indexing to get the last wlen samples
                patch = segment[:, -wlen:]
                # If the segment is still shorter, pad at the beginning
                if patch.shape[1] < wlen:
                    pad_width = wlen - patch.shape[1]
                    patch = np.pad(patch, ((0, 0), (pad_width, 0)), 'constant')
                data_patches.append(patch)
                label_patches.append(label_value)
            else:
                # Extract overlapping patches
                n_windows = int(np.ceil((T_segment - woverlap) / step))
                for i in range(n_windows):
                    s = i * step
                    e = s + wlen
                    if e > T_segment:
                        # Use negative indexing to get the last wlen samples
                        e = T_segment
                        s = e - wlen
                    patch = segment[:, s:e]
                    data_patches.append(patch)
                    label_patches.append(label_value)

    # Convert lists to numpy arrays
    data_array = np.stack(data_patches)      # Shape: (N, C, W)
    labels_array = np.array(label_patches)   # Shape: (N,)

    return data_array, labels_array

def gen_sliding_arrays(train_paths, train_lbls, wlen=10, overlap=9, freq=250):
    all_data, all_lbls = [], []
    for i in range(len(train_paths)):
    
        fname = Path(train_paths[i]).stem
        
        eeg_data = np.load(train_paths[i])
        
        eeg_lbls = np.load(train_lbls[i])
        
        # filtered_indices = np.where(eeg_lbls != 5)[0]
        # eeg_lbls = eeg_lbls[filtered_indices]
        
        # eeg_data = eeg_data[:, 0: len(eeg_lbls)]
        
        # eeg_lbls = np.clip(eeg_lbls, 0, 1) # Only keep the TCS sezuer not sub_labels.
        
        data_array, labels_array = extract_patches(eeg_data, eeg_lbls, wlen, overlap, freq) # Shape: (N, C, W), (N)
        
        # if 'b_' in fname or 'PN' in fname:
        #     labels_array = labels_array * 2 # change labels to 2 for PNES
        
        
        all_data.append(data_array)
        all_lbls.append(labels_array)
    
    all_data = np.concatenate(all_data, axis=0)
    all_lbls = np.concatenate(all_lbls, axis=0)
    
    return all_data, all_lbls

#%%



train_paths = glob.glob('/home/user01/Data/npj/datasets/tuh/train/hrv/*.npy')
train_lbls = glob.glob('/home/user01/Data/npj/datasets/tuh/train/labels/*.npy')

train_paths =  sorted(train_paths, key=fmu.numericalSort)
train_lbls = sorted(train_lbls, key=fmu.numericalSort)

test_paths = glob.glob('/home/user01/Data/npj/datasets/tuh/dev/hrv/*.npy')
test_lbls = glob.glob('/home/user01/Data/npj/datasets/tuh/dev/labels/*.npy')

test_paths =  sorted(test_paths, key=fmu.numericalSort)
test_lbls = sorted(test_lbls, key=fmu.numericalSort)
#%%

all_train_data, all_train_lbls = gen_sliding_arrays(train_paths, train_lbls,
                                                   wlen=10, overlap=5, freq=250)

all_test_data, all_test_lbls = gen_sliding_arrays(test_paths, test_lbls,
                                                   wlen=10, overlap=5, freq=250)



# Combine train and test data for 'All_train_data' and 'All_train_label'
All_train_data = np.concatenate((all_train_data, all_test_data), axis=0)
All_train_label = np.concatenate((all_train_lbls, all_test_lbls), axis=0)

val_data = all_test_data
val_label = all_test_lbls

'''
np.unique(all_test_lbls)
[-1  0  1] [-1  0  1]
[0 1 2] [0 1 2]

JUST by adding +1 we will make bckg label==0 && foc_f2c labels==1
same as our ALFRED data
'''
print(np.unique(val_label), np.unique(all_train_lbls))

all_train_lbls = all_train_lbls + 1
all_test_lbls = all_test_lbls + 1
# Create the data dictionary
Data = {
    'train_data': all_train_data,
    'train_label': all_train_lbls,
    # 'val_data': val_data,
    # 'val_label': val_label,
    'test_data': all_test_data,
    'test_label': all_test_lbls,
    # 'All_train_data': All_train_data,
    # 'All_train_label': All_train_label
}
print(np.unique(all_test_lbls), np.unique(all_train_lbls))

#%%
os.makedirs('/home/user01/Data/npj/datasets/tuh/ts_analysis/', exist_ok=True)
# Save the Data dictionary to a .npy file
save_path = f'/home/user01/Data/npj/datasets/tuh/ts_analysis/TUHv152_ecgH_full_w10_o5.joblib'  # Replace with your desired file path
# np.save(save_path, Data, allow_pickle=True)

# Use protocol=4 or higher to handle files larger than 4 GiB
joblib.dump(Data, save_path, protocol=4)
# %%
