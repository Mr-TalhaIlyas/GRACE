#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:01:07 2025

@author: user01
"""

import os
import numpy as np
import torch
import torch.utils.data as data
import decord
from decord import VideoReader
from pathlib import Path
import pywt
from sklearn.preprocessing import MinMaxScaler

from configs.config import config
from data.utils import get_pose_indices, get_of_indices, generate_soft_labels
from data.augment import vid_augment, pose_augment, ecg_augment


from configs.config import config
import torch
import torch.utils.data as data
from fmutils import fmutils as fmu
from data.utils import get_pose_indices, get_of_indices, generate_soft_labels
import decord as de
from decord import VideoReader
# from decord import cpu, gpu
import numpy as np
import cv2, os, glob
import decord
from pathlib import Path
import pywt
# from data.augmentors import video_augment, pose_augment, ecg_augment, cwt_augment
from sklearn.preprocessing import MinMaxScaler

from data.augment import ecg_augment
# decord.bridge.set_bridge('torch')
#%%

class GEN_DATA_LISTS:
    def __init__(self, config):
        self.config = config

    def get_splits(self, dataset='tuh'):
        base_dir = self.config[f'{dataset}_data_dir']
        print(37*'^')
        print(f"Loading {dataset} data from {base_dir}")
        print(37*'v')
        # loading train paths
        train_data = {
            'eeg_paths': sorted(glob.glob(os.path.join(base_dir, 'train', 'eeg', '*.npy'))),
            'ecg_paths':  sorted(glob.glob(os.path.join(base_dir, 'train', 'hrv', '*.npy'))),
            'lbls':   sorted(glob.glob(os.path.join(base_dir, 'train', 'labels', '*.npy')))
        }
        
        test_data = {
            'eeg_paths': sorted(glob.glob(os.path.join(base_dir, 'dev', 'eeg', '*.npy'))),
            'ecg_paths':  sorted(glob.glob(os.path.join(base_dir, 'dev', 'hrv', '*.npy'))),
            'lbls':   sorted(glob.glob(os.path.join(base_dir, 'dev', 'labels', '*.npy')))
        }
        # Check if all paths exist
        self.chk_paths(train_data)
        self.chk_paths(test_data)
        # Ensure all lists are of the same length
        if len(train_data['eeg_paths']) != len(train_data['ecg_paths']) or \
           len(train_data['eeg_paths']) != len(train_data['lbls']):
            raise ValueError("Mismatch in number of EEG, ECG, or label files in training data.")
        if len(test_data['eeg_paths']) != len(test_data['ecg_paths']) or \
           len(test_data['eeg_paths']) != len(test_data['lbls']):
            raise ValueError("Mismatch in number of EEG, ECG, or label files in test data.")
        
        return train_data, test_data

    def chk_paths(self, data):
        """
        Verify that all file paths in the data dict exist.
        Prints missing paths if any.
        """
        error_flag = False
        for key, paths in data.items():
            for path in paths:
                if not os.path.exists(path):
                    print(f"The path {path} does not exist.")
                    error_flag = True
        if not error_flag:
            print("All paths exist.")
            
        
class SlidingWindowBioSignalLoader(data.Dataset):
    def __init__(self, dataset_dict, config=config, dataset='tuh', augment=False):
        self.dataset_name = dataset
        # file lists
        self.ecg_paths = dataset_dict['ecg_paths']
        self.eeg_paths = dataset_dict['eeg_paths']
        self.lbl_paths = dataset_dict['lbls']

        self.config   = config
        self.augment  = augment

        # window settings
        W = config['sample_duration']          # seconds
        ov = config['window_overlap'] # config.get('window_overlap', W/2) # seconds
        self.stride = W - ov                   # seconds
        
        # build a complete index of (record_idx, vid_start_frame, ecg_start_sample)
        self.mapping = []
        for ridx in range(len(self.ecg_paths)):
            fname = os.path.basename(self.lbl_paths[ridx])
            # load labels only to measure length & clip post-ictal
            lbls = np.load(self.lbl_paths[ridx])
            
            n_timesteps = len(lbls)
            total_secs = n_timesteps / config['ecg_freq']
            if total_secs < W:
                continue

            n_windows = int(np.floor((total_secs - W) / self.stride)) + 1
            for w in range(n_windows):
                t0 = w * self.stride
                ef = int(t0 * config['ecg_freq'])
                self.mapping.append((fname, ridx, ef))

        assert len(self.mapping) > 0, "No sliding windows generated!"

    def normalize_ecg_channel2(self, ecg_data: np.ndarray,
                            method: str = 'min_max') -> np.ndarray:
        ecg = ecg_data.astype(np.float32, copy=True)
        
        # select the 2nd channel
        ch = ecg[2, :].reshape(-1, 1)  # shape (T,1)
        
        if method == 'min_max':
            scaler = MinMaxScaler(feature_range=(0, 1))
            ecg[2, :] = scaler.fit_transform(ch).flatten()
            
        elif method == 'z_score':
            mean = ch.mean()
            std  = ch.std() + 1e-7
            ecg[2, :] = ((ch - mean) / std).flatten()
            
        else:
            raise ValueError(f"Unknown method {method!r}")
        
        return ecg

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        fname, ridx, ecg_start = self.mapping[index]
        # file stems
        filename = Path(self.ecg_paths[ridx]).stem
        
        #— load raw ECG
        ecg_hrv = np.load(self.ecg_paths[ridx])
        ecg_end = ecg_start + self.config['sample_duration'] * self.config['ecg_freq']
        
        # Clip ECG indices to valid range
        ecg_length = ecg_hrv.shape[1]
        ecg_start = min(ecg_start, ecg_length - 1)
        ecg_end = min(ecg_end, ecg_length)
        
        #- load ECG HRV
        ecg = ecg_hrv[1:2, :] # 2nd channed is filtered ECG
        hrv = ecg_hrv
        # print(ecg_hrv.shape, ecg.shape, hrv.shape)
        #— load EEG
        eeg = np.load(self.eeg_paths[ridx])
        #-- assert eeg and ecg have same length
        assert eeg.shape[1] == hrv.shape[1], \
            f"EEG length {eeg.shape[1]} does not match ECG length {hrv.shape[1]} for file {fname}."
        
        #— load & clip labels
        lbls = np.load(self.lbl_paths[ridx])

        lbl_seg = lbls[ecg_start:ecg_end]

        #— assemble raw inputs
        
        ecg = ecg[:, ecg_start:ecg_end]
        hrv = hrv[:, ecg_start:ecg_end] # all 19 channels
        eeg = eeg[:, ecg_start:ecg_end]
        hrv = self.normalize_ecg_channel2(hrv, method='min_max') # we only z-score channel 2 i.e., heart rate
        # print('Loaded:', fname, ridx, ecg_start, ecg_end)
        # print('Next=',ecg.shape, ecg.shape, hrv.shape)
        #-- Now map labels to super and sub classes
        if self.dataset_name == 'tuh':
            mapping = self.config['LABEL_MAP_TUH']
        elif self.dataset_name == 'seizeit2':
            mapping = self.config['LABEL_MAP_SeizeIT2']
        
        seizure_types = list(mapping.keys())
        seizure_types.sort(key=lambda x: mapping[x])
        # read filename and see if key exists in mapping and get most frequent label
        for seizure_type in seizure_types:
            if seizure_type in fname:
                lbl_seg = np.array([mapping[seizure_type]] * len(lbl_seg))
                break
        #— generate sub, sup class labels
        sub_lbls = lbl_seg
        super_lbls = np.clip(lbl_seg + 1, 0, 1) # # 0 for baseline, 1 for seizure


        #— optional augment
        if self.augment:
            hrv = ecg_augment.apply_hrv_augmentation(hrv, p=0.7)
            eeg = ecg_augment.apply_eeg_augmentation(eeg, p=0.7)
            ecg = ecg_augment.apply_ecg_augmentation(ecg, p=0.7)
            
        data = {
            'eeg': eeg,
            'ecg': ecg,
            'hrv': hrv,
            'sub_lbls':   sub_lbls[0], # take first index as labels
            'super_lbls': super_lbls[0],
            'filename':   filename
        }
        return data

