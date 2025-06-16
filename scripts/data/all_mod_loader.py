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
from data.augmentors import video_augment, pose_augment, ecg_augment, cwt_augment
from sklearn.preprocessing import MinMaxScaler

from data.augment import vid_augment, pose_augment, ecg_augment
# decord.bridge.set_bridge('torch')
#%%

class GEN_DATA_LISTS:
    def __init__(self, config):
        self.config = config
        self.folds = config.get('folds', None) # directory containing fold definition files
        self.ecg_dir = config['ecg_dir']     # directory for ECG .npy files
        self.eeg_dir = config['eeg_dir']     # directory for EEG .npy files
        self.flow_dir = config['flow_dir']   # directory for flow .mp4 files
        self.pose_dir = config['pose_dir']   # directory for pose .npy files (subfolders)
        self.lbl_dir = config['lbl_dir']     # directory for label .npy files

    def get_folds(self, num_fold=1):
        """
        If num_fold is None or negative, read all data directly from directories
        (ignoring any .txt files) and return a single combined data dict.
        Otherwise, return a tuple (train_data, test_data) for the specified fold index.
        """
        def read_samples(file_type, fold_number):
            path = os.path.join(self.folds, f"{file_type}_fold_{fold_number}.txt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Fold file not found: {path}")
            with open(path, 'r') as f:
                return [line.strip() for line in f]

        def generate_paths(samples):
            return {
                'flow_paths': [os.path.join(self.flow_dir, f"{s}_flow.mp4") for s in samples],
                'ecg_paths':  [os.path.join(self.ecg_dir,  f"{s}.npy")        for s in samples],
                'eeg_paths':  [os.path.join(self.eeg_dir,  f"{s}.npy")        for s in samples],
                'pose_paths':[os.path.join(self.pose_dir, s, "body_coco.npy")  for s in samples],
                'flow_lbls': [os.path.join(self.lbl_dir, f"{s}_vid_lbl.npy")  for s in samples],
                'ecg_lbls':  [os.path.join(self.lbl_dir, f"{s}_ecg_lbl.npy")  for s in samples]
            }

        # Default: use all data from directories (ignore .txt files)
        if num_fold is None or num_fold < 0:
            # Collect sample IDs from each modality
            ecg_samples = [os.path.splitext(os.path.basename(f))[0]
                           for f in sorted(glob.glob(os.path.join(self.ecg_dir, '*.npy')))]
            eeg_samples = [os.path.splitext(os.path.basename(f))[0]
                           for f in sorted(glob.glob(os.path.join(self.eeg_dir, '*.npy')))]
            flow_samples = [os.path.basename(f).replace('_flow.mp4', '')
                            for f in sorted(glob.glob(os.path.join(self.flow_dir, '*_flow.mp4')))]
            pose_samples = [d for d in sorted(os.listdir(self.pose_dir))
                            if os.path.isdir(os.path.join(self.pose_dir, d))]
            vid_lbl_samples = [os.path.basename(f).replace('_vid_lbl.npy', '')
                               for f in sorted(glob.glob(os.path.join(self.lbl_dir, '*_vid_lbl.npy')))]
            ecg_lbl_samples = [os.path.basename(f).replace('_ecg_lbl.npy', '')
                               for f in sorted(glob.glob(os.path.join(self.lbl_dir, '*_ecg_lbl.npy')))]
            # Intersection to ensure consistency across modalities
            common = set(ecg_samples) & set(flow_samples) & set(pose_samples) & set(vid_lbl_samples) & set(ecg_lbl_samples) & set(eeg_samples)
            # Preserve order based on ECG listing
            samples = [s for s in ecg_samples if s in common]
            return generate_paths(samples)

        # Specific fold: return train and test splits from text files
        train_samples = read_samples('train', num_fold)
        test_samples  = read_samples('test',  num_fold)

        train_data = generate_paths(train_samples)
        test_data  = generate_paths(test_samples)
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
            
    def calculate_class_weights(self, data_paths_dict, num_target_classes):
        """
        Calculates class weights based on the ENet formula.
        Args:
            data_paths_dict (dict): Dictionary of data paths, e.g., from get_folds().
            num_target_classes (int): Number of target classes (e.g., len(config['super_classes'])).
        Returns:
            torch.Tensor: Tensor of class weights.
        """
        # Using ecg_lbl_paths as the source for labels, assuming consistency
        # You could also use 'flow_lbls'
        label_files = data_paths_dict.get('ecg_lbls', data_paths_dict.get('flow_lbls'))
        if not label_files:
            print("Warning: No label files found in data_paths_dict for weight calculation. Returning uniform weights.")
            return torch.ones(num_target_classes, dtype=torch.float32)

        label_counts = torch.zeros(num_target_classes, dtype=torch.float64)
        
        print(f"Calculating class weights for {num_target_classes} classes...")
        for idx, label_path in enumerate(label_files):
            if idx % 100 == 0 or idx == len(label_files) - 1:
                print(f"Processing label file {idx+1}/{len(label_files)} for weights")
            try:
                raw_labels = np.load(label_path) # This is a 1D array of labels
            except FileNotFoundError:
                print(f"Warning: Label file not found {label_path}, skipping.")
                continue
            except Exception as e:
                print(f"Warning: Could not load label file {label_path} due to {e}, skipping.")
                continue

            # 1. Apply ignore_postictal filtering (if configured)
            if self.config.get('ignore_postictal', False):
                # Assuming postictal is class 5, as in SlidingWindowMMELoader logic
                raw_labels = raw_labels[raw_labels != 5] 
                if raw_labels.size == 0:
                    continue

            # 2. Map raw_labels to target super_classes
            # This logic must match how targets are derived for the loss function.
            # Current config['super_classes'] = ['baseline', 'seizure'] (num_target_classes = 2)
            # Original sub_classes are ['baseline', 'focal', 'tonic', 'clonic', 'pnes'] (0, 1, 2, 3, 4)
            
            processed_labels_for_counting = np.array([]) # Ensure it's an array

            if num_target_classes == 2:
                # Maps 0 (baseline) to 0.
                # Maps 1 (focal), 2 (tonic), 3 (clonic), 4 (pnes) to 1 (seizure).
                processed_labels_for_counting = np.clip(raw_labels, 0, 1)
            # Add elif clauses here if you have other num_target_classes schemes, e.g.:
            # elif num_target_classes == 3 and self.config.get('super_classes') == ['baseline', 'gtcs', 'pnes']:
            #     def map_to_3_classes(x):
            #         if x == 0: return 0       # baseline
            #         if x in [1, 2, 3]: return 1 # gtcs (focal, tonic, clonic)
            #         if x == 4: return 2       # pnes
            #         return -1 # To be filtered out
            #     temp_mapped = np.array([map_to_3_classes(x) for x in raw_labels])
            #     processed_labels_for_counting = temp_mapped[temp_mapped != -1]
            else:
                # Fallback: assumes raw_labels are already 0 to N-1.
                # This might need adjustment based on your specific label encoding if not 2 classes.
                print(f"Warning: Class weight calculation for {num_target_classes} classes using direct mapping. "
                    "Ensure raw labels are 0 to N-1 or add specific mapping logic.")
                processed_labels_for_counting = raw_labels[(raw_labels >= 0) & (raw_labels < num_target_classes)]

            if processed_labels_for_counting.size == 0:
                continue
                
            unique, counts = np.unique(processed_labels_for_counting, return_counts=True)
            
            for class_idx, count in zip(unique, counts):
                if 0 <= class_idx < num_target_classes:
                    label_counts[class_idx] += count
        
        if label_counts.sum() == 0:
            print("Warning: Total valid label count for weights is zero after processing. Returning uniform weights.")
            weights = torch.ones(num_target_classes, dtype=torch.float32)
        else:
            # Using ENet weighting: w_class = 1 / ln(1.02 + class_percentage)
            class_percentages = label_counts / label_counts.sum()
            weights = 1.0 / torch.log(1.02 + class_percentages)
            # Handle cases where a class might have 0 instances after all processing
            # Assign a weight similar to a class with very small representation
            weights[label_counts == 0] = 1.0 / torch.log(torch.tensor(1.02)) 
        
        # Move to GPU if configured and available
        if torch.cuda.is_available() and self.config.get('gpus_to_use', '-1') != '-1':
            return weights.float().cuda(torch.device(f"cuda:{self.config.get('gpus_to_use').split(',')[0]}"))
        return weights.float()


        
class SlidingWindowMMELoader(data.Dataset):
    def __init__(self, dataset_dict, config=config, augment=False):
        # file lists
        self.video_paths   = dataset_dict['flow_paths']
        self.pose_dirs     = dataset_dict['pose_paths']      # these point at body_coco.npy
        self.ecg_paths     = dataset_dict['ecg_paths']
        self.eeg_paths     = dataset_dict['eeg_paths']
        self.vid_lbl_paths = dataset_dict['flow_lbls']
        self.ecg_lbl_paths = dataset_dict['ecg_lbls']

        self.config   = config
        self.augment  = augment

        # window settings
        W = config['sample_duration']          # seconds
        ov = config['window_overlap'] # config.get('window_overlap', W/2) # seconds
        self.stride = W - ov                   # seconds

        # ECG CWT boilerplate
        self.sampling_period = 1. / config['ecg_freq']
        self.scales = pywt.central_frequency(config['wavelet']) \
                      * config['ecg_freq'] / np.arange(1, config['steps']+1)
        self.ecg_scaler     = MinMaxScaler((-1,1))
        self.ecg_seg_scaler = MinMaxScaler((0,1))
        self.seg_scale      = lambda x: self.ecg_seg_scaler.fit_transform(
                                    x.reshape(-1,1)).squeeze()

        # build a complete index of (record_idx, vid_start_frame, ecg_start_sample)
        self.mapping = []
        for ridx in range(len(self.video_paths)):
            fname = os.path.basename(self.vid_lbl_paths[ridx])
            # load labels only to measure length & clip post-ictal
            vid_lbls = np.load(self.vid_lbl_paths[ridx])
            ecg_lbls = np.load(self.ecg_lbl_paths[ridx])
            if config['ignore_postictal']:
                # keep = vid_lbls != 5
                # vid_lbls = vid_lbls[keep]
                # ecg_lbls = ecg_lbls[keep]
                
                filtered_indices = np.where(vid_lbls != 5)[0]
                vid_lbls = vid_lbls[filtered_indices]

                filtered_indices = np.where(ecg_lbls != 5)[0]
                ecg_lbls = ecg_lbls[filtered_indices]

            n_frames = len(vid_lbls)
            total_secs = n_frames / config['video_fps']
            if total_secs < W:
                continue

            n_windows = int(np.floor((total_secs - W) / self.stride)) + 1
            for w in range(n_windows):
                t0 = w * self.stride
                vf = int(t0 * config['video_fps'])
                ef = int(t0 * config['ecg_freq'])
                self.mapping.append((fname, ridx, vf, ef))

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
        fname, ridx, vid_start, ecg_start = self.mapping[index]
        # file stems
        filename = Path(self.video_paths[ridx]).stem

        #— load video & compute frame indices
        vr = VideoReader(self.video_paths[ridx],
                         width=self.config['video_width'],
                         height=self.config['video_height'])
        vid_idxs  = get_of_indices(self.config['sample_duration']) + vid_start

        # Clip video indices to valid range
        num_frames = len(vr)
        vid_idxs = np.clip(vid_idxs, 0, num_frames - 1)
        
        #— load poses
        base_pose = self.pose_dirs[ridx]
        body = np.load(base_pose)
        face = np.load(base_pose.replace('body_coco','face'))
        rh   = np.load(base_pose.replace('body_coco','r_hand'))
        lh   = np.load(base_pose.replace('body_coco','l_hand'))
        pose_idxs = get_pose_indices(self.config['sample_duration']) + vid_start

        # Clip pose indices to valid range
        num_pose_frames = body.shape[1]
        pose_idxs = np.clip(pose_idxs, 0, num_pose_frames - 1)
        
        #— load raw ECG
        ecg = np.load(self.ecg_paths[ridx])
        ecg_end = ecg_start + self.config['sample_duration'] * self.config['ecg_freq']
        
        # Clip ECG indices to valid range
        ecg_length = len(ecg)
        ecg_start = min(ecg_start, ecg_length - 1)
        ecg_end = min(ecg_end, ecg_length)
        
        #- load ECG HRV
        hrv = np.load(self.ecg_paths[ridx].replace('ecg','hrv'))

        # — load EEG
        eeg = np.load(self.eeg_paths[ridx])
        
        #— load & clip labels
        vid_lbls = np.load(self.vid_lbl_paths[ridx])
        ecg_lbls = np.load(self.ecg_lbl_paths[ridx])
        if self.config['ignore_postictal']:
            # mask = vid_lbls != 5
            # vid_lbls = vid_lbls[mask]
            # ecg_lbls = ecg_lbls[mask]
            filtered_indices = np.where(vid_lbls != 5)[0]
            vid_lbls = vid_lbls[filtered_indices]
            filtered_indices = np.where(ecg_lbls != 5)[0]
            ecg_lbls = ecg_lbls[filtered_indices]

        vid_lbl_seg = vid_lbls[vid_idxs]
        ecg_lbl_seg = ecg_lbls[ecg_start:ecg_end]

        #— assemble raw inputs
        frames = vr.get_batch(vid_idxs).asnumpy()
        body   = body[:, pose_idxs, :, :]
        face   = face[:, pose_idxs, :, :]
        rh     = rh[:,   pose_idxs, :, :]
        lh     = lh[:,   pose_idxs, :, :]
        ecg_seg = ecg[ecg_start:ecg_end]
        eeg = eeg[:, ecg_start:ecg_end]  # all channels, same length as ECG
        hrv     = hrv[:, ecg_start:ecg_end] # all 19 channels
        hrv = self.normalize_ecg_channel2(hrv, method='min_max') # we only z-score channel 2 i.e., heart rate
        
        #— generate soft labels
        sub_lbls = generate_soft_labels(ecg_lbl_seg, len(self.config['sub_classes']))
        if len(self.config['super_classes']) == 2:
            super_lbls = generate_soft_labels(ecg_lbl_seg.clip(0,1), 2)
        else:
            # example for 3 classes
            mapped = [0 if x==0 else 1 if x in [1,2,3] else 2 for x in ecg_lbl_seg]
            super_lbls = generate_soft_labels(mapped, 3)

        #— clean up missing keypoints
        for arr in (body, face, rh, lh):
            arr[arr[:,:,:,-1] == 0.5] = 0
            arr[arr[:,:,:,-1] == -0.5] = 0

        #— optional augment
        if self.augment:
            # frames = video_augment(frames)
            # body, face, rh, lh = pose_augment([body, face, rh, lh])
            # ecg_seg = ecg_augment(ecg_seg)
            frames = vid_augment.apply_video_augmentation(frames, p=0.7)
            hrv = ecg_augment.apply_hrv_augmentation(hrv, p=0.7)
            eeg = ecg_augment.apply_eeg_augmentation(eeg, p=0.7)
            body, face, rh, lh = pose_augment.apply_pose_augmentation(
                                                [body, face, rh, lh], p=0.7)
            
        #— CWT of ECG
        # ecg_coef, _ = pywt.cwt(ecg_seg, self.scales,
        #                        self.config['wavelet'],
        #                        self.sampling_period)
        # if self.augment:
        #     ecg_coef = cwt_augment(ecg_coef)
        # normalize video frames
        # frames = frames.astype(np.float32) / 255.0 # this is causing fluctuations.
        data = {
            'frames': frames,
            'body':   body,
            'face':   face,
            'rh':     rh,
            'lh':     lh,
            # 'ecg':    ecg_coef.astype(np.float32),
            'eeg':    eeg.astype(np.float32),
            'ecg_seg': self.seg_scale(ecg_seg),
            'hrv': hrv,
            'sub_lbls':   sub_lbls,
            'super_lbls': super_lbls,
            'filename':   filename
        }
        return data


#%%
