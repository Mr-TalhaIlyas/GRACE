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

    def get_splits(self, ):
        base_dir = self.config['sahzu_data_dir']
        print(37*'-')
        print(f"Loading SAHZU data from {base_dir}")
        print('ONLY ONE SET IS AVAILABLE FOR SAHZU DATASET')
        print(37*'-')
        # loading train paths
        full_data = {
            'flow_paths': sorted(glob.glob(os.path.join(base_dir, 'flow', '*_uvv.mp4'))),
            'pose_paths':  sorted(glob.glob(os.path.join(base_dir, 'pose', '*', 'body_coco.npy'))),
            'lbls':   sorted(glob.glob(os.path.join(base_dir, 'labels', '*_vid_lbl.npy')))
        }
        # Check if all paths exist
        self.chk_paths(full_data)
        # Ensure all lists are of the same length
        if len(full_data['flow_paths']) != len(full_data['pose_paths']) or \
           len(full_data['flow_paths']) != len(full_data['lbls']):
            raise ValueError("Mismatch in number of Flow, Pose, or label files in data.")
        
        return full_data

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

# test_data = data_gen.get_splits()

# a,b,c = test_data.values()

# for i in range(len(a)):
#     assert Path(a[i]).stem.replace('_uvv','') == Path(b[i]).parent.absolute().stem
    
#     assert Path(a[i]).stem.replace('_uvv','') == Path(c[i]).stem.replace('_uvv_vid_lbl','')
#     print('Pass', Path(a[i]).stem.replace('_uvv',''))
#%%


  
class SlidingWindowVisualLoader(data.Dataset):
    def __init__(self, dataset_dict, config=config, augment=False, split=None):
        # file lists
        self.split = split
        self.video_paths   = dataset_dict['flow_paths']
        self.pose_dirs     = dataset_dict['pose_paths']
        self.vid_lbl_paths = dataset_dict['lbls']

        self.config   = config
        self.augment  = augment

        # window settings
        W = config['sample_duration']          # seconds
        ov = config['window_overlap'] # config.get('window_overlap', W/2) # seconds
        self.stride = W - ov                   # seconds

        # build a complete index of (record_idx, vid_start_frame, ecg_start_sample)
        self.mapping = []
        for ridx in range(len(self.video_paths)):
            fname = os.path.basename(self.vid_lbl_paths[ridx])
            # load labels only to measure length & clip post-ictal
            lbls = np.load(self.vid_lbl_paths[ridx])

            n_frames = len(lbls)
            total_secs = n_frames / config['video_fps']
            if total_secs < W:
                continue

            n_windows = int(np.floor((total_secs - W) / self.stride)) + 1
            for w in range(n_windows):
                t0 = w * self.stride
                vf = int(t0 * config['video_fps'])
                self.mapping.append((fname, ridx, vf))
        # Now if split is NONE then we will use all the data/mapping
        # and if split is train we'll use 70% of the data and if it's
        # val then we'll use 30% of the data.
        if self.split is not None:
            n_total = len(self.mapping)
            n_train = int(n_total * 0.7)
            if self.split == 'train':
                print(f"Loading split: {self.split} with {n_train} samples")
                self.mapping = self.mapping[:n_train]
            elif self.split == 'val':
                print(f"Loading split: {self.split} with {n_total - n_train} samples")
                self.mapping = self.mapping[n_train:]
            else:
                raise ValueError(f"Unknown split: {self.split}")

        assert len(self.mapping) > 0, "No sliding windows generated!"

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        fname, ridx, vid_start = self.mapping[index]
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
        
        #— load & clip labels
        lbls = np.load(self.vid_lbl_paths[ridx])

        lbl_seg = lbls[vid_idxs]
        #— assemble raw inputs
        frames = vr.get_batch(vid_idxs).asnumpy()
        body   = body[:, pose_idxs, :, :]
        face   = face[:, pose_idxs, :, :]
        rh     = rh[:,   pose_idxs, :, :]
        lh     = lh[:,   pose_idxs, :, :]
        
        #— generate sub, sup class labels
        sub_lbls = lbl_seg
        super_lbls = lbl_seg

        #— clean up missing keypoints
        for arr in (body, face, rh, lh):
            arr[arr[:,:,:,-1] == 0.5] = 0
            arr[arr[:,:,:,-1] == -0.5] = 0

        #— optional augment
        if self.augment:
            frames = vid_augment.apply_video_augmentation(frames, p=0.7)
            body, face, rh, lh = pose_augment.apply_pose_augmentation(
                                                [body, face, rh, lh], p=0.7)
            
        data = {
            'frames': frames,
            'body':   body,
            'face':   face,
            'rh':     rh,
            'lh':     lh,
            'sub_lbls':   sub_lbls[0],
            'super_lbls': super_lbls[0],
            'filename':   filename
        }
        return data























