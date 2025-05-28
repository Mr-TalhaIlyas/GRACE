#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:01:32 2024

@author: talha
"""
from configs.config import config
import os
from data.utils import get_pose_indices, get_of_indices, generate_soft_labels
from configs.config import config
import decord as de
from pathlib import Path
import numpy as np
from data.dataloader import GEN_DATA_LISTS
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import pywt

sampling_period = 1. / config['ecg_freq']
scales = pywt.central_frequency(config['wavelet']) * config['ecg_freq'] / np.arange(1, config['steps']+1, 1)
ecg_scaler = MinMaxScaler((-1,1))

ecg_seg_scaler = MinMaxScaler((0,1))
seg_scale = lambda x: ecg_seg_scaler.fit_transform(x.reshape(-1,1)).squeeze()



def get_infer_sample(dataset_dict, sample_idx=0, overlap_sec=3):

    index = sample_idx
    
    vid_lbls_paths = dataset_dict['flow_lbls']
    ecg_lbls_paths = dataset_dict['ecg_lbls']
    # loade video
    ecg_paths = dataset_dict['ecg_paths']
    pose_paths = dataset_dict['pose_paths']
    video_paths = dataset_dict['flow_paths']
            
    data_sample = {}
    # LODAING INPUT DATA
    filename = Path(video_paths[index]).stem
    # load video in memory
    vr = de.VideoReader(video_paths[index],
                        width=config['video_width'],
                        height=config['video_height'],
                        # ctx=gpu(1),
                        # num_threads=4
                        )
    #load poses body, face, hands
    body_pose = np.load(pose_paths[index])
    face_pose = np.load(pose_paths[index].replace('body_coco', 'face'))
    rh_pose = np.load(pose_paths[index].replace('body_coco', 'r_hand'))
    lh_pose = np.load(pose_paths[index].replace('body_coco', 'l_hand'))
    # load ECG
    ecg = np.load(ecg_paths[index])
    
    # LODAING LABELS
    ecg_lbls = np.load(ecg_lbls_paths[index])
    vid_lbls = np.load(vid_lbls_paths[index])
    
    if config['ignore_postictal']:
        filtered_indices = np.where(vid_lbls != 5)[0]
        vid_lbls = vid_lbls[filtered_indices]
    
        filtered_indices = np.where(ecg_lbls != 5)[0]
        ecg_lbls = ecg_lbls[filtered_indices]
    
    # check max number of data points available in both video and ecg streams
    # after dividing them by their frequency/fps the data points should be same
    # and after shifting to max we still need to have same number of data points
    # for extraction so subtract the sample duration from the total length.
    vid_max_sift_in_seconds = len(vid_lbls) // config['video_fps'] - config['sample_duration']
    ecg_max_sift_in_seconds = len(ecg_lbls) // config['ecg_freq'] - config['sample_duration']
    
    assert vid_max_sift_in_seconds == ecg_max_sift_in_seconds
    # this is the max shift in seconds we can do
    max_sift_in_seconds = vid_max_sift_in_seconds
    
    # here instead of getting shift seconds we will window size and instead of adding a 
    # random shift to the indices (i.e., get_of_indices) to get one window we will start
    # from 0 and keep shiftign towards the end of recording.
    # Here we'll move +window+ seconds till we reach +max_sift_in_seconds+
    window = config['sample_duration'] - overlap_sec
    
    sliding_windows = []
    for step in range(0, max_sift_in_seconds+config['sample_duration'], window):
        shift_seconds = step
                
        if (max_sift_in_seconds+config['sample_duration'])-shift_seconds < config['sample_duration']:
            # shift_seconds = shift_seconds - (config['sample_duration'] - ((max_sift_in_seconds+config['sample_duration'])-shift_seconds))
            shift_seconds = max_sift_in_seconds # simplified version of above line
        if shift_seconds == max_sift_in_seconds+config['sample_duration']:
            break
        
        sliding_windows.append(shift_seconds)
        
    sliding_windows = np.unique(np.asarray(sliding_windows))
    # print(sliding_windows)
    all_frames, all_body, all_face, all_rh, all_lh, all_ecg, all_sub_lbls, all_sup_lbls = [],[],[],[],[],[], [], []
    
    # for shift_seconds in tqdm(sliding_windows, total=len(sliding_windows), desc=f'Loading {filename}'):
    for shift_seconds in sliding_windows:
        try: # try will get index error for videos who have seizur till end of recording
            # this shift will move the window randomly on the entire recording to get the required sample
            shift = shift_seconds * config['video_fps']
            ecg_shift = shift_seconds * config['ecg_freq']
            ecg_end_point = ecg_shift + (config['sample_duration'] * config['ecg_freq'])
            
            vid_indices = get_of_indices(config['sample_duration']) + shift
            pose_indices = get_pose_indices(config['sample_duration']) + shift
        
            # get input sample
            frames = vr.get_batch(vid_indices).asnumpy()
            body = body_pose[:, pose_indices, :, :]
            face = face_pose[:, pose_indices, :, :]
            rh = rh_pose[:, pose_indices, :, :]
            lh = lh_pose[:, pose_indices, :, :]
            ecg_seg = ecg[ecg_shift:ecg_end_point]
            # get labels
            vid_lbl = vid_lbls[vid_indices]
            ecg_lbl = ecg_lbls[ecg_shift:ecg_end_point]
            
            # the vid_lbl and ecg_lbls have same distribution
            # so get only one. OR you can use any of both.
            sub_lbls = generate_soft_labels(ecg_lbl, len(config['sub_classes']))
            if len(config['super_classes']) == 2:
                super_lbls = generate_soft_labels(np.clip(ecg_lbl, 0, 1), 2)
            elif len(config['super_classes']) == 3:
                # cluster 1,2,3 into 1 class
                ecg_lbl = [0 if x==0 else 1 if x in [1,2,3] else 2 for x in ecg_lbl]
                super_lbls = generate_soft_labels(ecg_lbl, 3)
            
            # Adjusting Pose arrays, Videos will be normalized in the trainer see video_transform
            # adjusting the unknow datapoints to zero, as pose is already normalized
            body[body[:, :, :, -1] == 0.5] = 0.0
            body[body[:, :, :, -1] == -0.5] = 0.0
            rh[rh[:, :, :, -1] == 0.5] = 0.0
            rh[rh[:, :, :, -1] == -0.5] = 0.0
            lh[lh[:, :, :, -1] == 0.5] = 0.0
            lh[lh[:, :, :, -1] == -0.5] = 0.0
            face[face[:, :, :, -1] == 0.5] = 0.0
            face[face[:, :, :, -1] == -0.5] = 0.0
            
            # generate ecg cwt
            ecg_coef, _ = pywt.cwt(ecg_seg, scales, config['wavelet'], sampling_period)
            # ecg_coef = ecg_scaler.fit_transform(ecg_coef) # [-1, 1]
            ecg_seg = seg_scale(ecg_seg) # scale to [0,1]
        except IndexError as e:
            print(f'Error: {e} @  {filename}')
            # pass

        all_frames.append(frames)
        all_body.append(body)
        all_face.append(face)
        all_rh.append(rh)
        all_lh.append(lh)
        all_ecg.append(ecg_seg) # ecg_coef-> this for AST ecg_seg -> for DILViT
        all_sub_lbls.append(sub_lbls)
        all_sup_lbls.append(super_lbls)
        
    
    data_sample['frames'] = np.asarray(all_frames) * 0
    data_sample['body'] = np.asarray(all_body)#*0
    data_sample['face'] = np.asarray(all_face)#*0
    data_sample['rh'] = np.asarray(all_rh)##*0
    data_sample['lh'] = np.asarray(all_lh)#*0
    data_sample['ecg'] = np.asarray(all_ecg)# * 0
    data_sample['sub_lbls'] = np.asarray(all_sub_lbls)
    data_sample['sup_lbls'] = np.asarray(all_sup_lbls)
    data_sample['filename'] = filename.strip('_flow')
    
    return data_sample, sliding_windows

# # Example
# data = GEN_DATA_LISTS(config)

# train_data, test_data = data.get_folds(config['num_fold'])

# dataset_dict = test_data

# data_sample, sliding_windows = get_infer_sample(dataset_dict, sample_idx=0, overlap_sec=3)

# sub_lbls = data_sample['sub_lbls']
# sup_lbls = data_sample['sup_lbls']
#%%

















