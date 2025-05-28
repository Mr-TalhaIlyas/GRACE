#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 08:38:35 2023

@author: talha

Adapted form: https://github.com/fpv-iplab/rulstm/
"""

import cv2
import lmdb
import numpy as np
import torch
from torchvision import transforms
from model import ResNet2plus1D
from tqdm import tqdm, trange
from PIL import Image
from pathlib import Path
from fmutils import fmutils as fmu
from sklearn.preprocessing import MinMaxScaler
scalar  = MinMaxScaler(feature_range=(0, 255))
# function to convert BGR to optical flow
def bgr_to_flow(bgr_image):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h = hsv_image[..., 0] / 180.0 * np.pi * 2
    v = hsv_image[..., 2] / 255.0
    u = v * np.cos(h)
    v = v * np.sin(h)

    u = scalar.fit_transform(u).astype(np.uint8)
    v = scalar.fit_transform(v).astype(np.uint8)
    return Image.fromarray(u).convert('L'), Image.fromarray(v).convert('L')

# initialize the LMDB environment
env = lmdb.open('../Optical Flow Pre/op/', map_size=1024**3)

# init the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet2plus1D(frames_per_clip=32, remove_fc=True)
model.to(device)
model.eval()

#  the transformation
transform = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255),
    transforms.Normalize(mean=[128], std=[1]),
])
#%%
all_keys = []
# Process the video
video_paths = fmu.get_all_files('/home/talha/data/of/vid/')

for i in trange(len(video_paths)):
    video_path = video_paths[i]  
    feat_sec = 1
    cap = cv2.VideoCapture(video_path)
    vid_name = Path(video_path).stem.replace('cvof_','')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = 30#int(cap.get(cv2.CAP_PROP_FPS)) # fix these for all
    sampling_stride = frame_rate // 6  # achieve 5FPS
    
    flow_buffer = []
    for f in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
    
        if f % sampling_stride == 0:
            flow_u, flow_v = bgr_to_flow(frame) # as we operate on optical flow so first convert to optical flow
            # make 3 channels repeat the uvv
            flow = np.stack([np.array(flow_u), np.array(flow_v), np.zeros_like(np.array(flow_u))], axis=-1)
            flow = Image.fromarray(flow)
            # add in buffer
            flow_buffer.append(transform(flow))
    

        if len(flow_buffer) == 32: # for pretrained model
            data = torch.stack(flow_buffer, dim=0).to(device) # B*T*C*H*W == B*32*3*112*112
            data = data.permute(0,2,1,3,4) # B*C*T*H*W
            feat = model(data).detach().cpu().numpy() # outputs Bx512 vector 
            # key = f"{vid_name}_frame_{_ // sampling_stride}"
            key = f"{vid_name}_sec_{feat_sec}"
            with env.begin(write=True) as txn:
                txn.put(key.encode(), feat.tobytes())
            flow_buffer = []
            feat_sec += 1
            all_keys.append(key)
    cap.release()
    
