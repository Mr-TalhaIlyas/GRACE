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
from pretrainedmodels import bninception
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
model = bninception(pretrained=None)
model.conv1_7x7_s2 = torch.nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
state_dict = torch.load('../TSN-flow.pth.tar')['state_dict']
state_dict = {k.replace('module.base_model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.last_linear = torch.nn.Identity()
model.global_pool = torch.nn.AdaptiveAvgPool2d(1)
model.to(device)
model.eval()

#  the transformation
transform = transforms.Compose([
    transforms.Resize([256, 454]),
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
    sampling_stride = 1#frame_rate // 5  # achieve 5FPS
    
    flow_buffer = []
    for f in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
    
        if _ % sampling_stride == 0:
            flow_u, flow_v = bgr_to_flow(frame)
            flow_buffer.extend([transform(flow_u), transform(flow_v)])
    

        if len(flow_buffer) == 10:
            data = torch.cat(flow_buffer, dim=0).unsqueeze(0).to(device)
            feat = model(data).squeeze().detach().cpu().numpy()
            # key = f"{vid_name}_frame_{_ // sampling_stride}"
            key = f"{vid_name}_sec_{feat_sec}"
            with env.begin(write=True) as txn:
                txn.put(key.encode(), feat.tobytes())
            flow_buffer = []
            feat_sec += 1
            all_keys.append(key)
    cap.release()
    
