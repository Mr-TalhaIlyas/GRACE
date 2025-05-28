import cv2
import glob
import json
import numpy as np
import os
from fmutils import fmutils as fmu
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from pathlib import Path
from utils import JOINT_PAIRS_MAP_ALL, HAND_JOINT_PAIRS_MAP_ALL, FACE_JOINT_PAIRS_MAP_ALL, joint_colors
import json

conf_thresh = 0.0

def draw_body(pose_kpts, img):
    
    pose_coords = pose_kpts[:, 0:2]
    confidences = pose_kpts[:, 2]
    
    for j1, j2 in JOINT_PAIRS_MAP_ALL.keys():
        
        # get values at index j1 and j2 and round them of to an int.
        x1, y1 = list(map(lambda v: int(round(v)), pose_coords[j1]))
        x2, y2 = list(map(lambda v: int(round(v)), pose_coords[j2]))
        c1, c2 = confidences[j1], confidences[j2]
        
        clr = joint_colors[JOINT_PAIRS_MAP_ALL[(j1, j2)]['joint_names'][1]]
        
        # draw body
        if c1 > conf_thresh:
            cv2.circle(img, (x1, y1), radius=4, color=(255, 128, 0), thickness=3)
        if c2 > conf_thresh:
            cv2.circle(img, (x2, y2), radius=4, color=(255, 128, 0), thickness=3)
        if c1 > conf_thresh and c2 > conf_thresh:
            cv2.line(img, (x1, y1), (x2, y2), color=clr, thickness=2)
    
    return img

def draw_hand(hand_kpts, img):
    
    pose_coords = hand_kpts[:, 0:2]
    confidences = hand_kpts[:, 2]
    
    for j1, j2 in HAND_JOINT_PAIRS_MAP_ALL.keys():
        
        # get values at index j1 and j2 and round them of to an int.
        x1, y1 = list(map(lambda v: int(round(v)), pose_coords[j1]))
        x2, y2 = list(map(lambda v: int(round(v)), pose_coords[j2]))
        c1, c2 = confidences[j1], confidences[j2]
        
        clr = HAND_JOINT_PAIRS_MAP_ALL[(j1, j2)]['color']
        
        # draw body
        if c1 > conf_thresh:
            cv2.circle(img, (x1, y1), radius=3, color=(255, 128, 0), thickness=2)
        if c2 > conf_thresh:
            cv2.circle(img, (x2, y2), radius=3, color=(255, 128, 0), thickness=2)
        if c1 > conf_thresh and c2 > conf_thresh:
            cv2.line(img, (x1, y1), (x2, y2), color=clr, thickness=2)
    
    return img

def draw_face(face_kpts, img):
    
    pose_coords = face_kpts[:, 0:2]
    confidences = face_kpts[:, 2]
    
    for j1, j2 in FACE_JOINT_PAIRS_MAP_ALL.keys():
        
        # get values at index j1 and j2 and round them of to an int.
        x1, y1 = list(map(lambda v: int(round(v)), pose_coords[j1]))
        x2, y2 = list(map(lambda v: int(round(v)), pose_coords[j2]))
        c1, c2 = confidences[j1], confidences[j2]
        
        # clr = HAND_JOINT_PAIRS_MAP_ALL[(j1, j2)]['color']
        
        # draw body
        if c1 > conf_thresh:
            cv2.circle(img, (x1, y1), radius=3, color=(255, 255, 0), thickness=2)
        if c2 > conf_thresh:
            cv2.circle(img, (x2, y2), radius=3, color=(255, 255, 0), thickness=2)
        if c1 > conf_thresh and c2 > conf_thresh:
            cv2.line(img, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)
    
    return img

def draw_bbox(pose_kpts, img):
    # Calculate the bounding box coordinates
    shift = 20
    max_x =int(max(pose_kpts[:,0][np.nonzero(pose_kpts[:,0])])) + shift
    min_x = int(min(pose_kpts[:,0][np.nonzero(pose_kpts[:,0])])) - shift
    
    max_y =int(max(pose_kpts[:,1][np.nonzero(pose_kpts[:,1])])) + shift
    min_y = int(min(pose_kpts[:,1][np.nonzero(pose_kpts[:,1])])) - shift

    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    return img

