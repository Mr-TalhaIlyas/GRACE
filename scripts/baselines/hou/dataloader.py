#%%
from config import config
import torch
import torch.utils.data as data
from fmutils import fmutils as fmu
from utils import get_pose_indices, get_of_indices, generate_soft_labels
import decord as de
from decord import VideoReader
# from decord import cpu, gpu
import numpy as np
import cv2, os
import decord
from pathlib import Path
import pywt
# from data.augmentors import video_augment, pose_augment, ecg_augment, cwt_augment
from sklearn.preprocessing import MinMaxScaler
# decord.bridge.set_bridge('torch')
#%%
class GEN_DATA_LISTS():

    def __init__(self, config):
        self.folds = config['folds']
        self.flow_dir = config['flow_dir']
        self.pose_dir = config['pose_dir']
        self.lbl_dir = config['lbl_dir']

    def get_folds(self, num_fold=1):
        def read_samples(file_type):
            path = f"{self.folds}/{file_type}_fold_{num_fold}.txt"
            with open(path, 'r') as file:
                samples = [line.strip() for line in file.readlines()]
            return samples

        def generate_paths(samples):
            data = {
                'flow_paths': [f'{self.flow_dir}{sample}_flow.mp4' for sample in samples],
                'pose_paths': [f'{self.pose_dir}{sample}_body.npy' for sample in samples],
                'flow_lbls': [f'{self.lbl_dir}{sample}_vid_lbl.npy' for sample in samples]
            }
            return data

        train_samples = read_samples('train')
        test_samples = read_samples('test')

        train_data = generate_paths(train_samples)
        test_data = generate_paths(test_samples)

        return train_data, test_data
    
    def chk_paths(self, data):
        error_flag = False
        for key, paths in data.items():
            for path in paths:
                if not os.path.exists(path):
                    print(f"The path {path} does not exist.")
                    error_flag = True
        if not error_flag:
            print("All paths exist.")
        

#%%
class MME_Loader(data.Dataset):
    def __init__(self, dataset_dict, config=config, augment=False):
        # get labels paths
        self.vid_lbls_paths = dataset_dict['flow_lbls']
        # loade video
        self.pose_paths = dataset_dict['pose_paths']
        self.video_paths = dataset_dict['flow_paths']

        # dataloading parameters
        self.config = config
        self.augment = augment
        
    def __len__(self):
        return len(self.video_paths)

    def pose_denorm_cent(self, pose_array, H, W):
        # centralization
        pose_array[0:2,...] = pose_array[0:2,...] + 0.5
        # denormalize
        pose_array[0:1,:,:,:] = pose_array[0:1,:,:,:] * W
        pose_array[1:2,:,:,:] = pose_array[1:2,:,:,:] * H
        return pose_array

    def genFaceCropfromKpts(self, face_kpts, flow_frames, expand_pixels=20):
        """
        Generate face crops from face keypoints.

        Parameters:
        - face_kpts: numpy.ndarray
            Array of shape (3, T, num_keypoints, 1), where:
            - 3: Coordinates (x, y, confidence)
            - T: Number of time steps (frames)
            - num_keypoints: Number of face keypoints (e.g., 43 after removing outline)
            - 1: Number of persons

        - flow_frames: numpy.ndarray
            Array of shape (T, H, W, C), representing the flow frames.

        - expand_pixels: int
            Number of pixels to expand the bounding box on all sides.

        Returns:
        - face_crops_array: numpy.ndarray
            Array of shape (T, 48, 48, 3), containing the resized face crops.
        """
        
        # As GCN takes more inputs like 150 frames kpts as compared to 48 for slowFast
        # for a 10 sec video, so resample 48 kpts from 150 kpts of face to extract face crops
        
        n_frames = flow_frames.shape[0]
        indices = np.linspace(0, n_frames - 1, num=48, dtype=int) # sampled at uniform intervals
        face_kpts = face_kpts[:, indices, :, :]        
        
        # De-normalize the keypoints
        face_kpts = self.pose_denorm_cent(face_kpts,
                                          H=flow_frames.shape[1],
                                          W=flow_frames.shape[2])
        
        T = face_kpts.shape[1]  # Number of time steps
        H, W, C = flow_frames.shape[1], flow_frames.shape[2], flow_frames.shape[3]
        face_crops = []

        for t in range(T):
            # Extract keypoints for time step t
            keypoints = face_kpts[:, t, :, 0]  # Shape: (3, num_keypoints)

            # Get x, y coordinates and confidences
            x_coords = keypoints[0, :]
            y_coords = keypoints[1, :]
            confs = keypoints[2, :]

            # Filter out keypoints with low confidence (optional)
            confidence_threshold = 0.1
            valid_indices = confs > confidence_threshold
            x_coords = x_coords[valid_indices]
            y_coords = y_coords[valid_indices]

            # Check if there are any valid keypoints
            if len(x_coords) == 0 or len(y_coords) == 0:
                # If no valid keypoints, append a black image
                face_crop = np.zeros((48, 48, 3), dtype=np.uint8)
                face_crops.append(face_crop)
                continue

            # Compute bounding box
            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)

            # Expand bounding box
            min_x -= expand_pixels
            max_x += expand_pixels
            min_y -= expand_pixels
            max_y += expand_pixels

            # Ensure bounding box is within image boundaries
            min_x = max(0, int(min_x))
            min_y = max(0, int(min_y))
            max_x = min(W - 1, int(max_x))
            max_y = min(H - 1, int(max_y))

            # Handle cases where min >= max (invalid bounding box)
            if min_x >= max_x or min_y >= max_y:
                # Append a black image if bounding box is invalid
                face_crop = np.zeros((48, 48, 3), dtype=np.uint8)
                face_crops.append(face_crop)
                continue

            # Crop the image
            frame_t = flow_frames[t]  # Shape: (H, W, C)
            crop_img = frame_t[min_y:max_y, min_x:max_x, :]  # Shape: (crop_height, crop_width, C)

            # Resize the cropped image to 48x48 pixels
            resized_img = cv2.resize(crop_img, (48, 48), interpolation=cv2.INTER_LINEAR)

            # Append the resized image to the list
            face_crops.append(resized_img)

        # Stack the list into an array of shape (T, 48, 48, 3)
        face_crops_array = np.stack(face_crops, axis=0)
        # min-max normalization crops
        scaler = MinMaxScaler(feature_range=(0,255))
        face_crops_array = scaler.fit_transform(face_crops_array.reshape(-1, 48*48*3)).reshape(face_crops_array.shape)
        
        return face_crops_array.astype(np.uint8)
        
    
    def __getitem__(self, index):
        data_sample = {}
        # LODAING INPUT DATA
        filename = Path(self.video_paths[index]).stem
        # load video in memory
        vr = de.VideoReader(self.video_paths[index],
                            width=config['video_width'],
                            height=config['video_height'],
                            # ctx=gpu(1),
                            # num_threads=4
                            )
        #load poses body, face, hands
        body_pose = np.load(self.pose_paths[index])
        face_pose = np.load(self.pose_paths[index].replace('body', 'face'))
        # LODAING LABELS
        self.vid_lbls = np.load(self.vid_lbls_paths[index])

        if self.config['ignore_postictal']:
            filtered_indices = np.where(self.vid_lbls != 5)[0]
            self.vid_lbls = self.vid_lbls[filtered_indices]

        # check max number of data points available in both video and ecg streams
        # after dividing them by their frequency/fps the data points should be same
        # and after shifting to max we still need to have same number of data points
        # for extraction so subtract the sample duration from the total length.
        vid_max_sift_in_seconds = len(self.vid_lbls) // config['video_fps'] - config['sample_duration']
        # this is the max shift in seconds we can do
        max_sift_in_seconds = vid_max_sift_in_seconds

        # By setting alpha much higher than beta, the distribution is skewed towards the upper limit.
        # because seizuer duration is quite small and is at the end of recording
        shift_seconds = int(np.random.beta(a=config['alpha'], b=config['beta'], size=1) * max_sift_in_seconds)

        # this shift will move the window randomly on the entire recording to get the required sample
        shift = shift_seconds * config['video_fps']
        ecg_shift = shift_seconds * config['ecg_freq']
        ecg_end_point = ecg_shift + (config['sample_duration'] * config['ecg_freq'])

        vid_indices = get_of_indices(config['sample_duration']) + shift
        pose_indices = get_pose_indices(config['sample_duration']) + shift

        # get input sample
        frames = vr.get_batch(vid_indices).asnumpy()
                        
        # handel indexing error
        # get pose features
        body = body_pose[:, pose_indices, :, :]
        face = face_pose[:, pose_indices, :, :]
        
        # genereate center heatmaps
        face_frames = self.genFaceCropfromKpts(face, frames, expand_pixels=10)

        # get labels
        vid_lbls = self.vid_lbls[vid_indices]
        # ecg_lbls = self.ecg_lbls[ecg_shift:ecg_end_point]

        # the vid_lbl and ecg_lbls have same distribution
        # so get only one.
        sub_lbls = generate_soft_labels(vid_lbls, len(self.config['sub_classes']))
        if len(config['super_classes']) == 2:
            super_lbls = generate_soft_labels(np.clip(vid_lbls, 0, 1), 2)
        elif len(config['super_classes']) == 3:
            # cluster 1,2,3 into 1 class
            vid_lbls = [0 if x==0 else 1 if x in [1,2,3] else 2 for x in vid_lbls]
            super_lbls = generate_soft_labels(vid_lbls, 3)
        
        data_sample['frames'] = frames
        data_sample['face_frames'] = face_frames
        data_sample['body'] = body
        data_sample['face'] = face
        data_sample['sub_lbls'] = sub_lbls
        data_sample['super_lbls'] = super_lbls

        data_sample['filename'] = filename

        return data_sample
# %%
