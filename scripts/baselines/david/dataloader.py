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
                'pose_paths': [f'{self.pose_dir}{sample}.npy' for sample in samples],
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

    def genCenterMap(self, x, y, sigma, size_w, size_h):
        """
        generate Gaussian heat map
        :param x: center point
        :param y: center point
        :param sigma:
        :param size_w: image width
        :param size_h: image height
        :return:            numpy           w * h
        """
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - x) ** 2 + (gridy - y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)  # numpy 2d
    
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
        # genereate center heatmaps
        center_map = self.genCenterMap(x=config['video_width'] / 2.0, y=config['video_height'] / 2.0, sigma=21,
                                       size_w=config['video_width'], size_h=config['video_height'])        
        center_map = np.expand_dims(center_map, axis=0)
        # handel indexing error
        # get pose features
        try:
            body = body_pose[pose_indices, :]  # T, 200
            body = body.mean(axis=0) # get 200 dim hand crafted feature vector for 10 sec clip
            body = np.expand_dims(body, axis=0)
        except IndexError:
            # call again
            next_idx = (index + 1) % len(self.video_paths)
            return self.__getitem__(next_idx)
        
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
        data_sample['center_map'] = center_map
        data_sample['body'] = body
        data_sample['sub_lbls'] = sub_lbls
        data_sample['super_lbls'] = super_lbls

        data_sample['filename'] = filename

        return data_sample
# %%
