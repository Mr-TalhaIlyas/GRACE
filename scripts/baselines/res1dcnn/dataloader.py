#%%
from config import config
import torch
import torch.utils.data as data
from fmutils import fmutils as fmu
from data_utils import get_pose_indices, get_of_indices, generate_soft_labels
import decord as de
from decord import VideoReader
# from decord import cpu, gpu
import numpy as np
import cv2, os
import decord
from pathlib import Path
import pywt
from augmentors import video_augment, pose_augment, ecg_augment, cwt_augment
from sklearn.preprocessing import MinMaxScaler
# decord.bridge.set_bridge('torch')
#%%
class GEN_DATA_LISTS():

    def __init__(self, config):
        self.folds = config['folds']
        self.ecg_dir = config['ecg_dir']
        self.lbl_dir = config['lbl_dir']
        
    def get_folds(self, num_fold=1):
        def read_samples(file_type):
            path = f"{self.folds}/{file_type}_fold_{num_fold}.txt"
            with open(path, 'r') as file:
                samples = [line.strip() for line in file.readlines()]
            return samples

        def generate_paths(samples):
            data = {
                'ecg_paths': [f'{self.ecg_dir}{sample}.npy' for sample in samples],
                'ecg_lbls': [f'{self.lbl_dir}{sample}_ecg_lbl.npy' for sample in samples]
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
        self.ecg_lbls_paths = dataset_dict['ecg_lbls']
        # loade video
        self.ecg_paths = dataset_dict['ecg_paths']
        
        # all files should have same length
        assert len(self.ecg_lbls_paths) == len(self.ecg_paths)
        # dataloading parameters
        self.config = config
        self.augment = augment
        # ECG CWT settings
        self.sampling_period = 1. / config['ecg_freq']
        self.scales = pywt.central_frequency(config['wavelet']) * config['ecg_freq'] / np.arange(1, config['steps']+1, 1)
        self.ecg_scaler = MinMaxScaler((-1,1))
        self.ecg_seg_scaler = MinMaxScaler((0,1))
        self.seg_scale = lambda x: self.ecg_seg_scaler.fit_transform(x.reshape(-1,1)).squeeze()

    def __len__(self):
        return len(self.ecg_paths)

    def __getitem__(self, index):
        data_sample = {}
        # LODAING INPUT DATA
        filename = Path(self.ecg_paths[index]).stem
        
        # load ECG
        ecg = np.load(self.ecg_paths[index])

        # LODAING LABELS
        self.ecg_lbls = np.load(self.ecg_lbls_paths[index])
        
        if self.config['ignore_postictal']:
            filtered_indices = np.where(self.ecg_lbls != 5)[0]
            self.ecg_lbls = self.ecg_lbls[filtered_indices]

        # check max number of data points available in both video and ecg streams
        # after dividing them by their frequency/fps the data points should be same
        # and after shifting to max we still need to have same number of data points
        # for extraction so subtract the sample duration from the total length.
        ecg_max_sift_in_seconds = len(self.ecg_lbls) // config['ecg_freq'] - config['sample_duration']
        
        # this is the max shift in seconds we can do
        max_sift_in_seconds = ecg_max_sift_in_seconds

        # By setting alpha much higher than beta, the distribution is skewed towards the upper limit.
        # because seizuer duration is quite small 
        shift_seconds = int(np.random.beta(a=config['alpha'], b=config['beta'], size=1) * max_sift_in_seconds)

        # this shift will move the window randomly on the entire recording to get the required sample
        ecg_shift = shift_seconds * config['ecg_freq']
        ecg_end_point = ecg_shift + (config['sample_duration'] * config['ecg_freq'])



        # get input sample

        ecg_seg = ecg[ecg_shift:ecg_end_point]
        # get labels
        ecg_lbls = self.ecg_lbls[ecg_shift:ecg_end_point]

        # the vid_lbl and ecg_lbls have same distribution
        # so get only one.
        sub_lbls = generate_soft_labels(ecg_lbls, len(self.config['sub_classes']))
        if len(config['super_classes']) == 2:
            super_lbls = generate_soft_labels(np.clip(ecg_lbls, 0, 1), 2)
        elif len(config['super_classes']) == 3:
            # cluster 1,2,3 into 1 class
            ecg_lbls = [0 if x==0 else 1 if x in [1,2,3] else 2 for x in ecg_lbls]
            super_lbls = generate_soft_labels(ecg_lbls, 3)
        
        # generate ecg cwt
        # ecg_coef, _ = pywt.cwt(ecg_seg, self.scales, config['wavelet'], self.sampling_period)
        # # ecg_coef = self.ecg_scaler.fit_transform(ecg_coef) # [-1, 1]
        # if self.augment:
        #     ecg_coef = cwt_augment(ecg_coef)
        # Ensure labels are probabilities (sum to 1 and non-negative)
        # sub_lbls = sub_lbls / sub_lbls.sum(dim=1, keepdim=True)
        # super_lbls = super_lbls / super_lbls.sum(dim=1, keepdim=True)

        # data_sample['ecg'] = ecg_coef.astype(np.float32)
        data_sample['ecg_seg'] = ecg_seg#self.seg_scale(ecg_seg) # scale to [0,1]
        data_sample['sub_lbls'] = sub_lbls
        data_sample['super_lbls'] = super_lbls

        data_sample['filename'] = filename

        return data_sample
# %%

class BiasedSampler(data.Sampler):
    def __init__(self, data_source, bias_prefix='a_', bias_factor=2):
        """
        data_source: the dataset object (instance of MME_Loader).
        bias_prefix: the prefix of filenames you want to sample more frequently.
        bias_factor: how much more frequently to sample files with the bias_prefix.
        """
        self.data_source = data_source
        self.bias_prefix = bias_prefix
        self.bias_factor = bias_factor
        self.indices = self._create_biased_indices()
        
    def _create_biased_indices(self):
        biased_indices = []
        # Use video_paths to check the prefix
        for idx, filepath in enumerate(self.data_source.video_paths):
            filename = Path(filepath).name  # Use .name for the full filename
            if filename.startswith(self.bias_prefix):
                # Add the index multiple times to increase its sampling probability
                biased_indices.extend([idx] * self.bias_factor)
            else:
                biased_indices.append(idx)
        return biased_indices
    
    def __iter__(self):
        # Randomly shuffle the indices each epoch
        np.random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)