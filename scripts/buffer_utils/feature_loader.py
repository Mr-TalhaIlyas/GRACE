import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import random
from collections import Counter

class SequenceFeatureDataset(Dataset): # Assuming this is your SequenceFeatureDatasetV2
    def __init__(self, base_feature_dir, data_split, feature_type,
                 sequence_length=6, stride=1,
                 augmentation_pad_prob=0.0, min_valid_on_pad=1):
        """
        Args:
            base_feature_dir (str): Base directory where features are stored.
            data_split (str): "train" or "val".
            feature_type (str): Type of feature to load (e.g., "mod_fusion_feats", "ecg_feats").
            sequence_length (int): Number of 10s windows to stack.
            stride (int): Step size for creating sequences.
            augmentation_pad_prob (float): Probability of applying padding augmentation.
            min_valid_on_pad (int): Minimum number of real features if padding occurs (must be < sequence_length).
        """
        self.base_feature_dir = base_feature_dir
        self.data_split = data_split
        self.feature_type = feature_type
        self.sequence_length = sequence_length
        self.stride = stride
        self.num_final_classes = 3
        self.augmentation_pad_prob = augmentation_pad_prob
        
        if min_valid_on_pad >= sequence_length and augmentation_pad_prob > 0:
            raise ValueError("min_valid_on_pad must be less than sequence_length for padding augmentation.")
        self.min_valid_on_pad = min_valid_on_pad

        self.split_dir = os.path.join(self.base_feature_dir, self.data_split)
        
        targets_path = os.path.join(self.split_dir, "targets.npy")
        if not os.path.exists(targets_path):
            raise FileNotFoundError(f"targets.npy not found in {self.split_dir}")
        self.all_window_original_targets = np.load(targets_path)
        
        self.label_remapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}

        feat_dir = os.path.join(self.split_dir, self.feature_type)
        if not os.path.isdir(feat_dir):
            raise FileNotFoundError(f"Feature directory not found: {feat_dir} for type {self.feature_type}")
        
        self.feature_files = sorted(glob.glob(os.path.join(feat_dir, "feature_*.npy")))
        
        if not self.feature_files:
            raise ValueError(f"No feature files found in {feat_dir} for feature type {self.feature_type}")

        if len(self.feature_files) != len(self.all_window_original_targets):
            raise ValueError(f"Mismatch in number of feature files ({len(self.feature_files)}) "
                             f"and targets ({len(self.all_window_original_targets)}) "
                             f"for feature type {self.feature_type}.")
        
        self.total_windows = len(self.feature_files)
        
        if self.total_windows > 0:
            sample_feat = np.load(self.feature_files[0])
            self.feature_dim = sample_feat.shape[-1] 
        else:
            self.feature_dim = 512 

        self.indices = []
        for i in range(0, self.total_windows - self.sequence_length + 1, self.stride):
            self.indices.append(i)

        if not self.indices:
            print(f"Warning: No sequences could be formed for {data_split}/{feature_type} "
                  f"with sequence_length={sequence_length} and stride={stride}. "
                  f"Total windows: {self.total_windows}")

    def _remap_and_get_target_label(self, original_window_labels_for_sequence):
        """
        Remaps original labels and creates a final 3-class label (one-hot or probabilistic).
        Args:
            original_window_labels_for_sequence (list/np.array): List of original labels for the windows in the sequence.
        """
        if not len(original_window_labels_for_sequence):
            return torch.zeros(self.num_final_classes, dtype=torch.float32)

        remapped_labels = [self.label_remapping.get(lbl, -1) for lbl in original_window_labels_for_sequence]
        remapped_labels = [lbl for lbl in remapped_labels if lbl != -1]

        if not remapped_labels: 
             return torch.zeros(self.num_final_classes, dtype=torch.float32)

        if len(set(remapped_labels)) == 1:
            final_label_idx = remapped_labels[0]
            target_label = torch.zeros(self.num_final_classes, dtype=torch.float32)
            if 0 <= final_label_idx < self.num_final_classes:
                target_label[final_label_idx] = 1.0
        else: 
            label_counts = Counter(remapped_labels)
            target_label = torch.zeros(self.num_final_classes, dtype=torch.float32)
            for label_idx, count in label_counts.items():
                if 0 <= label_idx < self.num_final_classes:
                    target_label[label_idx] = count / len(remapped_labels)
        return target_label
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_window_idx = self.indices[idx]
        
        apply_padding_augmentation = random.random() < self.augmentation_pad_prob
        
        num_valid_features_this_seq = self.sequence_length # Initialize with full length
        
        if apply_padding_augmentation and self.sequence_length > self.min_valid_on_pad :
            num_valid_features_this_seq = random.randint(self.min_valid_on_pad, self.sequence_length - 1)

        actual_features_to_load = []
        for i in range(num_valid_features_this_seq): # Load only valid features
            current_window_file_idx = start_window_idx + i
            if current_window_file_idx < self.total_windows:
                feat_file = self.feature_files[current_window_file_idx]
                feature_data = np.load(feat_file).squeeze() 
                if feature_data.ndim == 0: 
                    feature_data = np.array([feature_data] * self.feature_dim) 
                elif feature_data.shape[0] != self.feature_dim: 
                     raise ValueError(f"Feature dim mismatch in {feat_file}. Expected {self.feature_dim}, got {feature_data.shape}")
                actual_features_to_load.append(feature_data)
            else: 
                actual_features_to_load.append(np.zeros(self.feature_dim, dtype=np.float32))
        
        # Pad if necessary
        sequence_parts = actual_features_to_load
        num_to_pad = self.sequence_length - len(actual_features_to_load)
        if num_to_pad > 0:
            for _ in range(num_to_pad):
                sequence_parts.append(np.zeros(self.feature_dim, dtype=np.float32))
            
        sequence_tensor = torch.tensor(np.array(sequence_parts), dtype=torch.float32)

        original_labels_for_target_derivation = self.all_window_original_targets[
            start_window_idx : start_window_idx + num_valid_features_this_seq # Use actual number of valid features
        ]
        
        target_label_tensor = self._remap_and_get_target_label(original_labels_for_target_derivation)
            
        return sequence_tensor, target_label_tensor, torch.tensor(num_valid_features_this_seq, dtype=torch.long)
