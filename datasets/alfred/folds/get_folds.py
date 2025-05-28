# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 19:11:09 2024

@author: talha
"""
from fmutils import fmutils as fmu
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

filenames = os.listdir('D:/MMAI/data/MMAI/data_v4/alfred/cv/ecg/')
filenames = [file.strip('.npy') for file in filenames]
# Generate labels: 0 for 'a' files and 1 for 'b' files
labels = [0 if name.startswith('a') else 1 for name in filenames]

# Prepare stratified K-fold
skf = StratifiedKFold(n_splits=5)

fold = 1
for train_index, test_index in skf.split(filenames, labels):
    # Split filenames based on stratified indices
    train_filenames = np.array(filenames)[train_index]
    test_filenames = np.array(filenames)[test_index]

    # Write to text files (for reproducibility)
    with open(f'D:/MMAI/data/MMAI/data_v4/alfred/folds/train_fold_{fold}.txt', 'w') as f:
        for item in train_filenames:
            f.write("%s\n" % item)
    
    with open(f'D:/MMAI/data/MMAI/data_v4/alfred/folds/test_fold_{fold}.txt', 'w') as f:
        for item in test_filenames:
            f.write("%s\n" % item)

    fold += 1

