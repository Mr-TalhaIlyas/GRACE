#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:08:13 2024

@author: talha.ilyas@monash.edu
"""
#%%

import os
os.chdir('/home/user01/Data/npj/scripts/')
import joblib
import wandb
from tsai.all import *
import sklearn.metrics as skm
from functools import partial
from fastai.metrics import Precision, Recall, F1Score
from fastai.callback.wandb import WandbCallback
from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback
from fastai.losses import CrossEntropyLossFlat, LabelSmoothingCrossEntropyFlat
from fastai.learner import Learner
from fastai.metrics import accuracy
from torch.optim import AdamW
from fastai.optimizer import OptimWrapper
from fastai.callback.training import GradientClip
from tsai.data.mixed_augmentation import MixUp1d, IntraClassCutMix1d
from in_silo.utils.augs import ExclusiveMixer
from in_silo.models.dilvit import DILVIT
from in_silo.models.timemil import TimeMIL
#%%
exp_name = 'ecgH_bvs_InceptionTime_fold1' 
# Define your configuration for wandb
config = {
    'batch_size_train': 64,#64, 8 
    'batch_size_valid': 128,#128, 16
    'lr_max': 0.0001,#1e-5,
    'epochs': 100,
    'architecture': 'InceptionTime',
    'dataset': 'ALFRED',
    'data_dir': '/home/user01/Data/npj/datasets/alfred/ts_analysis/',
    'chkpt_dir': '/home/user01/Data/npj/scripts/in_silo/chkpt/',
    'log_dir': '/home/user01/Data/npj/scripts/in_silo/logs/',
    'exp_typ': 'bvs', # bvp, gvp, bvg None bvs
    'data_type': 'hrv', # hrv, eeg ecg
    # Add any other parameters you'd like to track
}

# Initialize wandb with your project and configuration
# wandb.init(dir=config['log_dir'], project="ALFRED_EEG", name = exp_name, config=config)

# Load your data
# Alfred_ecgH_full_w10_o5 Alfred_ecgH_fold1_w10_o5.joblib
data = joblib.load(f'{config["data_dir"]}Alfred_ecgH_fold1_w10_o5.joblib')

if config['data_type'] == 'ecg':
    X_train = data['train_data'][:,1:2,:]
else:
    X_train = data['train_data']
y_train = data['train_label']

if config['exp_typ'] == 'gvp':
    # filter incdices based on the labels
    idx = np.where(y_train != 0) # removed basline
    X_train = X_train[idx]
    y_train = y_train[idx] - 1 # to make it 0 based
elif config['exp_typ'] == 'bvg':
    # filter incdices based on the labels
    idx = np.where(y_train != 2) # removed pnes
    X_train = X_train[idx]
    y_train = y_train[idx]
elif config['exp_typ'] == 'bvp':
    # filter incdices based on the labels
    idx = np.where(y_train != 1) # removed gtcs
    X_train = X_train[idx]
    y_train = y_train[idx]
elif config['exp_typ'] == 'bvs':
    print('Baseline vs Seizure Experiment')
    # baseline vs seizuer (so group all seizures as 1)
    y_train = np.clip(y_train, 0, 1)
else:
    pass
y_train = y_train.astype(np.float16).astype(str)

if config['data_type'] == 'ecg':
    X_test = data['test_data'][:,1:2,:]
else:
    X_test = data['test_data']
y_test = data['test_label']

if config['exp_typ'] == 'gvp':
    # filter incdices based on the labels
    idx = np.where(y_test != 0) # removed basline
    X_test = X_test[idx]
    y_test = y_test[idx] - 1 # to make it 0 based
elif config['exp_typ'] == 'bvg':
    # filter incdices based on the labels
    idx = np.where(y_test != 2) # removed pnes
    X_test = X_test[idx]
    y_test = y_test[idx]
elif config['exp_typ'] == 'bvp':
    # filter incdices based on the labels
    idx = np.where(y_test != 1) # removed gtcs
    X_test = X_test[idx]
    y_test = y_test[idx]
elif config['exp_typ'] == 'bvs':
    print('Baseline vs Seizure Experiment')
    # baseline vs seizuer (so group all seizures as 1)
    y_test = np.clip(y_test, 0, 1)
else:
    pass

#%%
if config['data_type'] == 'hrv':
    # use sklearn minmax scaler to scale only the 3rd channel of data only
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    for i in range(X_train.shape[0]):
        ecg = X_train[i, 2, :].reshape(-1, 1)
        ecg = scaler.fit_transform(ecg)
        X_train[i, 2, :] = ecg.reshape(-1)
    for i in range(X_test.shape[0]):
        ecg = X_test[i, 2, :].reshape(-1, 1)
        ecg = scaler.fit_transform(ecg)
        X_test[i, 2, :] = ecg.reshape(-1)

#%%
y_test = y_test.astype(np.float16).astype(str)

# # Combine training and testing data
X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])

# print(X.shape, y.shape)
#%%
tfms = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
# Create DataLoaders with batch sizes from config
dls = TSDataLoaders.from_dsets(
    dsets.train,
    dsets.valid,
    bs=[config['batch_size_train'], config['batch_size_valid']],
    # batch_tfms=[TSStandardize(by_sample=False, by_var=False)],
    num_workers=0
)
#%%
# Optionally, display a batch
dls.show_batch(sharey=False)
print('Classes = ', dls.c, 'i.e.,', np.unique(y_train))
# Initialize your model
# model = DILVIT(num_classes=dls.c)
model = InceptionTime(dls.vars, dls.c, fc_dropout=.5, nf=64, return_features=False)
# model = TST(dls.vars, dls.c, dls.len, dropout=.3, fc_dropout=.5)
# model = XCM(dls.vars, dls.c, dls.len, fc_dropout=.5, nf=64)
# model = ConvTranPlus(dls.vars, dls.c, dls.len, fc_dropout=.5)

# Define metrics with macro averaging
precision = Precision(average='macro')
recall = Recall(average='macro')
f1 = F1Score(average='macro')

adamw = wrap_optimizer(torch.optim.AdamW)
# Create Learner with callbacks
learn = Learner(
    dls,
    model,
    loss_func=CrossEntropyLossFlat(), # 
    # loss_func=LabelSmoothingCrossEntropyFlat(),
    metrics=[accuracy, precision, recall, f1],
    
    opt_func=adamw,
    wd=0.01,
    # cbs=cbs
)

#%%
# Save initial model state
# learn.save(f'{config["chkpt_dir"]}{exp_name}_0')

# # Load the initial model state
# learn.load(f'{config["chkpt_dir"]}{exp_name}_0')

# # Find optimal learning rate
learn.lr_find()

# learn.lr_find(suggest_funcs=[minimum, steep, valley, slide])

# adding augmentations before the lr_find creates an error
# https://github.com/fastai/fastai/issues/3239
learn.add_cbs(  
                [ 
                # WandbCallback(log='all',log_preds=False, log_model=False, dataset_name=config['dataset']),
                # GradientClip(1.0), 
                # MixUp1d(0.4),
                ExclusiveMixer(p_apply_any=0.5, p_cutmix_if_apply=0.5), 
                # CutMix1d(),
                # IntraClassCutMix1d(),
                SaveModelCallback(monitor='f1_score',
                                fname=f'{config["chkpt_dir"]}best_{exp_name}'),
                # EarlyStoppingCallback(monitor='f1_score', comp=np.greater, min_delta=0.0, patience=5)
                ]
            )

# Train the model
learn.fit_one_cycle(config['epochs'], lr_max=config['lr_max'])

learn.plot_metrics()
# Save the trained model
learn.save(f'{config["chkpt_dir"]}{exp_name}_last')

# load best mode
# learn.load(f'{config["data_dir"]}best_{exp_name}')
learn.remove_cbs([SaveModelCallback,
                  WandbCallback
                  , EarlyStoppingCallback,
                  ExclusiveMixer])
#%%
learn.load(f'{config["chkpt_dir"]}best_{exp_name}')
learn.validate(dl=dls.valid)
#%%
test_probas, test_targets, test_preds = learn.get_preds(dl=dls.valid, with_decoded=True)
print(test_probas.shape, test_targets.shape, test_preds.shape)

# calculate metrics and AUROC
test_preds = test_preds.numpy()
test_targets = test_targets.numpy()
test_probas = test_probas.numpy()

f1 = skm.f1_score(test_targets, test_preds, average='weighted')
if test_probas.shape[1] > 2:
    auroc = skm.roc_auc_score(test_targets, test_probas, average='weighted')#, multi_class='ovr')
else:
    auroc = skm.roc_auc_score(test_targets, test_probas[:,1], average='weighted')
recall = skm.recall_score(test_targets, test_preds, average='weighted')
precision = skm.precision_score(test_targets, test_preds, average='weighted')
accuracy = skm.accuracy_score(test_targets, test_preds)
print(f'F1: {f1:.4f}, AUROC: {auroc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f}')
#%%
