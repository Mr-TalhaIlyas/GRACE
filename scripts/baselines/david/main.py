#%%
import os, psutil
# os.chdir(os.path.dirname(__file__))
os.chdir('/home/user01/Data/mme/scripts_miccai/baselines/david/')

from config import config

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

if config['LOG_WANDB']:
    import wandb
    # from datetime import datetime
    # my_id = datetime.now().strftime("%Y%m%d%H%M")
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'],
            #    resume='allow', id=my_id, # this one introduces werid behaviour in the app
               config_include_keys=config.keys(), config=config)
    # print(f'WANDB config ID : {my_id}')

import pprint
print(f'Printing Configuration File:\n{30*"="}\n')
pprint.pprint(config)


import torch
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available")
print(f"Number of CPUs: {os.cpu_count()}")
memory = psutil.virtual_memory()
print(f"Total Memory: {memory.total / (1024**3):.2f} GB") 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from fmutils import fmutils as fmu
import sklearn.metrics as skm

import cv2, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
from termcolor import cprint
from tqdm import tqdm

from IPython.display import HTML
from utils import display_video
from dataloader import GEN_DATA_LISTS, MME_Loader
from lstm_pm import LSTM_PM
from model import LandMarkBased_SeizureDet
from tools import Trainer, Evaluator
import utils

utils.seed_torch(42)

num_classes = len(config['sub_classes'])
sub_classes = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = GEN_DATA_LISTS(config)

train_data, test_data = data.get_folds(config['num_fold'])
if config['sanity_check']:
    data.chk_paths(train_data)
    data.chk_paths(test_data)

train_dataset = MME_Loader(train_data, config, augment=False)

train_loader = DataLoader(train_dataset,
                          batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        )

test_dataset = MME_Loader(test_data, config, augment=False)
test_loader = DataLoader(test_dataset,
                            batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], drop_last=True,
                            collate_fn=None, pin_memory=config['pin_memory'],
                            prefetch_factor=2, persistent_workers=True,
                            )

if config['sanity_check']:
    # DataLoader Sanity Checks
    batch = next(iter(train_loader))
    print('Visualizing a optical flow...')
    # plt.imshow(batch['frames'][0][0,...])
    HTML(display_video(batch['frames'][0]).to_html5_video())

    print('Done')
    x = batch['sub_lbls'][0]
    y = batch['super_lbls'][0]
    print(x)
    print(y)
#%%

model = LandMarkBased_SeizureDet(num_classes=2)
model = model.to(DEVICE)
optimizer = torch.optim.Adam([{'params': model.parameters(),
                            'lr':config['learning_rate']}],
                            weight_decay=config['WEIGHT_DECAY'])

trainer = Trainer(model, optimizer)
evaluator = Evaluator(model)
# %%
num_repeats_per_epoch = config['num_repeats_per_epoch']
start_epoch = 0
epoch, best_auc = 0, 0
best_f1 = 0
total_avg_acc = []

for epoch in range(start_epoch, config['num_epochs']):
    
    model.train() # <-set mode important
    trloss = []
    tracc = []
    
    for _ in range(num_repeats_per_epoch):
        for step, data_batch in enumerate(train_loader):
            
            loss, acc = trainer.training_step(data_batch)
            # for printing
            trloss.append(loss)
            tracc.append(acc)

    print(f'Epoch: {epoch+1}/{config["num_epochs"]}=> Average loss: {np.nanmean(trloss):.4f}, Average Acc: {np.nanmean(tracc):.4f}')

    # Evaluation
    val_acc, val_f1, val_rec, val_prec = [], [], [], []
    for _ in range(num_repeats_per_epoch): # as we randomly sample clips so we need to repeat the evaluation
        for _, data_batch in enumerate(test_loader):
            probs, preds, labels = evaluator.eval_step(data_batch)
            val_acc.append(skm.accuracy_score(labels, preds))
            val_f1.append(skm.f1_score(labels, preds, average='weighted', zero_division=0.0))
            val_rec.append(skm.recall_score(labels, preds, average='weighted', zero_division=0.0))
            val_prec.append(skm.precision_score(labels, preds, average='weighted', zero_division=0.0))
    
    avg_acc = np.nanmean(val_acc)
    avg_f1 = np.nanmean(val_f1)
    avg_rec = np.nanmean(val_rec)
    avg_prec = np.nanmean(val_prec)
    
    print(f'Validation: Average Acc: {avg_acc:.4f}, Average F1: {avg_f1:.4f}, Average Recall: {avg_rec:.4f}, Average Precision: {avg_prec:.4f}')
    
    curr_f1 = avg_f1
    if curr_f1 > best_f1:
        cprint(f'Saving best model: @ epoch{epoch+1} with F1 {curr_f1}', 'green')
        best_f1 = curr_f1
        torch.save(model.state_dict(), f"{config['checkpoint_path']}landmark_based_seizure_det.pth")
    
# %%
