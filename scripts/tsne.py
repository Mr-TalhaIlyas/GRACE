#%%
import os, psutil
# os.chdir(os.path.dirname(__file__))
os.chdir('/home/talha/Data/mme/scripts/')

from configs.config import config

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

# Use an environment variable to override the fold number, if provided
fold_number = os.getenv('FOLD_NUMBER', config['num_fold'])
config['num_fold'] = int(fold_number)
config['experiment_name'] = f"Exp3_ecg-sig_fold{config['num_fold']}"
config['model']['ewt_pretrainned_chkpts'] = config['model']['ewt_pretrainned_chkpts'].replace('fold3', f'fold{config["num_fold"]}')

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

import cv2, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
from termcolor import cprint
from tqdm import tqdm

from data.dataloader import GEN_DATA_LISTS, MME_Loader, BiasedSampler
from data.utils import collate, values_fromreport, print_formatted_table

from models.model import MME_Model
from models.utils.lr_scheduler import LR_Scheduler
from models.utils.tools import save_chkpt
from tools.training import Trainer, update_ema

from sklearn.metrics import confusion_matrix, classification_report

from models.utils.visualization import viz_pose, display_video
from tsaug.visualization import plot
from IPython.display import HTML
from tools.inference import run_inference
from plot_utils import plot_tsne
import numpy as np
from sklearn.manifold import TSNE
import umap
from plot_utils import (stabilize_predictions, find_split_indices,
                        split_array_at_indices, calculate_latency_directly,
                        plot_predictions_over_labels, plot_compact_horizontal_bars)
#%
num_classes = len(config['sub_classes'])
sub_classes = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = GEN_DATA_LISTS(config)

train_data, test_data = data.get_folds(1)
data.chk_paths(train_data)


#%%
chkpt = "/home/talha/Data/mme/chkpts/Exp3_ecg-sig_fold1.pth"
chkpt = torch.load(chkpt)

config['model']['batch_size'] = 1
model = MME_Model(config['model'])
model.to(DEVICE)
model.load_state_dict(chkpt['model_state_dict'])

#%%

model.eval()
with torch.no_grad():
    infer_report = run_inference(train_data, model, overlap_sec=3, return_feats=True)

# current_acc = np.nanmean(infer_report['fusion'][0:3])
# Keys for the first dictionary
keys_dict1 = ['fusion', 'ecg', 'pose', 'flow']

# Splitting the dictionary
# dict1 = {k: original_dict[k] for k in keys_dict1}
# dict2 = {k: original_dict[k] for k in original_dict if k not in keys_dict1}

print_formatted_table({k: infer_report[k] for k in ['fusion', 'ecg', 'pose', 'flow']})
#%%


X = infer_report['all_ecg_feats']

# reducer = umap.UMAP(n_neighbors=20)
# X_embedded = reducer.fit_transform(X)

X_embedded = TSNE(n_components=2, random_state=43,
                  perplexity=60, early_exaggeration=12.0,
                  # perplexity=3
                  ).fit_transform(X)
   
plot_tsne(X_embedded, infer_report['all_lbls'], exclude_class_0=False)
#%%
clip_duration=10
overlap=3
modalities = ['fusion', 'ecg', 'pose', 'flow']
modality = 'fusion'
for modality in modalities:
    lbls = infer_report['all_lbls']
    preds = infer_report[f'all_{modality}_preds']
    
    lbls = np.clip(lbls, a_min=0, a_max=1)
    preds = np.clip(preds, a_min=0, a_max=1)
    
    
    indices = find_split_indices(lbls)
    
    split_labels = split_array_at_indices(lbls, indices)
    split_preds = split_array_at_indices(preds, indices)
    
    # from each split label only get value form 49 th index onwards
    # split_labels = [x[49:] for x in split_labels]
    # split_preds = [x[49:] for x in split_preds]
    
    
    stablized_preds = []
    for pred in split_preds:
        stablized_preds.append(stabilize_predictions(pred, window_size=6))
    #%
    # plot_predictions_over_labels(stablized_preds, split_labels)
    # plot_compact_horizontal_bars(stablized_preds, split_labels)
    #%
    
    
    latencies = []
    for pred, lbl in zip(split_preds, split_labels):
        latencies.append(calculate_latency_directly(pred, lbl, consecutive_ones=6,
                                                    clip_duration=10, overlap=3))
    
    latencies = np.asarray(latencies)
    # print(latencies)
    try:
        FPs = np.unique(np.where(np.abs(latencies) > 100, 1, 0), return_counts=True)[1][1]
    except IndexError:
        FPs = 0
    try:    
        missed = np.unique(np.where(np.isnan(latencies), 1, 0), return_counts=True)[1][1]
    except IndexError:
        missed = 0
    # avg_lat = np.nanmean(latencies)
    avg_lat = np.nanmean(np.where(np.abs(latencies) > 100, 0, latencies)) + (clip_duration - overlap)
    print(f'Mode : {modality}')
    print(f'Average Latency: {avg_lat:.2f} seconds', f'False Positives: {FPs}', f'Missed: {missed}', sep='\n')
#%%
modality = 'ecg'
lbls = infer_report['all_lbls']
preds = infer_report[f'all_{modality}_preds']

lbls = np.clip(lbls, a_min=0, a_max=1)
preds = np.clip(preds, a_min=0, a_max=1)


indices = find_split_indices(lbls)

split_labels = split_array_at_indices(lbls, indices)
split_preds = split_array_at_indices(preds, indices)

stablized_preds = []
for pred in split_preds:
    stablized_preds.append(stabilize_predictions(pred, window_size=6))
#%
plot_predictions_over_labels(stablized_preds, split_labels)

#%%

test_feat_dir = "/home/talha/Data/mme/data/test_feats/"
trains_feat_dir = "/home/talha/Data/mme/data/train_feats/"

filenames = infer_report['filenames']

fusion_feats = infer_report['all_fusion_feats']
ecg_feats = infer_report['all_ecg_feats']
pose_feats = infer_report['all_pose_feats']
flow_feats = infer_report['all_flow_feats']

lbls = infer_report['all_lbls']

indices = find_split_indices(lbls)

split_labels = split_array_at_indices(lbls, indices)
split_fusion_feats = split_array_at_indices(fusion_feats, indices)
split_ecg_feats = split_array_at_indices(ecg_feats, indices)
split_pose_feats = split_array_at_indices(pose_feats, indices)
split_flow_feats = split_array_at_indices(flow_feats, indices)


for i in range(len(split_labels)):
    filtered_indices = np.where(split_labels[i] != 0)[0] # remove basline/preictal indices
    
    split_labels[i] = split_labels[i][filtered_indices]

    split_fusion_feats[i] = split_fusion_feats[i][filtered_indices]
    split_ecg_feats[i] = split_ecg_feats[i][filtered_indices]
    split_pose_feats[i] = split_pose_feats[i][filtered_indices]
    split_flow_feats[i] = split_flow_feats[i][filtered_indices]
#%%
# start writing arrays
for i, filename in enumerate(filenames):
    np.savez_compressed(os.path.join(test_feat_dir, filename),
                        fusion=split_fusion_feats[i],
                        ecg=split_ecg_feats[i],
                        pose=split_pose_feats[i],
                        flow=split_flow_feats[i],
                        lbls=split_labels[i])
#%%
# d = np.load("/home/talha/Data/mme/data/test_feats/a_patient_1.npz")

# t = d['fusion']

# t = d['ecg']

# t = d['lbls']































# num_repeats_per_epoch = config['num_repeats_per_epoch']
 
# start_epoch = 0
# epoch, best_acc = 0, 0
# total_avg_acc = []

# for epoch in range(start_epoch, config['epochs']):
#     model.train()  # Set model to training mode
#     tacc, tbacc, tloss = [], [], []

#     for n_repeat in range(num_repeats_per_epoch):
#         for step, data_batch in enumerate(train_loader):
#             scheduler(optimizer, step, epoch)
#             loss_value, sub_acc, sup_acc = trainer.training_step(data_batch)
#             tloss.append(loss_value)
#             tacc.append(sub_acc)
#             tbacc.append(sup_acc)
            
#             # Log metrics at every step of num_repeats_per_epoch
#             if config['LOG_WANDB']:
#                 wandb.log({"Train Acc Step": sub_acc, "Train Bin Acc Step": sup_acc,
#                            "Loss Step": loss_value}, step=epoch * num_repeats_per_epoch + n_repeat * len(train_loader) + step)

#     # Log average metrics after num_repeats_per_epoch
#     avg_loss, avg_acc, avg_bin_acc = np.nanmean(tloss), np.nanmean(tacc), np.nanmean(tbacc)
#     print(f'Epoch: {epoch+1}/{config["epochs"]} => Average loss: {avg_loss:.4f}, Average Acc: {avg_acc:.4f}, Average Bin Acc: {avg_bin_acc:.4f}')
    

#     if (epoch + 1) % 1 == 0:  # eval every N epoch
#         print('Validating...')
#         model.eval()  # Set model to evaluation mode
#         test_acc, test_bacc = [], []
#         for _ in range(num_repeats_per_epoch):
#             for step, test_batch in enumerate(test_loader):
#                 sub_acc, sup_acc = evaluator.eval_step(test_batch)
#                 test_acc.append(sub_acc)
#                 test_bacc.append(sup_acc)
#         print(f'Epoch: {epoch+1} => Average Acc: {np.nanmean(test_acc):.4f}; Average Bin Acc : {np.nanmean(test_bacc):.4f}')
#         total_avg_acc.append(np.nanmean(test_acc))
#         current_acc = np.nanmax(total_avg_acc)
#         #####################
#         all_preds = np.asarray(all_preds).reshape(-1, num_classes)
#         all_lbls = np.asarray(all_lbls).reshape(-1,)
        
#         matrix = confusion_matrix(all_lbls, np.argmax(all_preds, axis=1),
#                                   labels=[0,1,2,3,4], normalize='true')
#         report = classification_report(all_lbls, np.argmax(all_preds, axis=1),
#                                         output_dict=True,
#                                         zero_division=0)
#         p, r, f1 = values_fromreport(report)
#         cprint(f'Class Accuracies:: {matrix.diagonal()/matrix.sum(axis=1)}', 'blue')
#         cprint(f'Class-wise Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}', 'light_magenta')
#         #####################
#         if current_acc > best_acc and epoch != 0:
#             best_acc = current_acc
#             chkpt = save_chkpt(model, optimizer, epoch, loss=np.nanmean(tloss),
#                                acc=current_acc, return_chkpt=True)

#         if config['LOG_WANDB']:
#             wandb.log({"Epoch Loss": avg_loss, 
#                        "Epoch Test Acc": np.nanmean(test_acc), "Epoch Train Acc": avg_acc,
#                        "Epoch Test Bin Acc": np.nanmean(test_bacc), "Epoch Train Bin Acc": avg_bin_acc,
#                        "Learning Rate": optimizer.param_groups[0]['lr']}, step=epoch + 1)

# # Final logging outside the loop
# if config['LOG_WANDB']:
#     # Log precision, recall, F1-score after the loop if necessary
#     # wandb.log({"Precision": p, "Recall": r, "F1": f1}, step=epoch + 1)
#     wandb.run.finish()
# #%%