#%%
import os, psutil
# os.chdir(os.path.dirname(__file__))
os.chdir('/home/user01/Data/npj/scripts/')

from configs.config import config

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

# Use an environment variable to override the fold number, if provided
# fold_number = os.getenv('FOLD_NUMBER', config['num_fold'])
# config['num_fold'] = int(fold_number)
# config['experiment_name'] = f"Exp3_ecg-sig_fold{config['num_fold']}"
# config['model']['ewt_pretrainned_chkpts'] = config['model']['ewt_pretrainned_chkpts'].replace('fold3', f'fold{config["num_fold"]}')

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
from scripts.tools.training_old import Trainer, update_ema

from sklearn.metrics import confusion_matrix, classification_report

from models.utils.visualization import viz_pose, display_video
from tsaug.visualization import plot
from IPython.display import HTML
from tools.inference import run_inference
#%%
num_classes = len(config['sub_classes'])
sub_classes = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = GEN_DATA_LISTS(config)

train_data, test_data = data.get_folds(config['num_fold'])
if config['sanity_check']:
    data.chk_paths(train_data)
    data.chk_paths(test_data)

train_dataset = MME_Loader(train_data, config, augment=True)

train_loader = DataLoader(train_dataset,
                          batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        # sampler=BiasedSampler(train_dataset)
                        )

#%%
if config['sanity_check']:
    # DataLoader Sanity Checks
    batch = next(iter(train_loader))
    print('Visualizing a optical flow...')
    # plt.imshow(batch['frames'][0][0,...])
    HTML(display_video(batch['frames'][0]).to_html5_video())
    print('Done\nVisualizing a pose...')
    HTML(viz_pose(batch['body'][0],
                  batch['face'][0],
                  batch['rh'][0],
                  batch['rh'][0]).to_html5_video())
    print('Done')
    x = batch['sub_lbls'][0]
    y = batch['super_lbls'][0]
    print(x)
    print(y)
    plt.imshow(batch['ecg'][0][:,0:750], 'jet') # first 750 samples.
    plot(batch['ecg_seg'][0].numpy())
# #%%
# pbar = tqdm(train_loader)
# for step, data_batch in enumerate(pbar):
#     x = data_batch['sub_lbls']
#     y = data_batch['super_lbls']
#     print(x)
#     print(y)

#%%
model = MME_Model(config['model'])
model.to(DEVICE)
# create inference model fix config for batch_size==1
config['model']['batch_size'] = 1
infer_model = MME_Model(config['model'])
infer_model.to(DEVICE)

optimizer = torch.optim.AdamW([{'params': model.parameters(),
                            'lr':config['learning_rate']}],
                            weight_decay=config['WEIGHT_DECAY'])

scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

accuracy = Accuracy(task="multiclass", num_classes=num_classes)

trainer = Trainer(model, optimizer)
# evaluator = Evaluator(model)

#%%
# Initializing plots
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({ # training
                "Flow Acc": 0, "Pose Acc": 0,
                "Fusion Acc": 0,"ECG Acc": 0,
                # losses
                "total_loss": 10,
                "fusion_loss": 10, "flow_loss":10,
                "pose_loss":10, "ecg_loss":10,
                # testing
                "Test Flow Acc": 0, "Test Pose Acc": 0,
                "Test Fusion Acc": 0,"Test ECG Acc": 0,
                "learning_rate": 0}, step=0)

#%%
num_repeats_per_epoch = config['num_repeats_per_epoch']
 
start_epoch = 0
epoch, best_acc = 0, 0
total_avg_acc = []

for epoch in range(start_epoch, config['epochs']):
     
    model.train() # <-set mode important
    tloss, fusion_loss, flow_loss, pose_loss, ecg_loss = [], [], [], [], []
    # Repeat the training data num_repeats_per_epoch times before moving to the next epoch
    fusion_acc, pose_acc, flow_acc, ecg_acc = [], [], [], []
    for _ in range(num_repeats_per_epoch):
        for step, data_batch in enumerate(train_loader):
            scheduler(optimizer, step, epoch)
            total_loss, all_loss, all_acc = trainer.training_step(data_batch)
            # for printing
            tloss.append(total_loss)
            fusion_acc.append(all_acc['fusion_outputs'])
            # for plotting
            fusion_loss.append(all_loss['fusion_outputs'])
            flow_loss.append(all_loss['flow_outpus'])
            pose_loss.append(all_loss['joint_pose_outputs'])
            ecg_loss.append(all_loss['ecg_outputs'])
            pose_acc.append(all_acc['joint_pose_outputs'])
            flow_acc.append(all_acc['flow_outpus'])
            ecg_acc.append(all_acc['ecg_outputs'])
    # update EMA model i.e, infer_model if USE_EMA_UPDATES is True    
    if config['USE_EMA_UPDATES']:
        update_ema(model, infer_model, alpha=config['ema_momentum'],
                   epoch=epoch, ema_warmup_epochs=config['warmup_epochs'])

    print(f'Epoch: {epoch+1}/{config["epochs"]}=> Average loss: {np.nanmean(tloss):.4f}, Average Acc: {np.nanmean(fusion_acc):.4f}')
    # fix following to infer only every 2nd epoch 
    if (epoch + 1) % 2 == 0:  # eval every N epoch
        print('Validating...')
        # load model state dict into infer_model if USE_EMA_UPDATES is False 
        # else infer_model is already updated
        if not config['USE_EMA_UPDATES']:
            infer_model.load_state_dict(model.state_dict())
        infer_report = run_inference(test_data, infer_model)

        if config['LOG_WANDB']:
            wandb.log({ "Test Flow Acc": np.nanmean(infer_report['flow'][0:3]),
                        "Test Pose Acc": np.nanmean(infer_report['pose'][0:3]),
                        "Test Fusion Acc": np.nanmean(infer_report['fusion'][0:3]),
                        "Test ECG Acc": np.nanmean(infer_report['ecg'][0:3])}, step=epoch+1)
            
        current_acc = np.nanmean(infer_report['fusion'][0:3])
        if current_acc > best_acc and epoch != 0:
                best_acc = current_acc
                chkpt = save_chkpt(model, optimizer, epoch, loss=np.nanmean(tloss),
                                acc=current_acc, return_chkpt=True)
                print(40*'$')
        print_formatted_table(infer_report)
        
    if config['LOG_WANDB']:
        wandb.log({"total_loss": np.nanmean(tloss), 
                   "fusion_loss": np.nanmean(fusion_loss), "flow_loss":np.nanmean(flow_loss),
                   "pose_loss":np.nanmean(pose_loss), "ecg_loss":np.nanmean(ecg_loss),
                   
                   "Flow Acc": np.nanmean(flow_acc), "Pose Acc": np.nanmean(pose_acc),
                   "Fusion Acc": np.nanmean(fusion_acc),"ECG Acc": np.nanmean(ecg_acc),

                   "learning_rate": optimizer.param_groups[0]['lr']}, step=epoch+1)
            

if config['LOG_WANDB']:
    wandb.run.finish()

#%%

























































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