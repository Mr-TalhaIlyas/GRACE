import os, psutil
# os.chdir(os.path.dirname(__file__))
os.chdir('/home/user01/Data/mme/scripts_miccai/baselines/res1dcnn/')

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

import cv2, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
from termcolor import cprint
from tqdm import tqdm

from dataloader import GEN_DATA_LISTS, MME_Loader, BiasedSampler
# from data.utils import collate, values_fromreport
from res1dcnn import Res1DCNN
from lr_scheduler import LR_Scheduler
from tools import save_chkpt, values_fromreport
from training import Trainer, Evaluator

from sklearn.metrics import confusion_matrix, classification_report

from tsaug.visualization import plot
from IPython.display import HTML

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

test_dataset = MME_Loader(test_data, config, augment=False)
test_loader = DataLoader(test_dataset,
                         batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        # sampler=BiasedSampler(test_dataset)
                        )
#%%
# DataLoader Sanity Checks
batch = next(iter(train_loader))
x = batch['sub_lbls'][0]
y = batch['super_lbls'][0]
print(x)
print(y)
# plt.imshow(batch['ecg'][0][:,0:750], 'jet')
#%%
plot(batch['ecg_seg'][0].numpy())
#%%
model = Res1DCNN(num_classes=len(config['super_classes']))
# model =  EWT(5,3, pretrained_path='/home/talha/Data/mme/ecg_test/ast.pth')
model.to(DEVICE)


optimizer = torch.optim.AdamW([{'params': model.parameters(),
                            'lr':config['learning_rate']}],
                            weight_decay=config['WEIGHT_DECAY'])

scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

accuracy = Accuracy(task="multiclass", num_classes=num_classes)

trainer = Trainer(model, optimizer)
evaluator = Evaluator(model)
#%%
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({"ECG Acc": 0, "Test ECG Acc": 0,
               "ecg_loss": 10, "learning_rate": 0}, step=0)
#%%

start_epoch = 0
epoch, best_acc = 0, 0
total_avg_acc = []

for epoch in range(start_epoch, config['epochs']):
    epoch 
    pbar = tqdm(train_loader)
    model.train() # <-set mode important
    tacc, tloss = [], []
    for _ in range(30):
        for step, data_batch in enumerate(pbar):

            scheduler(optimizer, step, epoch)
            loss_value, acc = trainer.training_step(data_batch)
            tloss.append(loss_value)
            tacc.append(acc)
            
            pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - t_loss {loss_value:.4f} - Train Acc {acc:.4f}')
        # break
    print(f'=> Average loss: {np.nanmean(tloss):.4f}, Average Acc: {np.nanmean(tacc):.4f}')

    all_preds, all_lbls = [], []
    if (epoch + 1) % 1 == 0: # eval every 2 epoch
        model.eval() # <-set mode important
        test_acc, preds, lbls = [], [], []
        vbar = tqdm(test_loader)
        for _ in range(40):
            for step, test_batch in enumerate(vbar):
                acc, preds, lbl_batch = evaluator.eval_step(test_batch)
                test_acc.append(acc)
                vbar.set_description(f'Validation - Acc {acc:.4f}')
                all_preds.append(preds)
                all_lbls.append(lbl_batch)
            # break

        print(f'=> Average Acc: {np.nanmean(test_acc):.4f}')
        total_avg_acc.append(np.nanmean(test_acc))
        current_acc = np.nanmax(total_avg_acc)

        #####################
        all_preds = np.asarray(all_preds).reshape(-1, 3)
        all_lbls = np.asarray(all_lbls).reshape(-1,)
        
        matrix = confusion_matrix(all_lbls, np.argmax(all_preds, axis=1), normalize='true')
        report = classification_report(all_lbls, np.argmax(all_preds, axis=1),
                                       output_dict=True,
                                       zero_division=0)
        p, r, f1 = values_fromreport(report)
        cprint(f'Class Accuracies:: {matrix.diagonal()/matrix.sum(axis=1)}', 'blue')
        cprint(f'Class-wise Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}', 'light_magenta')
        #####################
        if current_acc > best_acc and epoch != 0:
            best_acc = current_acc
            chkpt = save_chkpt(model, optimizer, epoch, loss=np.nanmean(tloss),
                               acc=current_acc, return_chkpt=True)
    
    if config['LOG_WANDB']:
        wandb.log({"ecg_loss": np.nanmean(tloss), 
                   "Test ECG Acc": np.nanmean(test_acc),"ECG Acc": np.nanmean(tacc),
                   "learning_rate": optimizer.param_groups[0]['lr']}, step=epoch+1)

if config['LOG_WANDB']:
    # wandb.log({"Precision": p, "Recall": r, "F1": f1})
    wandb.run.finish()
    
#%%