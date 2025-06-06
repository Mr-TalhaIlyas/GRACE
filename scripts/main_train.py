#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 09:43:52 2025

@author: user01
"""

import os, psutil
# os.chdir(os.path.dirname(__file__))
os.chdir('/home/user01/Data/npj/scripts/')

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED
from rich.text import Text
from rich import print as rprint

console = Console()

console.print(Panel.fit("[bold magenta]EXPERIMENT INITIALIZATION[/]", 
                       style="bold blue", padding=(1, 2)))

from configs.config import config
config_panel = Panel.fit(
    Text.assemble(
        ("\n[bold]PROJECT:[/] ", "bold cyan"), f"{config['project_name']}\n",
        ("[bold]EXPERIMENT:[/] ", "bold cyan"), f"{config['experiment_name']}\n",
        ("[bold]FOLD:[/] ", "bold cyan"), f"{config['num_fold']}\n",
    ),
    title="[bold green]CONFIGURATION OVERVIEW[/]",
    border_style="bright_green",
    padding=(1, 4)
)
console.print(config_panel)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

hardware_table = Table(title="[bold]HARDWARE CONFIGURATION[/]", 
                      box=ROUNDED, header_style="bold magenta")
hardware_table.add_column("Component", style="cyan")
hardware_table.add_column("Details")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
torch.autograd.set_detect_anomaly(False)
hardware_table.add_row("PyTorch Version", torch.__version__)
if torch.cuda.is_available():
    hardware_table.add_row("CUDA Version", torch.version.cuda)
    hardware_table.add_row("GPUs Available", str(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        hardware_table.add_row(f"GPU {i}", torch.cuda.get_device_name(i))
else:
    hardware_table.add_row("CUDA", "Not Available")
hardware_table.add_row("CPU Cores", str(os.cpu_count()))
memory = psutil.virtual_memory()
hardware_table.add_row("Total Memory", f"{memory.total / (1024**3):.2f} GB")

console.print(hardware_table)

if config['LOG_WANDB']:
    import wandb
    # from datetime import datetime
    # my_id = datetime.now().strftime("%Y%m%d%H%M")
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'],
            #    resume='allow', id=my_id, # this one introduces werid behaviour in the app
               config_include_keys=config.keys(), config=config)


# if config.get('verbose', False):
config_table = Table(title="[bold]FULL CONFIGURATION[/]", 
                    box=ROUNDED, header_style="bold yellow")
config_table.add_column("Key", style="cyan")
config_table.add_column("Value", style="green")

for key, value in config.items():
    if key == 'model':
        continue  # Skip the 'model' key
    if isinstance(value, (str, int, float, bool)):
        config_table.add_row(key, str(value))
    elif isinstance(value, dict):
        config_table.add_row(key, "\n".join(f"{k}: {v}" for k, v in value.items()))

console.print(config_table)

config_table = Table(title="[bold]MODEL CONFIGURATION[/]", 
                    box=ROUNDED, header_style="bold yellow")
config_table.add_column("Key", style="cyan")
config_table.add_column("Value", style="green")

for key, value in config['model'].items():
    if isinstance(value, (str, int, float, bool)):
        config_table.add_row(key, str(value))
    elif isinstance(value, dict):
        config_table.add_row(key, "\n".join(f"{k}: {v}" for k, v in value.items()))

console.print(config_table)

from fmutils import fmutils as fmu

import cv2, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
from termcolor import cprint
from tqdm import tqdm

from data.dataloader import GEN_DATA_LISTS, SlidingWindowMMELoader
from data.utils import collate, values_fromreport, print_formatted_table

from models.model import MME_Model
from torch.optim.lr_scheduler import CyclicLR
from models.utils.lr_scheduler import LR_Scheduler
from models.utils.tools import save_chkpt, load_chkpt
from tools.training_v6 import Trainer, Evaluator

from sklearn.metrics import confusion_matrix, classification_report

from models.utils.visualization import viz_pose, display_video
from tsaug.visualization import plot
from IPython.display import HTML
#%
# num_classes = len(config['sub_classes'])
# sub_classes = 1
num_super_classes = len(config['super_classes'])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config['num_fold'] < 0: # -1 means all folds
    data_gen = GEN_DATA_LISTS(config)
    train_data = data_gen.get_folds(-1)
    
    class_weights = data_gen.calculate_class_weights(train_data, num_super_classes)
    rprint(f"[bold cyan]Calculated class weights:[/] {class_weights}")

    data_gen = GEN_DATA_LISTS(config['external_data_dict'])
    test_data = data_gen.get_folds(-1)

else:
    data_gen = GEN_DATA_LISTS(config)
    train_data, test_data = data_gen.get_folds(config['num_fold'])

if config['sanity_check']:
    data_gen.chk_paths(train_data)
    data_gen.chk_paths(test_data)



train_dataset = SlidingWindowMMELoader(train_data, config, augment=True)

train_loader = DataLoader(train_dataset,
                        batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        )

# get test dataset with 0 overlap
# config['window_overlap'] = 0
val_dataset = SlidingWindowMMELoader(test_data, config, augment=False)
val_loader = DataLoader(val_dataset,
                        batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        )

header = " DATASET CONFIGURATION "
print(f"\n{header:-^60}")
print(f"│ {'• Window Size:':<20} {config['sample_duration']:<10} seconds │")
print(f"│ {'• Overlap:':<20} {config['window_overlap']:<10} seconds │")
print(f"│ {'• Train Windows:':<20} {len(train_dataset.mapping):<10} │")
print(f"│ {'• Test Windows:':<20} {len(val_dataset.mapping):<10} │")
print(f"{'':-^60}")

#%%

if config['sanity_check']:
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
    # plt.imshow(batch['ecg'][0][:,0:750], 'jet') # first 750 samples.
    plot(batch['ecg_seg'][0].numpy())
    plot(batch['hrv'][0].numpy())

#%%

model = MME_Model(config['model'])
model.to(DEVICE)

# optimizer = torch.optim.AdamW([{'params': model.parameters(),
#                             'lr':config['learning_rate']}],
#                             weight_decay=config['WEIGHT_DECAY'])

pose_params = []
other_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if 'bodygcn' in name or 'facegcn' in name or 'rhgcn' in name or 'lhgcn' in name:
        pose_params.append(param)
    else:
        other_params.append(param)

base_lr_main = config['learning_rate']
base_lr_pose = config['learning_rate'] * config['pose_lr_multiplier']

optimizer_params = [
    {'params': other_params, 'lr': base_lr_main}, # Set initial LR for this group
    {'params': pose_params, 'lr': base_lr_pose}    # Set initial LR for pose group
]

optimizer = torch.optim.AdamW(optimizer_params,
                            # lr will be taken from optimizer_params for each group
                            weight_decay=config['WEIGHT_DECAY'])

if config['lr_schedule'] == 'cyclic':
    base_lr_main = config['base_lr']
    base_lr_pose = config['base_lr'] * config['pose_lr_multiplier']
    max_lr_main = config['max_lr']
    max_lr_pose = config['max_lr'] * config['pose_lr_multiplier']
    step_size_up_iters = config['clr_step_epochs'] * len(train_loader)

    scheduler = CyclicLR(optimizer,
                         base_lr=[base_lr_main, base_lr_pose], # List of base LRs for each group
                         max_lr=[max_lr_main, max_lr_pose],    # List of max LRs for each group
                         step_size_up=step_size_up_iters,
                         mode=config['clr_mode'],
                         cycle_momentum=False) 
    print(f"Using PyTorch CyclicLR: base_lrs=[{base_lr_main:.1e}, {base_lr_pose:.1e}], max_lrs=[{max_lr_main:.1e}, {max_lr_pose:.1e}], step_up_iters={step_size_up_iters}")
else:
    # Fallback to your custom scheduler for 'cos', 'poly', 'step'
    scheduler = LR_Scheduler(config['lr_schedule'],
                             config['learning_rate'], # Main base LR for your custom scheduler
                             config['epochs'],
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=config['warmup_epochs'])

scaler    = GradScaler()

trainer   = Trainer(model, optimizer, scaler, cfg=config, class_weights=class_weights)
evaluator = Evaluator(model)
#%%
best_val_fusion = 0.0

encoders_frozen_status = {name: False for name in config.get('recall_thresholds', {}).keys()}

for epoch in range(config['epochs']):
    # TRAIN
    train_loss, train_accs, loss_breakdown = trainer.train_epoch(
        train_loader, epoch, scheduler
    )

    # VALIDATE
    val_f1 = {}
    if (epoch + 1) % config['val_every'] == 0:
        val_f1 = evaluator.validate(val_loader, epoch)
        #—————— Feezing encoders based on recall thresholds——————
        # freezing should be done after atleast 10 epochs of training
        if (epoch+1) < 5:
            print(f"Skipping encoder freezing in epoch {epoch+1} as it requires at least 10 epochs of training.")
        else:
            for modality_name, threshold in config.get('recall_thresholds', {}).items():
                if modality_name in val_f1 and not encoders_frozen_status[modality_name]:
                    current_recall = val_f1[modality_name]
                    if current_recall >= threshold:
                        rprint(f"[bold yellow]Validation recall for {modality_name} ({current_recall:.4f}) reached threshold ({threshold:.2f}). Freezing encoder(s).[/bold yellow]")
                        module_prefixes_to_freeze = config.get('encoder_module_names', {}).get(modality_name, [])
                        
                        if not module_prefixes_to_freeze:
                            rprint(f"[bold red]Warning: No module names defined in config['encoder_module_names'] for freezing modality {modality_name}[/bold red]")
                            continue

                        frozen_count = 0
                        for param_name, param in model.named_parameters():
                            for prefix in module_prefixes_to_freeze:
                                if param_name.startswith(prefix):
                                    if param.requires_grad:
                                        param.requires_grad = False
                                        frozen_count += 1
                        if frozen_count > 0:
                            rprint(f"[bold yellow]Froze {frozen_count} parameters for {modality_name} encoder(s): {', '.join(module_prefixes_to_freeze)}[/bold yellow]")
                        
                        encoders_frozen_status[modality_name] = True
                        trainer.update_frozen_status(encoders_frozen_status)

        fusion_f1 = val_f1['fusion']
        if fusion_f1 > best_val_fusion:
            best_val_fusion = fusion_f1
            chkpt = save_chkpt(model, optimizer, epoch, loss=np.nanmean(train_loss),
                                acc=fusion_f1, return_chkpt=True)
            
    # —————— LOG METRICS ——————
    if config['LOG_WANDB']:
        log_dict = {
            "learning_rate": optimizer.param_groups[0]['lr'],
            'train/loss': train_loss,
            **{f"train/{k}": v for k, v in train_accs.items()},
        }
        if loss_breakdown:
            for loss_name, loss_value in loss_breakdown.items():
                log_dict[f'train/loss_{loss_name}'] = loss_value
        if val_f1:
            log_dict.update({
                'val/fusion':    val_f1['fusion'],
                'val/flow':      val_f1['flow'],
                'val/ecg':       val_f1['ecg'],
                'val/pose':      val_f1['pose'],
                'best/fusion':   best_val_fusion,
            })
        wandb.log(log_dict, step=epoch+1)

    # —————— PRINT SUMMARY ——————
    cprint(f"\nEpoch {epoch+1:02d}/{config['epochs']} — "
            # f"Train Loss: {train_loss:.4f} | "
            f"Train Fusion F1-score: {train_accs['fusion']:.4f} | "
            f"Val Fusion F1-score: {val_f1.get('fusion', -1):.4f} | "
            f"Best: {best_val_fusion:.4f}\n", 'red')

if config['LOG_WANDB']:
    wandb.finish()
#%%
