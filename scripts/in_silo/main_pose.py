#%%
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

config['LOG_WANDB'] = False
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

from data.sahzu_loader import GEN_DATA_LISTS, SlidingWindowVisualLoader
from data.utils import collate, values_fromreport, print_formatted_table

from in_silo.models.pose_model import POSE_Model
from torch.optim.lr_scheduler import CyclicLR
from models.utils.lr_scheduler import LR_Scheduler
from models.utils.tools import save_chkpt, load_chkpt, save_ema_chkpts
from in_silo.utils.pose_trainer import Trainer, Evaluator

from sklearn.metrics import confusion_matrix, classification_report

from models.utils.visualization import viz_pose, display_video
from tsaug.visualization import plot
from IPython.display import HTML
#%
# num_classes = len(config['sub_classes'])
# sub_classes = 1
num_super_classes = len(config['super_classes'])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHKPT_PATH = '/home/user01/Data/npj/scripts/in_silo/chkpt'
EXP_NAME = 'sahzu_pose_in_silo_exp'  # config['experiment_name'] or 'Pose_Exp

data_gen = GEN_DATA_LISTS(config)
sahzu_data = data_gen.get_splits()



train_dataset = SlidingWindowVisualLoader(sahzu_data, config, augment=True, split='train')
train_loader = DataLoader(train_dataset,
                        batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        )

val_dataset = SlidingWindowVisualLoader(sahzu_data, config, augment=False, split='val')
val_loader = DataLoader(val_dataset,
                        batch_size=config['batch_size'], shuffle=True,
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
#%%

model = POSE_Model(config['model'])
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=config['learning_rate'],
                              weight_decay=config['WEIGHT_DECAY'])

scheduler = LR_Scheduler(config['lr_schedule'],
                        config['learning_rate'], # Main base LR for your custom scheduler
                        config['epochs'],
                        iters_per_epoch=len(train_loader),
                        warmup_epochs=config['warmup_epochs'])

scaler    = GradScaler()

trainer   = Trainer(model, optimizer, scaler, cfg=config)
evaluator = Evaluator(model)
#%%
best_val_fusion = 0.0

for epoch in range(config['epochs']):
    # ——————TRAIN——————
    train_loss, train_accs = trainer.train_epoch(
        train_loader, epoch, scheduler
    )

    # ——————VALIDATE——————
    val_f1 = {}
    if (epoch + 1) % config['val_every'] == 0:
        if config['USE_EMA_UPDATES']:
            with trainer.ema_model.average_parameters(): # Temporarily loads EMA parameters into model
                val_f1 = evaluator.validate(val_loader, epoch)
        else:
            val_f1 = evaluator.validate(val_loader, epoch)
        
        fusion_f1 = val_f1['joint_pose_outputs']
        if fusion_f1 > best_val_fusion:
            best_val_fusion = fusion_f1
            torch.save({
                'epoch': epoch,
                'loss': np.nanmean(train_loss),
                'acc': fusion_f1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(CHKPT_PATH, f'{EXP_NAME}.pth'))
            print(f"-> Saving best checkpoint at epoch {epoch+1} with F1-score {fusion_f1:.4f}")
    # —————— LOG METRICS ——————
    if config['LOG_WANDB']:
        log_dict = {
            "learning_rate": optimizer.param_groups[0]['lr'],
            'train/loss': train_loss,
            **{f"train/{k}": v for k, v in train_accs.items()},
        }

        if val_f1:
            log_dict.update({
                'val/fusion':    val_f1['joint_pose_outputs'],
                'val/face_outputs':  val_f1['face_outputs'],
                'val/body_outputs':  val_f1['body_outputs'],
                'best/fusion':   best_val_fusion,
            })
        wandb.log(log_dict, step=epoch+1)

    # —————— PRINT SUMMARY ——————
    cprint(f"\nEpoch {epoch+1:02d}/{config['epochs']} — "
            # f"Train Loss: {train_loss:.4f} | "
            f"Train Fusion F1-score: {train_accs['joint_pose_outputs']:.4f} | "
            f"Val Fusion F1-score: {val_f1.get('joint_pose_outputs', -1):.4f} | "
            f"Best: {best_val_fusion:.4f}\n", 'red')

if config['LOG_WANDB']:
    wandb.finish()
#%%