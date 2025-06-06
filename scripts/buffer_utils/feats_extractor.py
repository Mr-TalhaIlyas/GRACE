#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('/home/user01/Data/npj/scripts/')

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from data.utils import video_transform
from collections import defaultdict
# Import your model and data loading code here
from configs.config import config
from data.dataloader import GEN_DATA_LISTS, SlidingWindowMMELoader
from torch.utils.data import DataLoader
from models.model import MME_Model
from models.utils.tools import load_chkpt

from buffer_utils.extractor_utils import FeatureExtractor
# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
if config['num_fold'] < 0:
    data_gen = GEN_DATA_LISTS(config)
    train_data = data_gen.get_folds(-1)
    
    data_gen = GEN_DATA_LISTS(config['external_data_dict'])
    test_data = data_gen.get_folds(-1)
else:
    data = GEN_DATA_LISTS(config)
    _, test_data = data.get_folds(config['num_fold'])

train_dataset = SlidingWindowMMELoader(train_data, config, augment=False)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                        shuffle=False, num_workers=config['num_workers'], 
                        drop_last=False, pin_memory=config['pin_memory'])

val_dataset = SlidingWindowMMELoader(test_data, config, augment=False)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                        shuffle=False, num_workers=config['num_workers'], 
                        drop_last=False, pin_memory=config['pin_memory'])

# Load model
model = MME_Model(config['model']).to(device)
chkpt_path = os.path.join(config['checkpoint_path'], 
                            'alfred_cv_full_fusion_ce_margin_jsd_lossv16v4.pth')
load_chkpt(model, None, chkpt_path)
#%%

feat_extractor = FeatureExtractor(model, device=device)

x = feat_extractor.extract(
    loader=train_loader, 
    data_split='train', 
    output_base_dir='/home/user01/Data/npj/datasets/alfred/features/'
)

x = feat_extractor.extract(
    loader=val_loader, 
    data_split='val', 
    output_base_dir='/home/user01/Data/npj/datasets/alfred/features/'
)
#%%
