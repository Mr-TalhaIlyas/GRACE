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
from tools.eval_utils import (TestTimeEvaluator,
                              apply_temporal_smoothing_probs,
                              apply_temporal_smoothing_preds,
                              hysteresis_thresholding,
                              calculate_epoch_level_metrics)
from tools.eval_epochs import calculate_epoch_level_metrics_extended
from tools.viz_utils import (plot_model_comparison,
                             plot_ap_curves, plot_roc_curves,
                             plot_confusion_matrix_grid,
                             draw_temporal_seizure_plots)
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, f1_score,
    recall_score, accuracy_score, precision_score
)
# Import your model and data loading code here
from configs.config import config
from data.dataloader import GEN_DATA_LISTS, SlidingWindowMMELoader
from torch.utils.data import DataLoader
from models.model import MME_Model
from models.utils.tools import load_chkpt

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
if config['num_fold'] < 0:
    data = GEN_DATA_LISTS(config['external_data_dict'])
    test_data = data.get_folds(-1)
else:
    data = GEN_DATA_LISTS(config)
    _, test_data = data.get_folds(config['num_fold'])

val_dataset = SlidingWindowMMELoader(test_data, config, augment=False)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                        shuffle=False, num_workers=config['num_workers'], 
                        drop_last=False, pin_memory=config['pin_memory'])

# Load model
model = MME_Model(config['model']).to(device)
chkpt_path = os.path.join(config['checkpoint_path'], 
                            'alfred_cv_full_fusion_ce_margin_jsd_lossv16v4.pth')
load_chkpt(model, None, chkpt_path)

# Configure evaluation parameters
eval_config = {
    'window_duration': 10.0, # seconds
    'detection_tolerance': 60.0,  # ±60 seconds
    'min_prediction_duration': 5.0,
    'probability_threshold': 0.5,
    'smoothing_window': 3,
    'pre_seizure_fp_threshold': 3,
    'apply_temporal_smoothing': True,  # Enable temporal smoothing
    'calculate_window_metrics': True,  # Calculate window-based metrics
    'calculate_event_metrics': True,   # Calculate event-based metrics
    
    'output_dir': '/home/user01/Data/npj/output/results/'
}


test_evaluator = TestTimeEvaluator(model, device)

all_preds, all_probs, all_targets = test_evaluator.evaluate(val_loader)

# saving results all_preds, all_probs, are dicts and all_targets is a numpy array
# output_dir = config['output_dir']
# os.makedirs(output_dir, exist_ok=True)
# np.save(os.path.join(output_dir, 'all_preds.npy'), all_preds)
# np.save(os.path.join(output_dir, 'all_probs.npy'), all_probs)
# np.save(os.path.join(output_dir, 'all_targets.npy'), all_targets)
#%%
# calculate metrics
# resutls = calculate_epoch_level_metrics(all_preds, all_probs, all_targets, eval_config)
results = calculate_epoch_level_metrics_extended(all_preds, all_probs, all_targets, eval_config)
_ = draw_temporal_seizure_plots(all_preds, all_probs, all_targets, eval_config)
_ = plot_confusion_matrix_grid(all_preds, all_probs, all_targets, normalize='true', config=eval_config)
_ = plot_roc_curves(all_preds, all_probs, all_targets, eval_config)
_ = plot_ap_curves(all_preds, all_probs, all_targets, eval_config)
#%%

from tsaug.visualization import plot
import tools.eval_utils as tev
# reloading
from importlib import reload
reload(tev)
from tools.eval_utils import seprate_synchronize_events
# from tools.eval_utils import seprate_synchronize_events
#%%
events = seprate_synchronize_events(all_targets, all_probs['fusion_outputs'])

i = 0
plot(events[i]['model_output'], events[i]['ground_truth'])

# %%
