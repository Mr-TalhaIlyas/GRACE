#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('/home/user01/Data/npj/scripts/')

import torch, pickle
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from data.utils import video_transform
from collections import defaultdict
from evals.tools.eval_utils import (TestTimeEvaluator,
                                        TestTimeModalityEvaluator,
                                        SingleStreamTestTimeEvaluator,
                                        apply_temporal_smoothing_probs,
                                        apply_temporal_smoothing_preds,
                                        hysteresis_thresholding,
                                        calculate_epoch_level_metrics,
                                        seprate_synchronize_events)
from evals.tools.eval_epochs import calculate_epoch_level_metrics_extended
from evals.tools.viz_utils import (plot_model_comparison,
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
    'pickel_file': 'alfred_results.pkl',
    'output_dir': '/home/user01/Data/npj/output/alfred/'
}
os.makedirs(eval_config['output_dir'], exist_ok=True)

test_evaluator = TestTimeEvaluator(model, device)

all_preds, all_probs, all_targets = test_evaluator.evaluate(val_loader)

# %%
from tsai.models.InceptionTime import InceptionTime

ecg_model = InceptionTime(c_in=19, c_out=2,
                          fc_dropout=0.5,
                          nf=64,
                          return_features=False)

chkpt = '/home/user01/Data/npj/scripts/ts/chkpt/best_ecgH_bvg_InceptionTime.pth'
ecg_model.load_state_dict(torch.load(chkpt, map_location='cpu'))


hrveval = TestTimeModalityEvaluator(ecg_model, device,
                                    output_keys=["ecg_outputs"],
                                    modality_name='hrv')

hrv_preds, hrv_probs, hrv_targets = hrveval.evaluate(val_loader)

#%%
# Data to save
data_to_save = {
    'fusion_preds': all_preds,
    'fusion_probs': all_probs,
    'targets': all_targets,
    'hrv_preds': hrv_preds,
    'hrv_probs': hrv_probs
}

# File path for the pickle file
pickle_file_path = os.path.join(eval_config['output_dir'], eval_config['pickel_file'])

# Save the data
with open(pickle_file_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"Results saved to {pickle_file_path}")
#%%
# loading the saved results
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)
all_preds = data['fusion_preds']
all_probs = data['fusion_probs']
all_targets = data['targets']
hrv_preds = data['hrv_preds']
hrv_probs = data['hrv_probs']

#%%
# calculate metrics
# resutls = calculate_epoch_level_metrics(all_preds, all_probs, all_targets, eval_config)
results = calculate_epoch_level_metrics_extended(all_preds, all_probs, all_targets, eval_config)
_ = plot_confusion_matrix_grid(all_preds, all_probs, all_targets, normalize='true', config=eval_config)
_ = plot_roc_curves(all_preds, all_probs, all_targets, eval_config)
_ = plot_ap_curves(all_preds, all_probs, all_targets, eval_config)

#%%
r = all_probs['fusion_outputs']
fusion = apply_temporal_smoothing_probs(r, 5)

p = all_probs['ecg_outputs']
ecg = apply_temporal_smoothing_probs(p, 3)

p = all_probs['flow_outputs']
flow = apply_temporal_smoothing_probs(p, 3)

p = all_probs['joint_pose_outputs']
pose = apply_temporal_smoothing_probs(p, 3)

hrv = hrv_probs['ecg_outputs']
hrv = apply_temporal_smoothing_probs(hrv, 3)

_ = draw_temporal_seizure_plots(all_targets,
                                probs_list=[fusion, ecg, hrv, flow, pose],
                                labels_list=['Fusion', 'ECG-H',
                                             'HRV\n(in-silo)',
                                             'Flow', 'Pose'],
                                eval_config=eval_config)


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

