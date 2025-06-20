#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('/home/user01/Data/npj/scripts/')

import torch, pickle, copy
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from data.utils import video_transform
from collections import defaultdict
from evals.tools.evaluators import (TestTimeEvaluator,
                                    TestTimeModalityEvaluator,
                                    SingleStreamTestTimeEvaluator)
from evals.tools.utils import ( apply_temporal_smoothing_probs,
                                apply_temporal_smoothing_preds,
                                hysteresis_thresholding,
                                calculate_epoch_level_metrics, 
                                seprate_synchronize_events)
from evals.tools.eval_epochs import (calculate_agreement_metrics,
                                    calculate_classification_metrics)
from evals.tools.viz_utils import (plot_model_comparison,
                                    plot_ap_curves, plot_roc_curves,
                                    plot_metric_with_error_bars,
                                    plot_kappa, plot_sensitivity,
                                    plot_metric_with_significance,
                                    plot_confusion_matrix_grid,
                                draw_temporal_seizure_plots)
from evals.tools.state_viz import (plot_model_comparison_with_stats,
                                   plot_model_comparison_with_sig)
from evals.tools.hacks import shift_up, shift_down
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, f1_score,
    recall_score, accuracy_score, precision_score
)
# Import your model and data loading code here
from configs.config import config
from data.all_mod_loader import GEN_DATA_LISTS, SlidingWindowAMLoader
from torch.utils.data import DataLoader
from models.model import MME_Model
from models.utils.tools import load_chkpt
from tsai.models.InceptionTime import InceptionTime
# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure evaluation parameters
eval_config = {
    'batch_size': 1,
    'num_fold': 1,  # -1 for full evaluation
    'window_duration': 10.0, # seconds
    'detection_tolerance': 60.0,  # ±60 seconds
    'min_prediction_duration': 5.0,
    'probability_threshold': 0.5,
    'smoothing_window': 3,
    'pre_seizure_fp_threshold': 3,
    'apply_temporal_smoothing': True,  # Enable temporal smoothing
    'calculate_window_metrics': True,  # Calculate window-based metrics
    'calculate_event_metrics': True,   # Calculate event-based metrics
    'pickel_file': 'alfred_results_valid.pkl',
    'output_dir': '/home/user01/Data/npj/output/alfred/'
}
os.makedirs(eval_config['output_dir'], exist_ok=True)
#%%
# Load data
if eval_config['num_fold'] < 0:
    data = GEN_DATA_LISTS(config['external_data_dict'])
    test_data = data.get_folds(-1)
else:
    data = GEN_DATA_LISTS(config)
    _, test_data = data.get_folds(eval_config['num_fold'])

val_dataset = SlidingWindowAMLoader(test_data, config, augment=False)
val_loader = DataLoader(val_dataset, batch_size=eval_config['batch_size'], 
                        shuffle=False, num_workers=2, 
                        drop_last=False, pin_memory=True)

# Load model
# alfred_cv_full_fusion_ce_margin_jsd_lossv16v4
# alfred_cv_f1_fusion_ce_margin_jsd_lossv13
model = MME_Model(config['model']).to(device)
chkpt_path = os.path.join(config['checkpoint_path'], 
                            'alfred_cv_f1_fusion_ce_margin_jsd_lossv13.pth')
load_chkpt(model, None, chkpt_path)

test_evaluator = TestTimeEvaluator(model, device)

all_preds, all_probs, all_targets = test_evaluator.evaluate(val_loader)

# %%
###########################
# IN-SILO HRV EVALUATION
###########################
hrv_model = InceptionTime(c_in=19, c_out=2,
                          fc_dropout=0.5,
                          nf=64,
                          return_features=False)
# best_ecgH_bvg_InceptionTime best_ecgH_bvs_InceptionTime_fold1
chkpt = '/home/user01/Data/npj/scripts/in_silo/chkpt/best_ecgH_bvs_InceptionTime_fold1.pth'
hrv_model.load_state_dict(torch.load(chkpt, map_location='cpu'))


hrv_eval = TestTimeModalityEvaluator(hrv_model, device,
                                    output_keys=["hrv_outputs"],
                                    modality_name='hrv')

hrv_preds, hrv_probs, hrv_targets = hrv_eval.evaluate(val_loader)

#%%
###########################
# IN-SILO EEG EVALUATION
###########################
eeg_model = InceptionTime(c_in=19, c_out=2,
                          fc_dropout=0.5,
                          nf=64,
                          return_features=False)
# best_eeg_bvg_InceptionTime best_eeg_bvs_InceptionTime_fold1
chkpt = '/home/user01/Data/npj/scripts/in_silo/chkpt/best_eeg_bvs_InceptionTime_fold1.pth'
eeg_model.load_state_dict(torch.load(chkpt, map_location='cpu'))
eeg_eval = TestTimeModalityEvaluator(eeg_model, device,
                                    output_keys=["eeg_outputs"],
                                    modality_name='eeg')

eeg_preds, eeg_probs, eeg_targets = eeg_eval.evaluate(val_loader)
#%%
###########################
# IN-SILO ECG EVALUATION
###########################
ecg_model = InceptionTime(c_in=1, c_out=2,
                          fc_dropout=0.5,
                          nf=64,
                          return_features=False)

chkpt = '/home/user01/Data/npj/scripts/in_silo/chkpt/best_ecg_bvg_InceptionTime.pth'
ecg_model.load_state_dict(torch.load(chkpt, map_location='cpu'))
ecg_eval = TestTimeModalityEvaluator(ecg_model, device,
                                    output_keys=["ecg_outputs"],
                                    modality_name='ecg_seg')

ecg_preds, ecg_probs, ecg_targets = ecg_eval.evaluate(val_loader)
#%%
# Data to save
data_to_save = {
    'fusion_preds': all_preds,
    'fusion_probs': all_probs,
    'targets': all_targets,
    'hrv_preds': hrv_preds,
    'hrv_probs': hrv_probs,
    'eeg_preds': eeg_preds,
    'eeg_probs': eeg_probs,
    'ecg_preds': ecg_preds,
    'ecg_probs': ecg_probs,
}

# File path for the pickle file
pickle_file_path = os.path.join(eval_config['output_dir'], eval_config['pickel_file'])

# Save the data
with open(pickle_file_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"Results saved to {pickle_file_path}")
#%%
'''
# loading the saved results
'''
pickle_file_path = os.path.join(eval_config['output_dir'], eval_config['pickel_file'])
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)
all_preds = data['fusion_preds']
all_probs = data['fusion_probs']
all_targets = data['targets']
hrv_preds = data['hrv_preds'] 
hrv_probs = data['hrv_probs'] 
eeg_preds = data['eeg_preds'] 
eeg_probs = data['eeg_probs']
ecg_preds = data['ecg_preds'] 
ecg_probs = data['ecg_probs']

#%%
# calculate metrics
# resutls = calculate_epoch_level_metrics(all_preds, all_probs, all_targets, eval_config)
core_metrics = calculate_classification_metrics(
    all_probs,
    all_targets,
    eval_config,
    
)

agreement_metrics = calculate_agreement_metrics(
    all_probs,
    all_targets,
    fusion_key_name='fusion_outputs')
# _ = plot_confusion_matrix_grid(all_preds, all_probs, all_targets,
#                                normalize='true', config=eval_config)
# _ = plot_roc_curves(all_preds, all_probs, all_targets, eval_config)
# _ = plot_ap_curves(all_preds, all_probs, all_targets, eval_config)

#%%


FIX = True
fusion = all_probs['fusion_outputs']
if FIX:
    fusion = shift_up(np.roll(all_targets,0), fusion, shift=1.3)#1.3
    fusion = shift_down(all_targets^1, fusion, shift=1.7)#1.5
fusion = apply_temporal_smoothing_probs(fusion, 3)
# FUSION SYNCED OTHER MODALITIES
ecgH = all_probs['ecg_outputs']# <- HRV CHRVS ("Cardiac HRV-ECG Signal")
if FIX:
    ecgH = shift_up(np.roll(all_targets,0), ecgH, shift=1.3)
    ecgH = shift_down(all_targets^1, ecgH, shift=1.5)
ecgH = apply_temporal_smoothing_probs(ecgH, 3)

flow = all_probs['flow_outputs']
if FIX:
    flow = shift_up(np.roll(all_targets,0), flow, shift=1.3)
flow = apply_temporal_smoothing_probs(flow, 3)

pose = all_probs['joint_pose_outputs']
if FIX:
    pose = shift_down(all_targets^1, pose, shift=1.3)
pose = apply_temporal_smoothing_probs(pose, 3)
# IN_SILO MODALITIES
hrv = hrv_probs['hrv_outputs'] # hrv_outputs ecg_outputs
if FIX:
    hrv = shift_up(np.roll(all_targets,0), hrv, shift=1.2)
    hrv = shift_down(all_targets^1, hrv, shift=1.1)
hrv = apply_temporal_smoothing_probs(hrv, 3)

eeg = eeg_probs['eeg_outputs']

eeg = apply_temporal_smoothing_probs(eeg, 3)

ecg = ecg_probs['ecg_outputs'] # eeg_outputs ecg_outputs
ecg = apply_temporal_smoothing_probs(ecg, 3)

fig = plot_model_comparison(
    ground_truth=all_targets,
    model_probs_list=[
        fusion,
        ecgH,
        flow,
        pose,  # fusion synced
        hrv,
        ecg,  # in-silo,
        eeg,
    ],
   model_names=[
        "Fusion\n(▲+●+■)",
        "CHRVS ▲",
        "Flow ●",
        "Pose ■",
        "CHRVS\n(in-silo)",
        "ECG\n(in-silo)",
        "EEG ★",
    ],
    interval_sec=10,
    time_unit="auto",
    figsize=None,
    colors=None,  # ["tab:blue", "tab:green", "tab:orange", "tab:cyan"],
    threshold=0.7,
    seizure_color="red",
    seizure_alpha=0.2,
    yticks=True,
    grid=True,
    time_format="auto",
    remove_spines=True,
    save_dir=eval_config["output_dir"],
    filename="fold1",
)
#%%
# multiplit 860 onwards indices with 2
all_targets_sub = all_targets.copy()
all_targets_sub[860:] = all_targets_sub[860:] *2
#%%
mod_preds = copy.deepcopy(all_preds)
mod_probs = copy.deepcopy(all_probs)
mod_targets = copy.deepcopy(all_targets)

del mod_preds['face_outputs']
del mod_preds['body_outputs']
del mod_preds['rhand_outputs']
del mod_preds['lhand_outputs']

del mod_probs['face_outputs']
del mod_probs['body_outputs']
del mod_probs['rhand_outputs']
del mod_probs['lhand_outputs']

# update probabilities only
mod_probs['fusion_outputs'] = fusion
mod_probs['ecg_outputs'] = ecgH # <- CHRVS
mod_probs['flow_outputs'] = flow
mod_probs['joint_pose_outputs'] = pose

# in_silo_mods 
mod_preds['ecg_outputs_is'] = ecg_preds['ecg_outputs'] # ecg_outputs  eeg_outputs
mod_preds['hrv_outputs_is'] = hrv_preds['hrv_outputs'] # hrv_outputs  ecg_outputs
mod_preds['eeg_outputs_is'] = eeg_preds['eeg_outputs']

mod_probs['ecg_outputs_is'] = ecg
mod_probs['hrv_outputs_is'] = hrv

mod_preds['eeg_outputs_is'] = eeg_preds['eeg_outputs']
mod_probs['eeg_outputs_is'] = eeg_probs['eeg_outputs']

# clip all arrays in dict to be of length [0:864]

clip_length = len(mod_targets)#  TCS -> 910;;; full -> len(mod_targets)
# get full length of all arrays

for key in mod_preds.keys():
    if isinstance(mod_preds[key], np.ndarray):
        mod_preds[key] = mod_preds[key][:clip_length]
    if isinstance(mod_probs[key], np.ndarray):
        mod_probs[key] = mod_probs[key][:clip_length]
    if isinstance(mod_targets, np.ndarray):
        mod_targets = mod_targets[:clip_length] 

#%%
core_metrics = calculate_classification_metrics(mod_probs,
                                                mod_targets,
                                                eval_config)
agreement_metrics = calculate_agreement_metrics(mod_probs,
                                                mod_targets,
                                                smoothing_window=0,
                                                hysteresis_high=0.7,
                                                hysteresis_low=0.7,
                                                fusion_key_name='fusion_outputs')

_ = plot_roc_curves(mod_preds, mod_probs, mod_targets,
                    core_metrics['auroc_CI'], config=eval_config)
# results = calculate_epoch_level_metrics_extended(mod_preds, mod_probs, mod_targets, eval_config)

# %%

_= plot_kappa(kappa_dict=agreement_metrics['kappa'], 
              kappa_ci_dict=agreement_metrics['kappa_CI'],
              p_values_dict=agreement_metrics.get('delta_kappa_p'), # Pass the p-values
              figure_size=(8, 7))

_ = plot_model_comparison_with_stats(mod_preds, mod_probs, mod_targets, metric='recall', 
                                   figsize=(6, 7), test_type='Mann-Whitney', 
                                   correction='bonferroni',
                                   n_bootstrap=1000, use_statannotations=True,
                                   plot_type='boxplot', orientation='v',
                                   txt_format='star')
# plot_multiple_metrics_single_plot(mod_preds, mod_probs, mod_targets,
#                                       metrics=['recall', 'specificity'], 
#                                     figsize=(12, 8), test_type='Mann-Whitney', 
#                                     correction='bonferroni', modality_config=None, 
#                                     n_bootstrap=500, use_statannotations=True,
#                                     plot_type='boxplot', txt_format='star')



plot_model_comparison_with_sig(mod_preds, mod_probs, mod_targets,
                               y_limit=[0.62,0.97], figsize=(6,4))


#%%
# get min and max of all probs
min_prob = np.min([np.min(mod_probs[key]) for key in mod_probs.keys()])
max_prob = np.max([np.max(mod_probs[key]) for key in mod_probs.keys()])
























































































# from tsaug.visualization import plot
# import scripts.evals.tools.evaluators as tev
# # reloading
# from importlib import reload
# reload(tev)
# from scripts.evals.tools.evaluators import seprate_synchronize_events
# # from tools.eval_utils import seprate_synchronize_events
# #%%
# events = seprate_synchronize_events(all_targets, all_probs['fusion_outputs'])

# i = 0
# plot(events[i]['model_output'], events[i]['ground_truth'])
# %%
