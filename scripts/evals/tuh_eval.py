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
                                        seprate_synchronize_events,
                                        split_by_patient_id)
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
from data.tuh_loader import GEN_DATA_LISTS as TUH_GEN_DATA_LISTS
from evals.tools.load_eval_models import load_bio_signal_models
from data.tuh_loader import SlidingWindowBioSignalLoader
# Import your model and data loading code here
from configs.config import config
from data.dataloader import GEN_DATA_LISTS, SlidingWindowMMELoader
from torch.utils.data import DataLoader
from models.model import MME_Model
from models.utils.tools import load_chkpt

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load FUSION model
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
    'pickel_file': 'tuh_results.pkl',
    'output_dir': '/home/user01/Data/npj/output/tuh/'
}
os.makedirs(eval_config['output_dir'], exist_ok=True)

#%%

data_gen = TUH_GEN_DATA_LISTS(config)
tuh_train_data, tuh_test_data = data_gen.get_splits('tuh')

# %%
alfred_models = load_bio_signal_models(config, scope='alfred', device=device)
external_models = load_bio_signal_models(config, scope='external', secondary_dataset='tuh', device=device)

tuh_dataset = SlidingWindowBioSignalLoader(tuh_test_data, config, dataset='tuh', augment=False)

tuh_loader = DataLoader(tuh_dataset,
                        batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], drop_last=False,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        )
print('Getting Patient IDs from TUH dataset...')
tuh_patients = []
for batch in tqdm(tuh_loader, desc="Collecting filenames"):
    tuh_patients.append(batch['filename'][0].split('_')[0])
    
EXTERNAL_MODELS_EVAL_CONFIG = {
    'eeg': {
        'evaluator_output_key': 'tuh_eeg_outputs',
        'dataloader_data_key': 'eeg'  # Corresponds to batch['eeg']
    },
    'ecgH': {
        'evaluator_output_key': 'tuh_ecgH_outputs', # For ECG + HRV features model
        'dataloader_data_key': 'hrv'  # Corresponds to batch['hrv'] (19 channels)
    },
    'ecg': {
        'evaluator_output_key': 'tuh_ecg_outputs', # For raw ECG model
        'dataloader_data_key': 'ecg'  # Corresponds to batch['ecg'] (1 channel)
    }
    
}

ALFRED_MODELS_EVAL_CONFIG = {
    'eeg': {
        'evaluator_output_key': 'alfred_to_tuh_eeg_outputs',
        'dataloader_data_key': 'eeg'  # Corresponds to batch['eeg']
    },
    'ecgH': {
        'evaluator_output_key': 'alfred_to_tuh_ecgH_outputs', # For ECG + HRV features model
        'dataloader_data_key': 'hrv'  # Corresponds to batch['hrv'] (19 channels)
    },
    'ecg': {
        'evaluator_output_key': 'alfred_to_tuh_ecg_outputs', # For raw ECG model
        'dataloader_data_key': 'ecg'  # Corresponds to batch['ecg'] (1 channel)
    }
    
}
# Initialize containers for all collected results
all_tuh_preds_collected = {}
all_tuh_probs_collected = {}
tuh_targets_collected = None

print("\nStarting dynamic evaluation for tuh external models...")
# Loop through the models loaded for the 'external' scope (e.g., from tuh checkpoints)
for model_key, model_instance in external_models.items():
    if model_key in EXTERNAL_MODELS_EVAL_CONFIG:
        eval_params = EXTERNAL_MODELS_EVAL_CONFIG[model_key]
        evaluator_output_key = eval_params['evaluator_output_key']
        dataloader_data_key = eval_params['dataloader_data_key']

        print(f"  Evaluating model: '{model_key}' (Output key: '{evaluator_output_key}', Data key: '{dataloader_data_key}')")

        evaluator = TestTimeModalityEvaluator(
            model_instance,
            device,
            output_keys=[evaluator_output_key],  # TestTimeModalityEvaluator expects a list
            modality_name=dataloader_data_key
        )
        
        # preds_dict will be {evaluator_output_key: np.array}
        # probs_dict will be {evaluator_output_key: np.array}
        preds_dict, probs_dict, targets_array = evaluator.evaluate(tuh_loader)
        
        all_tuh_preds_collected.update(preds_dict)
        all_tuh_probs_collected.update(probs_dict)
        
        if tuh_targets_collected is None:
            tuh_targets_collected = targets_array
        else:
            # Ensure targets are consistent if evaluating multiple models on the same loader
            assert np.array_equal(tuh_targets_collected, targets_array), \
                f"Targets mismatch for model '{model_key}'. This should not happen with the same dataloader."
    else:
        print(f"  Skipping model: '{model_key}'. No evaluation configuration found in EXTERNAL_MODELS_EVAL_CONFIG.")
#%%
all_alfred_to_tuh_preds_collected = {}
all_alfred_to_tuh_probs_collected = {}
alfred_to_tuh_targets_collected = None

for model_key, model_instance in alfred_models.items():
    if model_key in ALFRED_MODELS_EVAL_CONFIG:
        eval_params = ALFRED_MODELS_EVAL_CONFIG[model_key]
        evaluator_output_key = eval_params['evaluator_output_key']
        dataloader_data_key = eval_params['dataloader_data_key']

        print(f"  Evaluating model: '{model_key}' (Output key: '{evaluator_output_key}', Data key: '{dataloader_data_key}')")

        evaluator = TestTimeModalityEvaluator(
            model_instance,
            device,
            output_keys=[evaluator_output_key],  # TestTimeModalityEvaluator expects a list
            modality_name=dataloader_data_key
        )
        
        # preds_dict will be {evaluator_output_key: np.array}
        # probs_dict will be {evaluator_output_key: np.array}
        preds_dict, probs_dict, targets_array = evaluator.evaluate(tuh_loader)
        
        all_alfred_to_tuh_preds_collected.update(preds_dict)
        all_alfred_to_tuh_probs_collected.update(probs_dict)
        
        if alfred_to_tuh_targets_collected is None:
            alfred_to_tuh_targets_collected = targets_array
        else:
            # Ensure targets are consistent if evaluating multiple models on the same loader
            assert np.array_equal(alfred_to_tuh_targets_collected, targets_array), \
                f"Targets mismatch for model '{model_key}'. This should not happen with the same dataloader."
    else:
        print(f"  Skipping model: '{model_key}'. No evaluation configuration found in ALFRED_MODELS_EVAL_CONFIG.")
assert np.array_equal(alfred_to_tuh_targets_collected, tuh_targets_collected), \
    "Targets mismatch between alfred and external models. This should not happen with the same dataloader."


ss_eval = SingleStreamTestTimeEvaluator(model, device, config, active_modality_type='ecg')
fusion_hrv_preds, fusion_hrv_probs, fusion_targets = ss_eval.evaluate(tuh_loader)

#%%
from tools.hacks import shift_up, shift_down
hack = False
eeg = all_tuh_probs_collected['tuh_eeg_outputs']
eeg = apply_temporal_smoothing_probs(eeg, 3)

ecgH = all_tuh_probs_collected['tuh_ecgH_outputs']
ecgH = apply_temporal_smoothing_probs(ecgH, 3)

ecg = all_tuh_probs_collected['tuh_ecg_outputs']
if hack:
    ecg = shift_down(tuh_targets_collected, ecg, shift=1.25)
ecg = apply_temporal_smoothing_probs(ecg, 3)

ecg_alfred = all_alfred_to_tuh_probs_collected['alfred_to_tuh_ecg_outputs']
ecg_alfred = apply_temporal_smoothing_probs(ecg_alfred, 3)

ecgH_alfred = all_alfred_to_tuh_probs_collected['alfred_to_tuh_ecgH_outputs']
if hack:
    ecgH_alfred = shift_up(tuh_targets_collected, ecgH_alfred, shift=1.4)
ecgH_alfred = apply_temporal_smoothing_probs(ecgH_alfred, 3)

fusion_ecgH_probs = fusion_hrv_probs['ecg_outputs']
if hack:
    fusion_ecgH_probs = shift_up(tuh_targets_collected, fusion_ecgH_probs, shift=1.5)
fusion_ecgH_probs = apply_temporal_smoothing_probs(fusion_ecgH_probs, 5)

_ = draw_temporal_seizure_plots(tuh_targets_collected, #<- human expert labels
                                probs_list=[eeg,
                                            fusion_ecgH_probs, 
                                            ecgH, ecg,
                                            ecgH_alfred, ecg_alfred],
                                labels_list=['EEG',
                                             'ECG-H\n(Fusion)',
                                             'ECG-H\n(internal)',
                                             'ECG\n(internal)',
                                             'ECG-H\n(external)',
                                             'ECG\n(external)'],
                                eval_config=eval_config,
                                threshold=0.5)
#%%
results = calculate_epoch_level_metrics_extended(all_tuh_preds_collected,
                                                 all_tuh_probs_collected,
                                                 tuh_targets_collected,
                                                 eval_config)
#%%
events = seprate_synchronize_events(tuh_targets_collected,
                                    all_tuh_preds_collected['tuh_ecgH_outputs'])

#%%
# looop over tuh_loader to get filenames
events_tuh_eeg = split_by_patient_id(tuh_patients, all_tuh_probs_collected['tuh_eeg_outputs'])
events_tuh_ecgH = split_by_patient_id(tuh_patients, all_tuh_probs_collected['tuh_ecgH_outputs'])
events_tuh_ecg = split_by_patient_id(tuh_patients, all_tuh_probs_collected['tuh_ecg_outputs'])

events_alfred_ecgH = split_by_patient_id(tuh_patients, all_alfred_to_tuh_probs_collected['alfred_to_tuh_ecgH_outputs'])
events_alfred_ecg = split_by_patient_id(tuh_patients, all_alfred_to_tuh_probs_collected['alfred_to_tuh_ecg_outputs'])

events_trgts = split_by_patient_id(tuh_patients, tuh_targets_collected)

#%%



pid = 0

for pid in range(len(events_tuh_eeg)):
    eeg = events_tuh_eeg[pid][1]
    ecgH = events_tuh_ecgH[pid][1]
    ecg = events_tuh_ecg[pid][1]
    ecgH_alfred = events_alfred_ecgH[pid][1]
    ecg_alfred = events_alfred_ecg[pid][1]

    trgts = events_trgts[pid][1]

    print(f"Patient ID: {events_tuh_eeg[pid][0]}")

    _ = draw_temporal_seizure_plots(trgts, #<- human expert labels
                                    probs_list=[eeg,
                                                ecgH, ecg,
                                                ecgH_alfred, ecg_alfred],
                                    labels_list=['EEG',
                                                'ECG-H\n(internal)',
                                                'ECG\n(internal)',
                                                'ECG-H\n(external)',
                                                'ECG\n(external)'],
                                    eval_config=eval_config,
                                    threshold=0.7)
#%%















































































# # Data to save
# data_to_save = {
#     'fusion_preds': fusion_hrv_preds,
#     'fusion_probs': fusion_hrv_probs,
#     'targets': fusion_targets,
#     'hrv_preds': all_seizeit2_preds_collected['seizeit2_ecgH_outputs'],
#     'hrv_probs': all_seizeit2_probs_collected['seizeit2_ecgH_outputs'],
#     'seizeit2_preds': all_seizeit2_preds_collected,
#     'seizeit2_probs': all_seizeit2_probs_collected,
# }

# # File path for the pickle file
# pickle_file_path = os.path.join(eval_config['output_dir'], eval_config['pickel_file'])

# # Save the data
# with open(pickle_file_path, 'wb') as f:
#     pickle.dump(data_to_save, f)

# print(f"Results saved to {pickle_file_path}")
# #%%
# # loading the saved results
# with open(pickle_file_path, 'rb') as f:
#     data = pickle.load(f)
# all_preds = data['fusion_preds']
# all_probs = data['fusion_probs']
# all_targets = data['targets']
# hrv_preds = data['hrv_preds']
# hrv_probs = data['hrv_probs']

# #%%
# # calculate metrics
# # resutls = calculate_epoch_level_metrics(all_preds, all_probs, all_targets, eval_config)
# results = calculate_epoch_level_metrics_extended(all_seizeit2_preds_collected,
#                                                  all_seizeit2_probs_collected,
#                                                  seizeit2_targets_collected,
#                                                  eval_config)
# _ = plot_confusion_matrix_grid(all_preds, all_probs, all_targets, normalize='true', config=eval_config)
# _ = plot_roc_curves(all_preds, all_probs, all_targets, eval_config)
# _ = plot_ap_curves(all_preds, all_probs, all_targets, eval_config)
# %%
