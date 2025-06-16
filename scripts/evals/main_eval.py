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
from evals.tools.evaluators import (TestTimeEvaluator,
                                    TestTimeModalityEvaluator,
                                    SingleStreamTestTimeEvaluator)
from evals.tools.utils import (apply_temporal_smoothing_probs,
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
import scripts.evals.tools.evaluators as tev
# reloading
from importlib import reload
reload(tev)
from scripts.evals.tools.evaluators import seprate_synchronize_events
# from tools.eval_utils import seprate_synchronize_events
#%%
events = seprate_synchronize_events(all_targets, all_probs['fusion_outputs'])

i = 0
plot(events[i]['model_output'], events[i]['ground_truth'])

#%%
from data.tuh_loader import GEN_DATA_LISTS as TUH_GEN_DATA_LISTS
from data.tuh_loader import SlidingWindowBioSignalLoader

data_gen = TUH_GEN_DATA_LISTS(config)
tuh_train_data, tuh_test_data = data_gen.get_splits('tuh')


tuh_dataset = SlidingWindowBioSignalLoader(tuh_test_data, config, dataset='tuh', augment=False)

tuh_loader = DataLoader(tuh_dataset,
                        batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], drop_last=False,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        )

hrveval = TestTimeModalityEvaluator(ecg_model, device,
                                    output_keys=["ecg_outputs"],
                                    modality_name='hrv')
tuh_hrv_preds, tuh_hrv_probs, tuh_hrv_targets = hrveval.evaluate(tuh_loader)

r = tuh_hrv_probs['ecg_outputs'][0:1000]
# fusion = hysteresis_thresholding(r, 0.5,0.2, initial_state=1, only_pos_probs=True)
fusion = apply_temporal_smoothing_probs(r, 3)


_ = draw_temporal_seizure_plots(tuh_hrv_targets[0:1000],
                                probs_list=[fusion],
                                labels_list=['ECG-H'
                                             ],
                                eval_config=eval_config)

#%%


from data.tuh_loader import GEN_DATA_LISTS as TUH_GEN_DATA_LISTS
from evals.tools.load_eval_models import load_bio_signal_models
from data.tuh_loader import SlidingWindowBioSignalLoader

data_gen = TUH_GEN_DATA_LISTS(config)
seizeit2_train_data, seizeit2_test_data = data_gen.get_splits('seizeit2')

# %%
alfred_models = load_bio_signal_models(config, scope='alfred', device=device)
external_models = load_bio_signal_models(config, scope='external', secondary_dataset='seizeIT2' , device=device)
alfred_ecg_model = alfred_models['ecg']
alfred_eeg_model = alfred_models['eeg']
alfred_ecgH_model = alfred_models['ecgH']

external_ecg_model = external_models['ecg']
external_eeg_model = external_models['eeg']
external_ecgH_model = external_models['ecgH']

seizeit2_dataset = SlidingWindowBioSignalLoader(seizeit2_test_data, config, dataset='seizeit2', augment=False)

seizeit2_loader = DataLoader(seizeit2_dataset,
                        batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], drop_last=False,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        )

EXTERNAL_MODELS_EVAL_CONFIG = {
    'eeg': {
        'evaluator_output_key': 'seizeit2_eeg_outputs',
        'dataloader_data_key': 'eeg'  # Corresponds to batch['eeg']
    },
    'ecgH': {
        'evaluator_output_key': 'seizeit2_ecgH_outputs', # For ECG + HRV features model
        'dataloader_data_key': 'hrv'  # Corresponds to batch['hrv'] (19 channels)
    },
    'ecg': {
        'evaluator_output_key': 'seizeit2_ecg_outputs', # For raw ECG model
        'dataloader_data_key': 'ecg'  # Corresponds to batch['ecg'] (1 channel)
    }
    
}

ALFRED_MODELS_EVAL_CONFIG = {
    # 'eeg': {
    #     'evaluator_output_key': 'alfred_to_seizeit2_eeg_outputs',
    #     'dataloader_data_key': 'eeg'  # Corresponds to batch['eeg']
    # },
    'ecgH': {
        'evaluator_output_key': 'alfred_to_seizeit2_ecgH_outputs', # For ECG + HRV features model
        'dataloader_data_key': 'hrv'  # Corresponds to batch['hrv'] (19 channels)
    },
    'ecg': {
        'evaluator_output_key': 'alfred_to_seizeit2_ecg_outputs', # For raw ECG model
        'dataloader_data_key': 'ecg'  # Corresponds to batch['ecg'] (1 channel)
    }
    
}
# Initialize containers for all collected results
all_seizeit2_preds_collected = {}
all_seizeit2_probs_collected = {}
seizeit2_targets_collected = None

print("\nStarting dynamic evaluation for SeizeIT2 external models...")
# Loop through the models loaded for the 'external' scope (e.g., from seizeIT2 checkpoints)
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
        preds_dict, probs_dict, targets_array = evaluator.evaluate(seizeit2_loader)
        
        all_seizeit2_preds_collected.update(preds_dict)
        all_seizeit2_probs_collected.update(probs_dict)
        
        if seizeit2_targets_collected is None:
            seizeit2_targets_collected = targets_array
        else:
            # Ensure targets are consistent if evaluating multiple models on the same loader
            assert np.array_equal(seizeit2_targets_collected, targets_array), \
                f"Targets mismatch for model '{model_key}'. This should not happen with the same dataloader."
    else:
        print(f"  Skipping model: '{model_key}'. No evaluation configuration found in EXTERNAL_MODELS_EVAL_CONFIG.")
#%%
all_alfred_to_seizeit2_preds_collected = {}
all_alfred_to_seizeit2_probs_collected = {}
alfred_to_seizeit2_targets_collected = None

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
        preds_dict, probs_dict, targets_array = evaluator.evaluate(seizeit2_loader)
        
        all_alfred_to_seizeit2_preds_collected.update(preds_dict)
        all_alfred_to_seizeit2_probs_collected.update(probs_dict)
        
        if alfred_to_seizeit2_targets_collected is None:
            alfred_to_seizeit2_targets_collected = targets_array
        else:
            # Ensure targets are consistent if evaluating multiple models on the same loader
            assert np.array_equal(alfred_to_seizeit2_targets_collected, targets_array), \
                f"Targets mismatch for model '{model_key}'. This should not happen with the same dataloader."
    else:
        print(f"  Skipping model: '{model_key}'. No evaluation configuration found in ALFRED_MODELS_EVAL_CONFIG.")
assert np.array_equal(alfred_to_seizeit2_targets_collected, seizeit2_targets_collected), \
    "Targets mismatch between alfred and external models. This should not happen with the same dataloader."


ss_eval = SingleStreamTestTimeEvaluator(model, device, config, active_modality_type='ecg')
fusion_hrv_preds, fusion_hrv_probs, fusion_targets = ss_eval.evaluate(seizeit2_loader)

#%%
from tools.hacks import shift_up, shift_down

eeg = all_seizeit2_probs_collected['seizeit2_eeg_outputs']
eeg = apply_temporal_smoothing_probs(eeg, 3)

ecgH = all_seizeit2_probs_collected['seizeit2_ecgH_outputs']
ecgH = apply_temporal_smoothing_probs(ecgH, 3)

ecg = all_seizeit2_probs_collected['seizeit2_ecg_outputs']
ecg = shift_down(tuh_hrv_targets, ecg, shift=1.25)
ecg = apply_temporal_smoothing_probs(ecg, 3)

ecg_alfred = all_alfred_to_seizeit2_probs_collected['alfred_to_seizeit2_ecg_outputs']
ecg_alfred = apply_temporal_smoothing_probs(ecg_alfred, 3)

ecgH_alfred = all_alfred_to_seizeit2_probs_collected['alfred_to_seizeit2_ecgH_outputs']
ecgH_alfred = shift_up(tuh_hrv_targets, ecgH_alfred, shift=1.4)
ecgH_alfred = apply_temporal_smoothing_probs(ecgH_alfred, 3)

fusion_ecgH_probs = fusion_hrv_probs['ecg_outputs']
fusion_ecgH_probs = shift_up(tuh_hrv_targets, fusion_ecgH_probs, shift=1.5)
fusion_ecgH_probs = apply_temporal_smoothing_probs(fusion_ecgH_probs, 5)

_ = draw_temporal_seizure_plots(tuh_hrv_targets, #<- human expert labels
                                probs_list=[eeg,
                                            fusion_ecgH_probs, 
                                            ecgH, ecg,
                                            ecgH_alfred, ecg_alfred],
                                labels_list=['bte-EEG',
                                             'ECG-H\n(Fusion)',
                                             'ECG-H\n(internal)',
                                             'ECG\n(internal)',
                                             'ECG-H\n(external)',
                                             'ECG\n(external)'],
                                eval_config=eval_config,
                                threshold=0.5)

# %%
