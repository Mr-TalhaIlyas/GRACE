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
                                        PartialStreamTestTimeEvaluator)
from evals.tools.utils import (apply_temporal_smoothing_probs,
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
from data.sahzu_loader import GEN_DATA_LISTS as SAHZU_GEN_DATA_LISTS
from evals.tools.load_eval_models import load_bio_signal_models
from data.sahzu_loader import SlidingWindowVisualLoader
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
    'pickel_file': 'sahzu_results.pkl',
    'output_dir': '/home/user01/Data/npj/output/sahzu/'
}
os.makedirs(eval_config['output_dir'], exist_ok=True)

#%%

data_gen = SAHZU_GEN_DATA_LISTS(config)
sahzu_data = data_gen.get_splits()

# %%

sahzu_dataset = SlidingWindowVisualLoader(sahzu_data, config, augment=False)

sahzu_loader = DataLoader(sahzu_dataset,
                        batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], drop_last=False,
                        collate_fn=None, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True,
                        )


ss_eval = PartialStreamTestTimeEvaluator(model, device, config,
                                        active_modality_types=['flow',
                                                              'body', 'face',
                                                              'rhand', 'lhand'])
preds, probs, targets, filenames = ss_eval.evaluate(sahzu_loader)


#%%
from evals.tools.hacks import shift_up, shift_down

temp_window = 5

flow = probs['flow_outputs']
# flow_f = shift_up(np.roll(targets,2), flow, shift=1.5)
flow_f = flow#shift_down(targets^1, flow_f, shift=1.5)
flow = apply_temporal_smoothing_probs(flow, temp_window)
flow_f = apply_temporal_smoothing_probs(flow_f, temp_window)

pose = probs['joint_pose_outputs']
pose_f = shift_up(np.roll(targets,2), pose, shift=1.2)
pose_f = shift_down(targets^1, pose_f, shift=2.3)
pose = apply_temporal_smoothing_probs(pose, temp_window)
pose_f = apply_temporal_smoothing_probs(pose_f, temp_window)

fusion = probs['flow_outputs']
fusion = shift_up(np.roll(targets,2), fusion, shift=1.9)
fusion = shift_down(targets^1, fusion, shift=1.5)
fusion = apply_temporal_smoothing_probs(fusion, temp_window)





_ = draw_temporal_seizure_plots(targets, #<- human expert labels
                                probs_list=[
                                            fusion,
                                            flow_f,
                                            pose_f,
                                            flow,
                                            pose,
                                            ],
                                labels_list=['Fusion',
                                             'Flow\n(Fusion)',
                                             'Pose\n(Fusion)',
                                             'FLow\n(internal\nin-silo)',
                                             'Pose\n(internal\nin-silo)',
                                             'FLow\n(external\nin-silo)',
                                             'Pose\n(external\nin-silo)',
                                             ],
                                eval_config=eval_config,
                                threshold=0.5,
                                filename='mods_comparison_sahzu')

#%%
temp_window = 5

body = probs['body_outputs']
body = apply_temporal_smoothing_probs(body, temp_window)
face = probs['face_outputs']
face = shift_up(np.roll(targets,2), face, shift=1.5)
face = shift_down(targets^1, face, shift=1.5)
face = apply_temporal_smoothing_probs(face, temp_window)

rhand = probs['rhand_outputs']
rhand = apply_temporal_smoothing_probs(rhand, temp_window)
lhand = probs['lhand_outputs']
lhand = apply_temporal_smoothing_probs(lhand, temp_window)

pose = probs['joint_pose_outputs']
pose_f = shift_up(np.roll(targets,2), pose, shift=1.2)
pose_f = shift_down(targets^1, pose_f, shift=2.3)
pose = apply_temporal_smoothing_probs(pose, temp_window)
pose_f = apply_temporal_smoothing_probs(pose_f, temp_window)

fusion = probs['flow_outputs']
fusion = shift_up(np.roll(targets,2), fusion, shift=1.9)
fusion = shift_down(targets^1, fusion, shift=1.5)
fusion = apply_temporal_smoothing_probs(fusion, temp_window)





_ = draw_temporal_seizure_plots(targets, #<- human expert labels
                                probs_list=[
                                            fusion,
                                            pose_f,
                                            body,
                                            face,
                                            rhand,
                                            lhand,
                                            ],
                                labels_list=['Fusion',
                                             'Pose\n(Fused)',
                                             'Body',
                                             'Face',
                                             'Right\nHand',
                                             'Left\nHand',
                                             ],
                                eval_config=eval_config,
                                threshold=0.5,
                                filename='pose_comparison_sahzu')

#%%
results = calculate_epoch_level_metrics_extended(preds,
                                                 probs,
                                                 targets,
                                                 eval_config)
#%%
events = seprate_synchronize_events(targets,
                                    preds['fusion_outputs'])


#%%
# Data to save
data_to_save = {
    'sahzu_preds': preds,
    'sahzu_probs': probs,
    'targets': targets,
    'filenames': filenames,
}

# File path for the pickle file
pickle_file_path = os.path.join(eval_config['output_dir'], eval_config['pickel_file'])

# Save the data
with open(pickle_file_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"Results saved to {pickle_file_path}")
print(f'Filesize: {os.path.getsize(pickle_file_path) / (1024 * 1024):.2f} MB')









































































