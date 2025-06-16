import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.stats import mode
from data.utils import video_transform
from MLstatkit.stats import Bootstrapping, Delong_test

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, f1_score,
    recall_score, accuracy_score, precision_score, cohen_kappa_score
)
class TestTimeEvaluator:
    """
    Evaluator for test-time inference that collects predictions, probabilities,
    and ground-truth labels for all modalities. Uses a tqdm progress bar and
    does not compute any metrics internally.
    """
    def __init__(self, model, device=torch.device("cuda")):
        self.model = model.to(device)
        self.device = device
        # Specify the output keys from the model and corresponding modality names
        self.output_keys = [
            "flow_outputs",
            "body_outputs",
            "face_outputs",
            "rhand_outputs",
            "lhand_outputs",
            "ecg_outputs",
            "joint_pose_outputs",
            "fusion_outputs",
        ]

    def evaluate(self, loader):
        """
        Runs inference on `loader` (e.g., a DataLoader) and returns dictionaries
        of predictions, probabilities, and labels for each modality.
        
        Returns:
            all_preds:   { modality_name: torch.Tensor of shape (N,) }
            all_probs:   { modality_name: torch.Tensor of shape (N, num_classes) }
            all_targets: torch.Tensor of shape (N,)
        """
        self.model.eval()

        # Prepare containers
        all_preds = {key: [] for key in self.output_keys}
        all_probs = {key: [] for key in self.output_keys}
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(loader, desc="Test-time evaluation", unit="batch", ascii='🔴🟢')
            for batch in pbar:
                # ---------------------------------------------------------
                # 1. Preprocess inputs (copy the same transforms as during training)
                # ---------------------------------------------------------
                frames = video_transform(batch["frames"]).to(self.device, non_blocking=True)
                body   = batch["body"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                face   = batch["face"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                rh     = batch["rh"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                lh     = batch["lh"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                hrv    = batch["hrv"].to(torch.float).to(self.device, non_blocking=True)
                targets = torch.argmax(batch["super_lbls"], dim=1).long().to(self.device, non_blocking=True)

                # ---------------------------------------------------------
                # 2. Forward pass
                # ---------------------------------------------------------
                outputs = self.model(frames, body, face, rh, lh, hrv)
                # outputs is expected to be a dict with keys matching self.output_keys

                # ---------------------------------------------------------
                # 3. Extract predicted class indices and softmax probabilities
                # ---------------------------------------------------------
                for key in self.output_keys:
                    logits = outputs[key]               # shape: (batch_size, num_classes)
                    prob   = F.softmax(logits, dim=1)   # shape: (batch_size, num_classes)
                    pred   = torch.argmax(prob, dim=1)  # shape: (batch_size,)

                    all_preds[key].append(pred.cpu())
                    all_probs[key].append(prob.cpu())

                # ---------------------------------------------------------
                # 4. Collect ground-truth labels
                # ---------------------------------------------------------
                all_targets.append(targets.cpu())

        # ---------------------------------------------------------
        # 5. Concatenate tensors across all batches
        # ---------------------------------------------------------
        for key in self.output_keys:
            all_preds[key] = torch.cat(all_preds[key], dim=0).numpy()   # shape: (N,)
            # all_probs[key] = torch.cat(all_probs[key], dim=0).numpy()   # shape: (N, num_classes)
            all_probs[key] = torch.cat(all_probs[key], dim=0)[:, 1].numpy()  # shape: (N,)
        all_targets = torch.cat(all_targets, dim=0).numpy()             # shape: (N,)

        return all_preds, all_probs, all_targets

class TestTimeModalityEvaluator:
    """
    Evaluator for test-time inference when only ECG modality is available.
    Returns predictions, probabilities, and labels using the same data-structure
    interface (dicts keyed by output_keys and a single targets array).
    """
    def __init__(self, model, device=torch.device("cuda"),
                 output_keys=None, modality_name=None):
        self.model = model.to(device)
        self.device = device
        # Only ECG outputs
        self.output_keys = output_keys#["ecg_outputs"]
        # modality name to get from data loader
        self.modality_name = modality_name

    def evaluate(self, loader):
        """
        Runs inference on `loader` and returns:
            all_preds:   { 'ecg_outputs': np.ndarray of shape (N,) }
            all_probs:   { 'ecg_outputs': np.ndarray of shape (N,) }  # probability of class 1
            all_targets: np.ndarray of shape (N,)
        """
        self.model.eval()

        # Prepare containers
        all_preds = {key: [] for key in self.output_keys}
        all_probs = {key: [] for key in self.output_keys}
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(loader, desc="ECG Test-time evaluation", unit="batch")
            for batch in pbar:
                # ---------------------------------------------------------
                # 1. Preprocess ECG input only
                # ---------------------------------------------------------
                hrv = batch[self.modality_name].to(torch.float).to(self.device, non_blocking=True)
                # if targets are single dimension, convert to long
                if batch["super_lbls"].ndim == 1:
                    targets = batch["super_lbls"].long().to(self.device, non_blocking=True)
                else:
                    targets = torch.argmax(batch["super_lbls"], dim=1)\
                                    .long().to(self.device, non_blocking=True)

                # ---------------------------------------------------------
                # 2. Forward pass (ECG-only model)
                # ---------------------------------------------------------
                outputs = self.model(hrv)  
                # outputs should be a dict: { "ecg_outputs": logits_tensor }

                # ---------------------------------------------------------
                # 3. Extract preds & probs
                # ---------------------------------------------------------
                for key in self.output_keys:
                    logits = outputs #[key]             # [batch_size, num_classes]
                    prob   = F.softmax(logits, dim=1) # [batch_size, num_classes]
                    pred   = torch.argmax(prob, dim=1)

                    all_preds[key].append(pred.cpu())
                    all_probs[key].append(prob.cpu())

                # ---------------------------------------------------------
                # 4. Collect ground-truth
                # ---------------------------------------------------------
                all_targets.append(targets.cpu())

        # ---------------------------------------------------------
        # 5. Concatenate & convert to numpy
        # ---------------------------------------------------------
        for key in self.output_keys:
            all_preds[key] = torch.cat(all_preds[key], dim=0).numpy()         # (N,)
            # keep only positive-class probability for compatibility
            all_probs[key] = torch.cat(all_probs[key], dim=0)[:, 1].numpy()   # (N,)

        all_targets = torch.cat(all_targets, dim=0).numpy()  # (N,)

        return all_preds, all_probs, all_targets
    
    
class PartialStreamTestTimeEvaluator: # Renamed to match your prompt
    """
    Evaluator for test-time inference with a multi-modal model (MME_Model)
    when one or more input modality streams are active (fed with real data)
    and others are fed with zero tensors.

    Collects 'fusion_outputs', individual outputs for each active modality
    (e.g., 'flow_outputs', 'body_outputs'), and 'joint_pose_outputs' if any
    pose-related modality is active.
    """
    def __init__(self, model, device, config, active_modality_types: list):
        """
        Args:
            model: The MME_Model instance.
            device: torch.device for computation.
            config: Global configuration dictionary, used for tensor shapes.
            active_modality_types (list[str]): Specifies the active input modalities.
        """
        active_modalities_str = ", ".join(active_modality_types) if active_modality_types else "None"
        print(40*"-")
        print(f"Initializing PartialStreamTestTimeEvaluator for active modalities: [{active_modalities_str}]")
        print(40*"=")
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.model_input_specs = {
            "frames": {
                "batch_data_key": "frames",
                "processor": lambda data: video_transform(data).to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 3, self.config['flow_frames'], self.config['video_height'], self.config['video_width']),
                "dtype": torch.float
            },
            "body": {
                "batch_data_key": "body",
                "processor": lambda data: data.permute(0,4,2,3,1).float().to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 1, self.config['pose_frames'], 17, 3),
                "dtype": torch.float
            },
            "face": {
                "batch_data_key": "face",
                "processor": lambda data: data.permute(0,4,2,3,1).float().to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 1, self.config['pose_frames'], 70, 3),
                "dtype": torch.float
            },
            "rhand": {
                "batch_data_key": "rhand",
                "processor": lambda data: data.permute(0,4,2,3,1).float().to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 1, self.config['pose_frames'], 21, 3),
                "dtype": torch.float
            },
            "lhand": {
                "batch_data_key": "lhand",
                "processor": lambda data: data.permute(0,4,2,3,1).float().to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 1, self.config['pose_frames'], 21, 3),
                "dtype": torch.float
            },
            "hrv": {
                "batch_data_key": "hrv",
                "processor": lambda data: data.to(torch.float).to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 19, int(self.config['ecg_freq'] * self.config['sample_duration'])),
                "dtype": torch.float
            }
        }
        
        # Maps user-facing modality type strings to their model input argument name,
        # a user-facing name for descriptions, and the key for their individual output branch.
        type_to_details_map = {
            "flow":  {"model_arg_name": "frames", "user_facing_name": "flow",  "individual_output_key": "flow_outputs"},
            "ecg":   {"model_arg_name": "hrv",    "user_facing_name": "ecg",   "individual_output_key": "ecg_outputs"},
            "body":  {"model_arg_name": "body",   "user_facing_name": "body",  "individual_output_key": "body_outputs"},
            "face":  {"model_arg_name": "face",   "user_facing_name": "face",  "individual_output_key": "face_outputs"},
            "rhand": {"model_arg_name": "rhand",  "user_facing_name": "rhand", "individual_output_key": "rhand_outputs"},
            "lhand": {"model_arg_name": "lhand",  "user_facing_name": "lhand", "individual_output_key": "lhand_outputs"},
        }
        
        self.pose_input_types = ["body", "face", "rhand", "lhand"] # User-facing types that are considered pose-related

        if not isinstance(active_modality_types, list):
            raise TypeError(f"active_modality_types must be a list, got {type(active_modality_types)}")

        self.active_model_input_args = [] 
        self.output_keys_to_collect = ["fusion_outputs"] # Always collect fusion
        active_user_facing_names_for_desc = []
        is_any_pose_modality_active = False

        for active_type in active_modality_types:
            if active_type not in type_to_details_map:
                raise ValueError(f"Unsupported active_modality_type: '{active_type}'. Supported types are: {list(type_to_details_map.keys())}")
            
            details = type_to_details_map[active_type]
            self.active_model_input_args.append(details["model_arg_name"])
            active_user_facing_names_for_desc.append(details["user_facing_name"])

            # Add the individual output key for this active modality if defined
            if "individual_output_key" in details:
                self.output_keys_to_collect.append(details["individual_output_key"])

            if active_type in self.pose_input_types:
                is_any_pose_modality_active = True

        if is_any_pose_modality_active:
            self.output_keys_to_collect.append("joint_pose_outputs") # Add combined pose output if any pose input is active
        
        self.active_model_input_args = sorted(list(set(self.active_model_input_args))) 
        self.output_keys_to_collect = sorted(list(set(self.output_keys_to_collect))) # Make unique and sort
        
        print(f"Output keys to collect: {self.output_keys_to_collect}") # For debugging
        self.tqdm_desc_active_modalities_str = ", ".join(sorted(list(set(active_user_facing_names_for_desc))))


    def _get_input_tensor(self, model_arg_name, batch, batch_size):
        spec = self.model_input_specs[model_arg_name]
        if model_arg_name in self.active_model_input_args:
            batch_data = batch[spec["batch_data_key"]]
            return spec["processor"](batch_data)
        else:
            return torch.zeros(spec["zero_shape_fn"](batch_size), dtype=spec["dtype"], device=self.device)

    def evaluate(self, loader):
        self.model.eval()
        filenames = []
        all_preds = {key: [] for key in self.output_keys_to_collect}
        all_probs = {key: [] for key in self.output_keys_to_collect}
        all_targets = []

        desc = f"Partial-stream eval (active: {self.tqdm_desc_active_modalities_str if self.tqdm_desc_active_modalities_str else 'None'})"
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc, unit="batch", ascii='🔵⚪')
            for batch_idx, batch in enumerate(pbar): # Added batch_idx for more specific warnings
                batch_size = len(batch["super_lbls"]) # Assuming 'super_lbls' determines batch size
                
                filenames.append(batch['filename'][0])
                
                frames_input = self._get_input_tensor("frames", batch, batch_size)
                body_input   = self._get_input_tensor("body",   batch, batch_size)
                face_input   = self._get_input_tensor("face",   batch, batch_size)
                rhand_input  = self._get_input_tensor("rhand",  batch, batch_size)
                lhand_input  = self._get_input_tensor("lhand",  batch, batch_size)
                hrv_input    = self._get_input_tensor("hrv",    batch, batch_size)
                
                if batch["super_lbls"].ndim == 1:
                    targets = batch["super_lbls"].long().to(self.device, non_blocking=True)
                else:
                    targets = torch.argmax(batch["super_lbls"], dim=1)\
                                    .long().to(self.device, non_blocking=True)
                
                # Ensure all model inputs are provided, even if they are zero tensors
                outputs = self.model(
                    frames=frames_input, 
                    body=body_input, 
                    face=face_input, 
                    rh=rhand_input,  # Assuming 'rh' is the model argument name for rhand
                    lh=lhand_input,  # Assuming 'lh' is the model argument name for lhand
                    ecg=hrv_input    # Assuming 'ecg' is the model argument name for hrv
                )


                for key in self.output_keys_to_collect:
                    if key in outputs:
                        logits = outputs[key]
                        prob   = F.softmax(logits, dim=1)
                        pred   = torch.argmax(prob, dim=1)
                        all_preds[key].append(pred.cpu())
                        all_probs[key].append(prob.cpu()) # Store full probabilities for now
                    else:
                        print(f"Warning (Batch {batch_idx}): Expected output key '{key}' not found in model outputs. This key will have empty results.")
                        # all_preds[key] and all_probs[key] will remain empty lists for this key if never found
                        pass 

                all_targets.append(targets.cpu())

        for key in self.output_keys_to_collect:
            if all_preds[key]: # Check if list is not empty (i.e., key was found in model outputs at least once)
                 all_preds[key] = torch.cat(all_preds[key], dim=0).numpy()
                 # For probabilities, decide how to handle multi-class.
                 # If you want prob of predicted class:
                 # probs_for_key = torch.cat(all_probs[key], dim=0)
                 # all_probs[key] = probs_for_key.gather(1, all_preds[key].reshape(-1,1)).squeeze().numpy()
                 # If you want prob of class 1 (for binary or specific interest):
                 probs_for_key = torch.cat(all_probs[key], dim=0)
                 if probs_for_key.shape[1] > 1: # Check if multi-class
                    all_probs[key] = probs_for_key[:, 1].numpy() # Prob of class 1
                 else: # Single class output (e.g. regression, or already a probability)
                    all_probs[key] = probs_for_key.squeeze().numpy()

            else: 
                 all_preds[key] = np.array([]) 
                 all_probs[key] = np.array([])


        all_targets = torch.cat(all_targets, dim=0).numpy()

        return all_preds, all_probs, all_targets, filenames

class SingleStreamTestTimeEvaluator:
    """
    Evaluator for test-time inference with a multi-modal model (MME_Model)
    when only a single input modality stream is active (fed with real data)
    and others are fed with zero tensors.

    Returns predictions and probabilities for 'fusion_outputs' and the output
    branch corresponding to the active input modality.
    """
    def __init__(self, model, device, config, active_modality_type):
        """
        Args:
            model: The MME_Model instance.
            device: torch.device for computation.
            config: Global configuration dictionary, used for tensor shapes.
            active_modality_type (str): Specifies the active input modality.
                Expected values: "flow", "ecg", "body", "face", "rhand", "lhand".
        """
        print(40*"<>")
        print(f"Initializing SingleStreamTestTimeEvaluator for active modality: {active_modality_type}")
        print(40*"<>")
        self.model = model.to(device)
        self.device = device
        self.config = config

        # Define all possible model input arguments and their processing/shape generation
        # These correspond to the arguments of model.forward(frames, body, face, rh, lh, hrv)
        self.model_input_specs = {
            "frames": {
                "batch_data_key": "frames",
                "processor": lambda data: video_transform(data).to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 3, self.config['flow_frames'], self.config['video_height'], self.config['video_width']),
                "dtype": torch.float
            },
            "body": {
                "batch_data_key": "body",
                "processor": lambda data: data.permute(0,4,2,3,1).float().to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 1, self.config['pose_frames'], 17, 3), # M, T, V_body, C
                "dtype": torch.float
            },
            "face": {
                "batch_data_key": "face",
                "processor": lambda data: data.permute(0,4,2,3,1).float().to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 1, self.config['pose_frames'], 70, 3), # M, T, V_face, C
                "dtype": torch.float
            },
            "rhand": {
                "batch_data_key": "rhand",
                "processor": lambda data: data.permute(0,4,2,3,1).float().to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 1, self.config['pose_frames'], 21, 3), # M, T, V_rhand, C
                "dtype": torch.float
            },
            "lhand": {
                "batch_data_key": "lhand",
                "processor": lambda data: data.permute(0,4,2,3,1).float().to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 1, self.config['pose_frames'], 21, 3), # M, T, V_lhand, C
                "dtype": torch.float
            },
            "hrv": { # Corresponds to ECG input
                "batch_data_key": "hrv",
                "processor": lambda data: data.to(torch.float).to(self.device, non_blocking=True),
                "zero_shape_fn": lambda b_size: (b_size, 19, int(self.config['ecg_freq'] * self.config['sample_duration'])), # C_hrv, T_ecg
                "dtype": torch.float
            }
        }
        
        # Map active_modality_type to model input argument name and corresponding output stem
        # This defines which input is real and which output branch (other than fusion) to collect
        type_to_details_map = {
            "flow":  {"model_arg_name": "frames", "output_stem": "flow"},
            "ecg":   {"model_arg_name": "hrv",    "output_stem": "ecg"},
            "body":  {"model_arg_name": "body",   "output_stem": "body"}, # Or "joint_pose" if preferred
            "face":  {"model_arg_name": "face",   "output_stem": "face"},
            "rhand": {"model_arg_name": "rhand",  "output_stem": "rhand"},
            "lhand": {"model_arg_name": "lhand",  "output_stem": "lhand"},
            # Add more mappings if "pose" means something combined or different
        }

        if active_modality_type not in type_to_details_map:
            raise ValueError(f"Unsupported active_modality_type: {active_modality_type}. Supported types are: {list(type_to_details_map.keys())}")
        
        self.active_modality_details = type_to_details_map[active_modality_type]
        self.active_model_input_arg = self.active_modality_details["model_arg_name"]

        # Determine which output keys to collect
        self.output_keys_to_collect = ["fusion_outputs"]
        specific_modality_output_key = f"{self.active_modality_details['output_stem']}_outputs"
        
        # Full list of possible output branches from the MME_Model
        all_possible_model_outputs = [
            "flow_outputs", "body_outputs", "face_outputs", "rhand_outputs",
            "lhand_outputs", "ecg_outputs", "joint_pose_outputs", "fusion_outputs"
        ]
        if specific_modality_output_key in all_possible_model_outputs and \
           specific_modality_output_key != "fusion_outputs":
            self.output_keys_to_collect.append(specific_modality_output_key)
        
        # Ensure uniqueness if somehow the specific key was fusion
        self.output_keys_to_collect = sorted(list(set(self.output_keys_to_collect)))


    def _get_input_tensor(self, model_arg_name, batch, batch_size):
        spec = self.model_input_specs[model_arg_name]
        if model_arg_name == self.active_model_input_arg:
            batch_data = batch[spec["batch_data_key"]]
            return spec["processor"](batch_data)
        else:
            return torch.zeros(spec["zero_shape_fn"](batch_size), dtype=spec["dtype"], device=self.device)

    def evaluate(self, loader):
        self.model.eval()

        all_preds = {key: [] for key in self.output_keys_to_collect}
        all_probs = {key: [] for key in self.output_keys_to_collect}
        all_targets = []

        desc = f"Single-stream eval (active: {self.active_modality_details['model_arg_name']})"
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc, unit="batch", ascii='🔵⚪')
            for batch in pbar:
                # Determine batch size (use a common key like 'super_lbls' or the active one)
                # Assuming 'super_lbls' is always present and indicates batch size
                batch_size = len(batch["super_lbls"])
                
                # 1. Prepare all model inputs (active one from batch, others as zeros)
                frames_input = self._get_input_tensor("frames", batch, batch_size)
                body_input   = self._get_input_tensor("body",   batch, batch_size)
                face_input   = self._get_input_tensor("face",   batch, batch_size)
                rhand_input  = self._get_input_tensor("rhand",  batch, batch_size)
                lhand_input  = self._get_input_tensor("lhand",  batch, batch_size)
                hrv_input    = self._get_input_tensor("hrv",    batch, batch_size)
                
                if batch["super_lbls"].ndim == 1:
                    targets = batch["super_lbls"].long().to(self.device, non_blocking=True)
                else:
                    targets = torch.argmax(batch["super_lbls"], dim=1)\
                                    .long().to(self.device, non_blocking=True)
                # 2. Forward pass
                outputs = self.model(frames_input, body_input, face_input, rhand_input, lhand_input, hrv_input)

                # 3. Extract predictions and probabilities for specified output keys
                for key in self.output_keys_to_collect:
                    if key in outputs:
                        logits = outputs[key]
                        prob   = F.softmax(logits, dim=1)
                        pred   = torch.argmax(prob, dim=1)
                        all_preds[key].append(pred.cpu())
                        all_probs[key].append(prob.cpu()) # Store full probabilities first
                    else:
                        print(f"Warning: Expected output key '{key}' not found in model outputs.")
                        # Append empty tensors or handle as error, for now, this might lead to issues in concatenation
                        # For robustness, ensure model always produces keys listed in self.output_keys_to_collect
                        # or handle missing keys gracefully during concatenation.

                all_targets.append(targets.cpu())

        # 5. Concatenate tensors
        for key in self.output_keys_to_collect:
            if all_preds[key]: # Check if list is not empty
                 all_preds[key] = torch.cat(all_preds[key], dim=0).numpy()
                 all_probs[key] = torch.cat(all_probs[key], dim=0)[:, 1].numpy() # Prob of positive class
            else: # Handle cases where a key might not have been collected if not in outputs
                 all_preds[key] = np.array([])
                 all_probs[key] = np.array([])


        all_targets = torch.cat(all_targets, dim=0).numpy()

        return all_preds, all_probs, all_targets
