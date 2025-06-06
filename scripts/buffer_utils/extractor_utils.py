import torch, os
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.stats import mode
from data.utils import video_transform


class FeatureExtractor:
    """
    Evaluator for test-time processing. 
    This version is modified to extract and save feature vectors from specified
    layers/attributes of the model, along with ground-truth labels.
    """
    def __init__(self, model, device=torch.device("cuda")):
        self.model = model.to(device)
        self.device = device
        # Define how to access the desired features from the model object.
        # These lambdas expect 'm' to be the model instance (self.model)
        # and assume that the forward pass has populated these attributes
        # with batch_size as the first dimension.
        self.feature_accessors = {
            "of_feats": lambda m: m.of_feats_raw,
            "ecg_feats": lambda m: m.ecg_feats_raw,
            "pose_feats": lambda m: m.fusion.pose_fusion.pose_feats_raw, 
            "mod_fusion_feats": lambda m: m.fusion.mod_fusion.fusion_feats_raw,
        }
        # Note: 
        # - Accessing m.fusion.pose_feats assumes model.fusion (Fusion class instance) sets this attribute.
        # - Accessing m.fusion.mod_fusion.fusion_feats assumes model.fusion has an attribute 'mod_fusion'
        #   (e.g., an AdaptiveModFusion instance if fusion type is 'graph'), which in turn sets 'fusion_feats'.
        # Ensure your model's forward pass populates these attributes correctly based on its architecture.

    def extract(self, loader, data_split, output_base_dir):
        """
        Runs inference on `loader`, extracts specified features, saves them to disk,
        and saves the corresponding targets. Each feature file corresponds to one
        10-second window from a sample in the loader.

        Args:
            loader: DataLoader for the data split (e.g., train or validation).
            data_split (str): Name of the data split (e.g., "train", "validation").
                               Used for creating subdirectories.
            output_base_dir (str): Base directory where features will be saved.
                                   Features will be stored in output_base_dir/data_split/feature_name/

        Returns:
            str: Path to the saved numpy file containing all targets for the processed loader.
        """
        self.model.eval()

        feature_save_dirs = {}
        for feat_name in self.feature_accessors.keys():
            save_dir = os.path.join(output_base_dir, data_split, feat_name)
            os.makedirs(save_dir, exist_ok=True)
            feature_save_dirs[feat_name] = save_dir
        
        # Ensure the main data_split directory exists for saving targets.npy
        main_data_split_dir = os.path.join(output_base_dir, data_split)
        os.makedirs(main_data_split_dir, exist_ok=True)

        all_targets_list = []
        # Global counter for sample filenames to ensure uniqueness across batches
        global_sample_idx = 0 

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Extracting features for {data_split}", unit="batch")
            for batch in pbar:
                # ---------------------------------------------------------
                # 1. Preprocess inputs (as in the original evaluator)
                # ---------------------------------------------------------
                frames = video_transform(batch["frames"]).to(self.device, non_blocking=True)
                body   = batch["body"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                face   = batch["face"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                rh     = batch["rh"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                lh     = batch["lh"].permute(0,4,2,3,1).float().to(self.device, non_blocking=True)
                hrv    = batch["hrv"].to(torch.float).to(self.device, non_blocking=True)
                
                # Assuming super_lbls is one-hot, convert to class indices
                targets_batch = torch.argmax(batch["sub_lbls"], dim=1).long()

                # ---------------------------------------------------------
                # 2. Forward pass to populate features in self.model
                # ---------------------------------------------------------
                # The direct output of the model (predictions) is not used here,
                # but the forward pass populates the necessary feature attributes.
                _ = self.model(frames, body, face, rh, lh, hrv) 

                current_batch_size = frames.size(0)
                for i in range(current_batch_size): # Iterate over each sample in the current batch
                    # ---------------------------------------------------------
                    # 3. Extract and save features for the i-th sample
                    # ---------------------------------------------------------
                    for feat_name, accessor_fn in self.feature_accessors.items():
                        try:
                            # accessor_fn(self.model) gets features for the whole batch, e.g., shape [B, 512]
                            batch_features = accessor_fn(self.model) 
                            # Extract the i-th sample's features from the batch
                            sample_feature_tensor = batch_features[i] # Shape [feature_dim], e.g., [512]
                            sample_feature_np = sample_feature_tensor.detach().cpu().numpy()
                            
                            # Save the feature vector for this 10s window
                            file_path = os.path.join(feature_save_dirs[feat_name], f"feature_{global_sample_idx + i:06d}.npy")
                            np.save(file_path, sample_feature_np)
                        except AttributeError:
                            print(f"Warning: Attribute for '{feat_name}' not found. Ensure it's set by the model. Skipping for sample {global_sample_idx + i}.")
                        except Exception as e:
                            print(f"Warning: Error saving '{feat_name}' for sample {global_sample_idx + i}: {e}")
                    
                    # ---------------------------------------------------------
                    # 4. Collect ground-truth label for the i-th sample
                    # ---------------------------------------------------------
                    all_targets_list.append(targets_batch[i].cpu().numpy())

                global_sample_idx += current_batch_size
        
        # ---------------------------------------------------------
        # 5. Save all collected targets
        # ---------------------------------------------------------
        all_targets_np = np.array(all_targets_list)
        targets_save_path = os.path.join(main_data_split_dir, "targets.npy")
        np.save(targets_save_path, all_targets_np)

        print(f"Features for {len(all_targets_list)} samples and their targets for '{data_split}' split saved under {main_data_split_dir}/")
        return targets_save_path
