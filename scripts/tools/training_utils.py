import torch
import torch.nn.functional as F


def log_model_gradients(model, cfg, step, wandb_module):
    """
    Log gradient norms for the entire model and for major components to Weights & Biases.
    Args:
        model (nn.Module): The model to log gradients for.
        cfg (dict): Configuration dictionary.
        current_batch_in_epoch (int): Current batch index within the epoch.
        global_step (int): Global training step count.
        wandb_module: The initialized wandb module/object for logging.
        log_freq (int, optional): Frequency of logging. Defaults to cfg['wandb_grad_log_freq'] or 500.
    """

    if cfg['LOG_WANDB']:
        log_data = {}
        
        # --- Overall Model Gradient Norm ---
        overall_total_norm_sq = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm_sq = p.grad.data.norm(2).item() ** 2
                overall_total_norm_sq += param_norm_sq
        log_data["GRADS/total_model_grad_norm"] = overall_total_norm_sq ** 0.5

        # --- Component-wise Gradient Norms ---
        # These component names must match the attribute names in your MME_Model
        components_map = {
            "slowfast": getattr(model, 'slowfast', None),
            "bodygcn": getattr(model, 'bodygcn', None),
            "facegcn": getattr(model, 'facegcn', None),
            "rhgcn": getattr(model, 'rhgcn', None),
            "lhgcn": getattr(model, 'lhgcn', None),
            "ecg_encoder": getattr(model, 'ewt', None), # Assuming ewt is the ECG encoder
            "fusion": getattr(model, 'fusion', None),
        }

        total_pose_gcn_norm_sq = 0
        pose_gcn_component_keys = ["bodygcn", "facegcn", "rhgcn", "lhgcn"]

        for name, module_instance in components_map.items():
            component_total_norm_sq = 0
            if module_instance and hasattr(module_instance, 'parameters'):
                for p in module_instance.parameters():
                    if p.grad is not None:
                        param_norm_sq = p.grad.data.norm(2).item() ** 2
                        component_total_norm_sq += param_norm_sq
                
                component_norm = component_total_norm_sq ** 0.5
                log_data[f"GRADS/{name}_grad_norm"] = component_norm
                
                if name in pose_gcn_component_keys:
                    total_pose_gcn_norm_sq += component_total_norm_sq
            # else:
            #     print(f"Warning: Module {name} not found or has no parameters in the model for gradient logging.")

        log_data["GRADS/total_posegcn_grad_norm"] = total_pose_gcn_norm_sq ** 0.5
        
        if log_data and hasattr(wandb_module, 'log'): 
            wandb_module.log(log_data, step=step)

def calculate_composite_loss(outputs, target, loss_hyperparams, current_epoch):
    """
    Calculates the composite loss based on model outputs and loss hyperparameters.
    Args:
        outputs (dict): Dictionary of model outputs (e.g., {'fusion_outputs': ..., 'ecg_outputs': ...}).
        target (torch.Tensor): Ground truth labels.
        loss_hyperparams (dict): Dictionary containing all necessary loss functions, weights,
                                 and parameters. Expected keys:
                                 'loss_fn': Main CE loss function.
                                 'fusion_loss_fn_smooth': Optional smoothed CE for fusion.
                                 'ce_weights': Dict of weights for individual CE losses.
                                 'consistent_loss_weight': Weight for fusion-modality consistency.
                                 'inter_group_consistency_weight': Weight for bio vs visual consistency.
                                 'fusion_warmup_epochs': Epochs to warm up fusion consistency.
                                 'bio_signal_keys': List of keys for bio-signal outputs.
                                 'visual_signal_keys': List of keys for visual-signal outputs.
                                 'eps': Epsilon for KL divergence stability.
        current_epoch (int): Current training epoch.
    Returns:
        tuple: (total_loss, loss_dict)
    """
    total_loss = 0.0
    loss_dict = {}

    loss_fn = loss_hyperparams['loss_fn']
    fusion_loss_fn_smooth = loss_hyperparams.get('fusion_loss_fn_smooth', loss_fn) # Fallback
    ce_weights = loss_hyperparams.get('ce_weights', {})
    consistent_loss_weight = loss_hyperparams.get('consistent_loss_weight', 0)
    inter_group_consistency_weight = loss_hyperparams.get('inter_group_consistency_weight', 0)
    fusion_warmup_epochs = loss_hyperparams.get('fusion_warmup_epochs', 0)
    bio_signal_keys = loss_hyperparams.get('bio_signal_keys', [])
    visual_signal_keys = loss_hyperparams.get('visual_signal_keys', [])
    eps = loss_hyperparams.get('eps', 1e-8)

    # --- 1. Calculate individual Cross-Entropy losses ---
    all_logits = {}
    for output_key, logits in outputs.items():
        # if output_key == 'fusion_outputs':
        ce_loss = fusion_loss_fn_smooth(logits, target)
        # else:
        #     ce_loss = loss_fn(logits, target)
        
        weight = ce_weights.get(output_key, 1.0)
        total_loss += weight * ce_loss
        loss_dict[f'CE/{output_key}'] = ce_loss.item()
        all_logits[output_key] = logits

    # --- 2. Consistency Loss (Phased Direction: Modalities <-> Fusion) ---
    if consistent_loss_weight > 0 and 'fusion_outputs' in all_logits:
        fusion_logits = all_logits['fusion_outputs']
        fusion_probs_log = F.log_softmax(fusion_logits, dim=1)
        fusion_probs = F.softmax(fusion_logits, dim=1)

        consistency_loss_value = 0.0
        num_consistency_pairs = 0

        modality_logits_list = {k: lgts for k, lgts in all_logits.items() if k != 'fusion_outputs'}
        
        for key, mod_logits in modality_logits_list.items():
            mod_probs = F.softmax(mod_logits, dim=1)
            mod_probs_log = F.log_softmax(mod_logits, dim=1)

            if current_epoch < fusion_warmup_epochs: # Modalities teach fusion
                # KL(fusion_probs || modality_probs) = sum(fusion_probs * (log(fusion_probs) - log(modality_probs)))
                # F.kl_div expects input=log_softmax(mod_logits), target=softmax(fusion_logits)
                consistency_loss_value += F.kl_div(mod_probs_log, fusion_probs, reduction='batchmean', log_target=False)
            else: # Fusion teaches modalities
                # KL(modality_probs || fusion_probs) = sum(mod_probs * (log(mod_probs) - log(fusion_probs)))
                # F.kl_div expects input=log_softmax(fusion_logits), target=softmax(mod_logits)
                consistency_loss_value += F.kl_div(fusion_probs_log, mod_probs, reduction='batchmean', log_target=False)
            num_consistency_pairs += 1
        
        if num_consistency_pairs > 0:
            consistency_loss_value /= num_consistency_pairs
            total_loss += consistent_loss_weight * consistency_loss_value
            loss_dict['CONSIST/modality_fusion'] = consistency_loss_value.item()

    # --- 3. Inter-Group Consistency Loss (Bio-signals vs. Visual-signals) ---
    if inter_group_consistency_weight > 0:
        bio_group_logits = [all_logits[key] for key in bio_signal_keys if key in all_logits]
        visual_group_logits = [all_logits[key] for key in visual_signal_keys if key in all_logits]

        if bio_group_logits and visual_group_logits:
            avg_bio_probs = torch.stack([F.softmax(lgts, dim=1) for lgts in bio_group_logits]).mean(dim=0)
            avg_visual_probs = torch.stack([F.softmax(lgts, dim=1) for lgts in visual_group_logits]).mean(dim=0)

            kl_bio_visual = F.kl_div((avg_visual_probs + eps).log(), avg_bio_probs, reduction='batchmean', log_target=False)
            kl_visual_bio = F.kl_div((avg_bio_probs + eps).log(), avg_visual_probs, reduction='batchmean', log_target=False)
            
            inter_group_div_loss = (kl_bio_visual + kl_visual_bio) / 2.0
            total_loss += inter_group_consistency_weight * inter_group_div_loss
            loss_dict['CONSIST/inter_group'] = inter_group_div_loss.item()
            
    loss_dict['TOTAL_LOSS'] = total_loss.item()
    return total_loss, loss_dict