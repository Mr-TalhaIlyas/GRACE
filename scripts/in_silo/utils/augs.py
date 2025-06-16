# ...existing code...
from tsai.data.mixed_augmentation import MixUp1d, CutMix1d # Ensure these are imported
from torch.distributions.beta import Beta
from fastai.callback.core import Callback
from fastai.layers import NoneReduce
from tsai.imports import unsqueeze # L is also from tsai.imports if needed for complex yb
from tsai.data.mixed_augmentation import _reduce_loss # For reducing the loss after NoneReduce
import random
import torch
import numpy as np
class ExclusiveMixer(Callback):
    """
    Applies either CutMix1d or MixUp1d exclusively to a batch, with a given probability.
    - p_apply_any: Probability that any augmentation (CutMix or MixUp) is applied.
    - p_cutmix_if_apply: If an augmentation is applied, this is the probability it's CutMix1d (otherwise MixUp1d).
    """
    run_valid = False # Augmentations typically run only on training data
    order = MixUp1d.order # Use the same order as MixUp/CutMix to ensure it runs at the right time

    def __init__(self, cutmix_alpha=1.0, mixup_alpha=0.4,
                 p_apply_any=0.5, p_cutmix_if_apply=0.5):
        # store_attr() # Stores all __init__ args as self.attributes
        self.p_apply_any = p_apply_any
        self.p_cutmix_if_apply = p_cutmix_if_apply
        self.cutmix_distrib = Beta(float(cutmix_alpha), float(cutmix_alpha))
        self.mixup_distrib = Beta(float(mixup_alpha), float(mixup_alpha))

    def before_train(self):
        self.old_lf = self.learn.loss_func
        self.learn.loss_func = self._mixed_loss_func
        # Determine if batches are labeled (have y)
        try:
            b = self.learn.dls.one_batch()
            self.labeled = len(b) > 1
        except Exception: # Handle cases where one_batch might fail or dls is not fully set up
            self.labeled = True # Assume labeled if one_batch fails, can be an issue.

    def after_train(self):
        if hasattr(self, 'old_lf'): # Ensure before_train was called
            self.learn.loss_func = self.old_lf

    def before_batch(self):
        # Reset state for the current batch
        self.current_mix_type = None
        self.lam_for_loss_blending = None # Lambda used for blending losses
        self.yb_shuffled_for_loss = None  # Shuffled targets for the loss function

        if not self.training or not self.labeled or random.random() > self.p_apply_any:
            return # Skip augmentation for this batch

        # Common setup: shuffle batch for mixing
        # self.x and self.y are populated by fastai from self.learn.xb[0] and self.learn.yb[0]
        current_x_batch = self.x 
        current_y_batch = self.y

        shuffle_indices = torch.randperm(current_x_batch.size(0)).to(current_x_batch.device)
        xb_shuffled = current_x_batch[shuffle_indices]
        if self.labeled:
            # Store y_shuffled as a tuple, consistent with how fastai handles yb
            self.yb_shuffled_for_loss = tuple((current_y_batch[shuffle_indices],))

        if random.random() < self.p_cutmix_if_apply:
            # --- Apply CutMix1d Logic ---
            self.current_mix_type = 'cutmix'
            _bs, *_, seq_len = current_x_batch.size()

            # Lambda for determining bbox size (from beta distribution)
            lam_bbox = self.cutmix_distrib.sample((1,)).to(current_x_batch.device)

            # rand_bbox logic (simplified from CutMix1d)
            cut_ratio = (1. - lam_bbox).item() # Proportion to cut
            cut_seq_len = int(seq_len * cut_ratio)

            if cut_seq_len == 0 : # If patch size is zero, effectively no cutmix
                self.current_mix_type = None 
                return

            cx = random.randint(0, seq_len -1) # Center of the patch
            x1 = np.clip(cx - cut_seq_len // 2, 0, seq_len)
            x2 = np.clip(cx + (cut_seq_len - cut_seq_len // 2), 0, seq_len) # Ensure full cut_seq_len

            if x1 >= x2 : # If patch width is zero or invalid
                self.current_mix_type = None
                return
            
            # Create a new tensor for augmented x
            new_xb = current_x_batch.clone()
            new_xb[..., x1:x2] = xb_shuffled[..., x1:x2]
            self.learn.xb = tuple((new_xb,)) # Update learner's current input batch

            # Lambda for blending targets/losses: proportion of the sequence from the shuffled batch
            self.lam_for_loss_blending = float(x2 - x1) / seq_len
        else:
            # --- Apply MixUp1d Logic ---
            self.current_mix_type = 'mixup'
            
            # Lambda for mixing (per sample, from beta distribution)
            # Ensure lambda is >= 0.5 for convention, or use raw samples if preferred
            lam_samples = self.mixup_distrib.sample((current_x_batch.size(0),)).to(current_x_batch.device)
            self.lam_for_loss_blending = torch.max(lam_samples, 1. - lam_samples)
            
            # Apply MixUp to x
            # unsqueeze lam_for_loss_blending to match dimensions of x for broadcasting
            weight_for_lerp = unsqueeze(self.lam_for_loss_blending, n=current_x_batch.ndim - 1)
            mixed_x = torch.lerp(current_x_batch, xb_shuffled, weight_for_lerp)
            self.learn.xb = tuple((mixed_x,))

    def _mixed_loss_func(self, pred, *yb_original):
        # yb_original is the y for the current (original, pre-shuffled) batch
        if self.current_mix_type is None or self.lam_for_loss_blending is None or not self.training:
            return self.old_lf(pred, *yb_original)

        # Use NoneReduce to get per-item losses before reduction
        with NoneReduce(self.old_lf) as loss_func_no_reduce:
            loss_original_targets = loss_func_no_reduce(pred, *yb_original)
            loss_shuffled_targets = loss_func_no_reduce(pred, *self.yb_shuffled_for_loss)
        
        # Blend losses: loss = (1-lam)*loss_orig + lam*loss_shuffled
        # torch.lerp(input, end, weight) = input * (1-weight) + end * weight
        # So, we want lerp(loss_original_targets, loss_shuffled_targets, self.lam_for_loss_blending)
        mixed_loss_per_item = torch.lerp(loss_original_targets, loss_shuffled_targets, self.lam_for_loss_blending)
        
        # Reduce the mixed loss (e.g., mean)
        return _reduce_loss(mixed_loss_per_item, getattr(self.old_lf, 'reduction', 'mean'))

