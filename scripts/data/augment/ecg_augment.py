import random
import numpy as np
import tsaug

def drop_random_channels(ecg: np.ndarray, n_drop_choices=(1, 3)) -> np.ndarray:
    """
    ecg: shape (C, T)
    n_drop_choices: tuple of channel‐counts to choose from; e.g. (1,5) means drop EXACTLY 1 or EXACTLY 5 channels.
    """
    ecg_aug = ecg.copy()
    C, _ = ecg_aug.shape

    # pick how many channels to drop: either 1 or 5
    n_drop = random.choice(n_drop_choices)
    n_drop = min(n_drop, C)

    # sample that many distinct channel indices
    drop_idxs = random.sample(range(C), n_drop)
    for ch in drop_idxs:
        ecg_aug[ch, :] = 0.0

    return ecg_aug


def drift_selected_channels(ecg: np.ndarray,
                            max_drift=0.3,
                            n_drift_points=3,
                            first_n=2) -> np.ndarray:
    """
    Apply tsaug.Drift ONLY to these channels:
      - the first `first_n` channels: 0,1, ecg_raw and ecg_clean
      - the last channel: C-1
      - the third‐last channel: C-3

    All other channels remain exactly as they were.
    """
    ecg_aug = ecg.copy()
    C, T = ecg_aug.shape

    # build list: [0,1,2,3, C-3, C-1], clipped to valid indices
    selected = list(range(first_n))# + [C-3, C-1] # not doing reduced performance
    # remove duplicates and out‐of‐bounds
    selected = sorted({i for i in selected if 0 <= i < C})

    # extract just those channels
    sub = ecg_aug[selected, :]                       # shape (len(selected), T)
    # drift them
    drifted = tsaug.Drift(max_drift=max_drift,
                          n_drift_points=n_drift_points).augment(sub)
    # put them back
    ecg_aug[selected, :] = drifted

    return ecg_aug


# Example usage inside your Dataset’s augment block:

AUGMENTATIONS = [
    lambda e: tsaug.AddNoise(scale=(0.01, 0.02)).augment(e),
    lambda e: drift_selected_channels(e, max_drift=0.3, n_drift_points=3),
    lambda e: tsaug.Dropout(p=(0.05, 0.15), size=(100, 150), fill=0.).augment(e),
    lambda e: tsaug.Dropout(p=(0.01, 0.02),   size=(1,   5),   fill=0.).augment(e),
    lambda e: tsaug.Pool(size=(1, 3)).augment(e),
    lambda e: drop_random_channels(e, n_drop_choices=(1, 3)),
]

def apply_hrv_augmentation(ecg, p=0.7):
    if random.random() < p:
        return random.choice(AUGMENTATIONS)(ecg)
    else:
        return ecg
