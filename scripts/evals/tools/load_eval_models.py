import os
import torch
from tsai.models.InceptionTime import InceptionTime

def load_bio_signal_models(
    config: dict,
    scope: str = 'external',
    secondary_dataset: str = None,
    device: torch.device = torch.device('cpu'),
    c_out: int = 2,
    fc_dropout: float = 0.5,
    nf: int = 64
):
    """
    Load InceptionTime models for EEG, ECG+HRV, and ECG modalities,
    for either external validation (on a secondary dataset) or
    internal (Alfred) validation.

    Args:
        config: must contain 'bio_signal_chkpts_dir' (str)
        scope: 'external' or 'alfred'
        secondary_dataset: if scope=='external', one of {'tuh','seizeit2'}
        device: torch.device for model & weights
        c_out: number of output classes
        fc_dropout: dropout in final FC
        nf: base filter count for InceptionTime

    Returns:
        dict with keys 'eeg', 'ecgH', 'ecg' mapping to loaded nn.Modules
    """
    chkpt_base = config['bio_signal_chkpts_dir']
    # Build checkpoint filenames
    if scope == 'external':
        if secondary_dataset not in ('tuh', 'seizeIT2'):
            raise ValueError("secondary_dataset must be 'tuh' or 'seizeIT2' for external scope")
        suffix = secondary_dataset
        ckpt_map = {
            'eeg':  f"best_{suffix}_eeg_bvg_InceptionTime.pth",
            'ecgH': f"best_{suffix}_ecgH_bvg_InceptionTime.pth",
            'ecg':  f"best_{suffix}_ecg_bvg_InceptionTime.pth",
        }
    elif scope == 'alfred':
        ckpt_map = {
            'eeg':  "best_eeg_bvg_InceptionTime.pth",
            'ecgH': "best_ecgH_bvg_InceptionTime.pth",
            'ecg':  "best_ecg_bvg_InceptionTime.pth",
        }
    else:
        raise ValueError("scope must be 'external' or 'alfred'")

    # Decide input channels for EEG
    if scope == 'external':
        eeg_cin = 19 if secondary_dataset == 'tuh' else 2
    else:  # alfred
        eeg_cin = config.get('alfred_eeg_cin', 19)

    # Default channel counts for ECG+HRV and ECG
    ecgH_cin = 19 # config.get('ecgH_cin', 19)
    ecg_cin  = 1  # config.get('ecg_cin', 1)

    # Instantiate models
    models = {}
    models['eeg'] = InceptionTime(
        c_in=eeg_cin, c_out=c_out, fc_dropout=fc_dropout, nf=nf, return_features=False
    )
    models['ecgH'] = InceptionTime(
        c_in=ecgH_cin, c_out=c_out, fc_dropout=fc_dropout, nf=nf, return_features=False
    )
    models['ecg'] = InceptionTime(
        c_in=ecg_cin, c_out=c_out, fc_dropout=fc_dropout, nf=nf, return_features=False
    )

    # Load checkpoint weights
    for name, model in models.items():
        ckpt_path = os.path.join(chkpt_base, ckpt_map[name])
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

    return models
