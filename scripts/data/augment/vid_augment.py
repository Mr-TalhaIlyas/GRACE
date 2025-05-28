import random
import numpy as np
import vidaug.augmentors as va

# ─── HELPERS ────────────────────────────────────────────────────────────────────
# wrap an augmentor so it only runs with prob q
sometimes = lambda aug, q=0.7: va.Sometimes(q, aug)
# pick exactly one from a list
oneof    = lambda augs: va.OneOf(augs)
# chain multiple augmentors in order
seq      = lambda augs: va.Sequential(augs)

# ─── DEFINE SINGLE-OP AUGMENTORS ───────────────────────────────────────────────
AUG_FLIP_H    = va.HorizontalFlip()      # mirror horizontally
AUG_FLIP_V    = va.VerticalFlip()        # mirror vertically
AUG_PEPPER    = va.Pepper(ratio=100)        # pepper noise
AUG_SALT      = va.Salt(ratio=100)          # salt noise
AUG_MUL       = va.Multiply(value=0.6)   # darken/brighten
AUG_ROTATE    = va.RandomRotate(degrees=30)
AUG_ADD       = va.Add(value=50)         # brighten by adding
AUG_TRANSLATE = va.RandomTranslate(x=30, y=30)
AUG_INVERT    = va.InvertColor()         # invert colors

# ─── DEFINE SMALL SEQUENCES ───────────────────────────────────────────────────
SEQ_FLIP_ROTATE = seq([
    sometimes(AUG_FLIP_H, q=0.5),
    sometimes(AUG_ROTATE, q=0.5),
])

SEQ_VERTICAL_MUL = seq([
    sometimes(AUG_FLIP_V, q=0.5),
    sometimes(AUG_MUL,    q=0.5),
])

SEQ_TRANSLATE_ADD = seq([
    sometimes(AUG_TRANSLATE, q=0.5),
    sometimes(AUG_ADD,       q=0.5),
])

# ─── MASTER LIST ───────────────────────────────────────────────────────────────
VIDEO_AUGMENTORS = [
    AUG_FLIP_H,
    AUG_FLIP_V,
    AUG_INVERT,
    AUG_PEPPER,
    AUG_SALT,
    AUG_MUL,
    AUG_ROTATE,
    AUG_ADD,
    AUG_TRANSLATE,
    SEQ_FLIP_ROTATE,
    SEQ_VERTICAL_MUL,
    SEQ_TRANSLATE_ADD,
]

# ─── AUGMENTATION FUNCTION ─────────────────────────────────────────────────────
def apply_video_augmentation(frames: np.ndarray, p: float = 0.7) -> np.ndarray:
    """
    frames: np.ndarray of shape (T, H, W, C) or (H, W, C) with dtype uint8, values [0,255]
    p: probability of applying one augmentation
    """
    orig_dtype = frames.dtype
    apply = random.random() < p
    if apply:
        aug = random.choice(VIDEO_AUGMENTORS)
        out = aug(frames)
    else:
        out = frames
    out = np.array(out)
    # --- RANGE CHECK & RESTORE ---
    # If the output is float and seems normalized to [0,1], rescale back to [0,255].
    if np.issubdtype(out.dtype, np.floating):
        vmin, vmax = float(out.min()), float(out.max())
        if 0.0 <= vmin and vmax <= 1.0:
            # scale back
            out = (out * 255.0).clip(0,255).astype(orig_dtype)
        else:
            # if floats but already in [0,255], just clip and cast
            out = out.clip(0,255).astype(orig_dtype)
    else:
        # if integer type, ensure clipping
        out = out.clip(0,255).astype(orig_dtype)

    return out