import numpy as np
import random

def random_affine_pose(
    data_numpy_list,
    angle_range=10.0,
    scale_range=0.1,
    trans_range=0.1
):
    """
    Apply a single global affine transform (rotation, scaling, translation)
    identically to all subgraphs (body, face, hands) to preserve topology.
    
    data_numpy_list: list of np.ndarray, each shape (C, T, V, M)
      C >= 2 (we use first two channels as X,Y), T frames, V keypoints, M persons (here M=1).
    angle_range: max rotation angle in degrees.
    scale_range: relative scale variation (e.g. 0.1 ➞ scale in [0.9,1.1]).
    trans_range: normalized translation range (fraction of coordinate frame), e.g. 0.1 ➞ ±10%.
    """
    # Sample one set of transform parameters
    theta_deg = random.uniform(-angle_range, angle_range)
    theta = np.deg2rad(theta_deg)
    scale = random.uniform(1-scale_range, 1+scale_range)
    tx = random.uniform(-trans_range, trans_range)
    ty = random.uniform(-trans_range, trans_range)
    
    # Build 2x2 rotation+scale matrix
    cos_t = np.cos(theta) * scale
    sin_t = np.sin(theta) * scale
    affine_mat = np.array([[cos_t, -sin_t],
                           [sin_t,  cos_t]])  # shape (2,2)
    
    augmented_list = []
    for data in data_numpy_list:
        C, T, V, M = data.shape
        assert C >= 2, "Need at least 2 channels for X,Y"
        data_aug = data.copy()
        for t in range(T):
            # Extract XY coords (2, V*M)
            xy = data[0:2, t, :, :].reshape(2, -1)
            # Apply affine
            new_xy = affine_mat @ xy
            # Apply translation
            new_xy[0, :] += tx
            new_xy[1, :] += ty
            # Reshape back to (2, V, M)
            new_xy = new_xy.reshape(2, V, M)
            data_aug[0:2, t, :, :] = new_xy
        augmented_list.append(data_aug)
    return augmented_list

def apply_pose_augmentation(
    data_numpy_list,
    p=0.7,
    angle_range=30.0,
    scale_range=0.4,
    trans_range=0.4
):
    """
    With probability p, apply random_affine_pose; else return input unchanged.
    """
    if random.random() < p:
        return random_affine_pose(
            data_numpy_list,
            angle_range=angle_range,
            scale_range=scale_range,
            trans_range=trans_range
        )
    else:
        return data_numpy_list
    
    
    