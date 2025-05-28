#%%
import numpy as np
import scipy.signal
import scipy.stats
from scipy.fft import fft
from tqdm import trange
from fmutils import fmutils as fmu
from pathlib import Path
import glob

from utils import compute_kinematic_features, compute_distance_features, compute_spectral_features

ouput_dir = "/home/user01/Data/mme/dataset/baseline_feats/david/pose/"
all_kpts = glob.glob("/home/user01/Data/mme/dataset/pose/*/body.npy")

# Load your keypoints data
# keypoints_data = np.load("/home/user01/Data/mme/dataset/pose/a_patient_1/body.npy")  # Shape: (3, T, 25, 1)
for i in range(len(all_kpts)):
    fname = str(Path(all_kpts[i]).parent).split('/')[-1]
    keypoints_data = np.load(all_kpts[i])  # Shape: (3, T, 25, 1)

    CONF_THRESHOLD = 0.0 # threshold on openpose confidence for coordinates.
    # Indices of the keypoints to use
    keypoint_indices = [0, 1, 2, 5, 3, 6, 4, 7]

    # Extract the keypoints
    keypoints = keypoints_data[:, :, keypoint_indices, 0]  # Shape: (3, T, 8)

    T = keypoints.shape[1]  # Total number of frames
    sequence_length = 50 # Number of frames per sequence e.g. @5 fps 50 sec ~ (10 seconds)

    feature_vectors = []

    for start_idx in trange(0, T - sequence_length + 1):
        end_idx = start_idx + sequence_length
        sequence_features = []
        
        # Extract the sequence of frames
        keypoints_seq = keypoints[:, start_idx:end_idx, :]  # Shape: (3, 5, 8)
        
        # Iterate over each keypoint
        for kp_idx in range(keypoints_seq.shape[2]):
            x_coords = keypoints_seq[0, :, kp_idx]
            y_coords = keypoints_seq[1, :, kp_idx]
            confs = keypoints_seq[2, :, kp_idx]
            
            # Check for valid keypoints (confidence threshold)
            if np.any(confs < CONF_THRESHOLD):  # Adjust threshold as needed
                # If keypoint is not reliably detected, fill with zeros
                kinematic_features = [0.0] * 17
                spectral_features = [0.0] * 8
            else:
                # Compute displacement trajectories
                trajectory_x = x_coords
                trajectory_y = y_coords
                
                # Combine X and Y for spectral features
                trajectory = np.sqrt(trajectory_x**2 + trajectory_y**2)
                
                # Compute kinematic features
                kinematic_features_x = compute_kinematic_features(trajectory_x)
                kinematic_features_y = compute_kinematic_features(trajectory_y)
                
                # Average the features from X and Y
                kinematic_features = [(kx + ky) / 2.0 for kx, ky in zip(kinematic_features_x, kinematic_features_y)]
                
                # Compute distance features
                distance_features = compute_distance_features(trajectory_x, trajectory_y)
                kinematic_features.extend(distance_features)  # Total of 17 features
                
                # Compute spectral features for displacement and velocity
                # Displacement signal
                spectral_features_disp = compute_spectral_features(trajectory)
                # Velocity signal
                velocity = np.gradient(trajectory)
                spectral_features_vel = compute_spectral_features(velocity)
                # Combine spectral features
                spectral_features = spectral_features_disp + spectral_features_vel  # Total of 8 features
            
            # Combine kinematic and spectral features
            keypoint_features = kinematic_features + spectral_features  # Total of 25 features per keypoint
            sequence_features.extend(keypoint_features)
        
        # list of 8 * 25 = 200 features
        feature_vectors.append(sequence_features)

    # convert to np array
    feature_vectors = np.array(feature_vectors)  # Shape: (num_sequences, 200)

    # replace NaNs with zeros
    feature_vectors = np.nan_to_num(feature_vectors)
    
    np.save(ouput_dir + fname + ".npy", feature_vectors)
    
    print("Feature vectors shape:", feature_vectors.shape)
#%%