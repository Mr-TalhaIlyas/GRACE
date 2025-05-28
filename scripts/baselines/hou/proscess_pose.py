#%%
import numpy as np
import glob
from pathlib import Path
from fmutils import fmutils as fmu
# from tqdm import trange
def remap_body_keypoints(keypoints_data):
    """
    Remaps OpenPose body keypoints to a new format for all time steps.
    https://doi.org/10.1109/AVSS52988.2021.9663770.
    Parameters:
        keypoints_data (numpy.ndarray): Array of shape (3, T, 25, 1), where:
            - 3: Coordinates (x, y, confidence)
            - T: Number of time steps (frames)
            - 25: Number of body keypoints
            - 1: Number of persons

    Returns:
        remapped_keypoints (numpy.ndarray): Array of remapped keypoints with shape (3, T, num_remapped_keypoints, 1)
    """
    # Mapping from original indices to new indices
    mapping = {
        0: 0,
        1: 1,
        5: 3,
        2: 6,
        8: 2,
        12: 9,
        9: 10,
        3: 7,
        6: 4,
        4: 8,
        7: 5
    }

    # Indices to keep from the original keypoints
    indices_to_keep = list(mapping.keys())
    indices_to_keep.sort()  # Ensure they are in order

    # Extract the keypoints to keep
    keypoints_to_keep = keypoints_data[:, :, indices_to_keep, :]  # Shape: (3, T, len(indices_to_keep), 1)

    # Initialize array for remapped keypoints
    num_remapped_keypoints = max(mapping.values()) + 1
    T = keypoints_data.shape[1]
    remapped_keypoints = np.zeros((3, T, num_remapped_keypoints, 1))

    # Remap the keypoints
    for original_index, new_index in mapping.items():
        # Find the position of the original index in indices_to_keep
        idx = indices_to_keep.index(original_index)
        remapped_keypoints[:, :, new_index, :] = keypoints_to_keep[:, :, idx, :]

    return remapped_keypoints

def remap_face_keypoints(face_keypoints_data):
    """
    Removes the face outline keypoints from OpenPose face keypoints for all time steps.

    Parameters:
        face_keypoints_data (numpy.ndarray): Array of shape (3, T, 70, 1), where:
            - 3: Coordinates (x, y, confidence)
            - T: Number of time steps (frames)
            - 70: Number of face keypoints
            - 1: Number of persons

    Returns:
        remapped_keypoints (numpy.ndarray): Array of face keypoints without the outline, shape (3, T, num_keypoints, 1)
    """
    # Indices to remove (outline keypoints)
    indices_to_remove = list(range(0, 27))  # Keypoints 0 to 26 inclusive

    # Indices to keep
    total_keypoints = face_keypoints_data.shape[2]
    indices_to_keep = [i for i in range(total_keypoints) if i not in indices_to_remove]

    # Extract the keypoints to keep
    remapped_keypoints = face_keypoints_data[:, :, indices_to_keep, :]  # Shape: (3, T, num_keypoints, 1)

    return remapped_keypoints
#%%
# keypoints_data = np.load("/home/user01/Data/mme/dataset/pose/a_patient_1/body.npy") 
# # Remap body keypoints
# remapped_body = remap_body_keypoints(keypoints_data)
# print("Remapped Body Keypoints Shape:", remapped_body.shape)
#%%

ouput_dir = "/home/user01/Data/mme/dataset/baseline_feats/hou/pose/"

all_bodies = glob.glob("/home/user01/Data/mme/dataset/pose/*/body.npy")
all_faces = glob.glob("/home/user01/Data/mme/dataset/pose/*/face.npy")

all_bodies = sorted(all_bodies, key=fmu.numericalSort)
all_faces = sorted(all_faces, key=fmu.numericalSort)

assert len(all_bodies) == len(all_faces), "Number of body and face keypoints files do not match."

for i in range(len(all_bodies)):
    fname = str(Path(all_bodies[i]).parent).split('/')[-1]
    keypoints_body = np.load(all_bodies[i])  # Shape: (3, T, 25, 1)
    keypoints_face = np.load(all_faces[i])  # Shape: (3, T, 70, 1)
    
    # Remap body keypoints
    remapped_body = remap_body_keypoints(keypoints_body)
    print("Remapped Body Keypoints Shape:", remapped_body.shape)
    remapped_face = remap_face_keypoints(keypoints_face)
    print("Remapped Face Keypoints Shape:", remapped_face.shape)
    print('File:', fname)
    
    # Save the remapped keypoints
    np.save(ouput_dir + fname + '_body.npy', remapped_body)
    np.save(ouput_dir + fname + '_face.npy', remapped_face)
#%%