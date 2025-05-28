import numpy as np
from vidaug import augmentors as va
import random
from tsaug import Quantize, AddNoise, Dropout

sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
oneof = lambda aug: va.OneOf(aug)
someof = lambda aug: va.SomeOf(aug, 2)

seq = va.Sequential([
    oneof([
    sometimes(va.HorizontalFlip()),
    sometimes(va.VerticalFlip()),
    # sometimes(va.Pepper(ratio=100)),
    # sometimes(va.Salt(ratio=100)),
    # sometimes(va.Multiply(value=0.6)),
    # rotate
    # sometimes(va.RandomRotate(degrees=10)),
    # sometimes(va.Add()),
    sometimes(va.RandomTranslate(x=50, y=50))
    ])
])


def video_augment(video):
    video_aug = seq(video)
    return np.stack(video_aug, axis=0)

#%%
def random_move_multiple(data_numpy_list, # list of numpy arrays pose
                angle_candidate=[-10., -5., 0., 5., 10.], # random rotat
                scale_candidate=[0.9, 1.0, 1.1],          # random scale
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # Assume all arrays have the same T dimension
    T = data_numpy_list[0].shape[1]
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # Generate the transformation parameters for the entire sequence
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s], [np.sin(a) * s, np.cos(a) * s]])

    # Apply the transformation to each pose array in the list
    for data_numpy in data_numpy_list:
        C, T, V, M = data_numpy.shape
        for i_frame in range(T):
            xy = data_numpy[0:2, i_frame, :, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]
            data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy_list

# pose_augment funtion for applying random_move_multiple with a probability of 0.5, else return identity
def pose_augment(data_numpy_list):
    if random.random() > 0.7:
        # print('pose_augment')
        return random_move_multiple(data_numpy_list)
    else:
        return data_numpy_list

#%%
# apply 1 of the 4 augmentations to ecg signal randomly with equal probability
noise = AddNoise(scale=(0.01, 0.05)) @ 0.3  # with 50% probability, add random noise up to 1% - 5%
quantize = Quantize(n_levels=[10, 20, 30]) @ 0.3  # with 50% probability, quantize to 10, 20, or 30 levels
drop = Dropout(
                p=0.1,
                fill=0,
                size=[1, 2, 25],
                # [int(0.001 * sampling_rate), int(0.01 * sampling_rate), int(0.1 * sampling_rate)]
            ) @ 0.3 # drop out 10% of the time points (dropped out units are 1 ms, 10 ms, or 100 ms) and fill the dropped out points with zeros
flip = lambda x: -x  # flip the signal

# randomly select 1 of the 4 augmentations
def ecg_augment(ecg):
    if random.random() < 0.25: # 25% probability of flipping the signal
        # flip signal with 50% probability
        # out = flip(ecg) if random.random() < 0.5 else ecg
        out = ecg
        # print('flip')
    else: # 75% probability of applying one of the 3 augmentations
        aug = random.choice([noise, quantize, drop])
        out = aug.augment(ecg)
        # print(aug)
    return out



def apply_frequency_masking(cwt_array, F=20, num_masks=1, replace_with_zero=True):
    """
    Apply frequency masking to a CWT array.
    
    Parameters:
    - cwt_array: numpy array, the CWT representation.
    - F: int, maximum width of the frequency mask.
    - num_masks: int, number of frequency masks to apply.
    - replace_with_zero: bool, whether to replace masked parts with zero.
    """
    cloned_array = np.copy(cwt_array)
    num_freq_channels = cwt_array.shape[0]
    
    for _ in range(num_masks):
        f = np.random.uniform(low=0, high=F)
        f0 = np.random.randint(0, num_freq_channels - f)
        if replace_with_zero:
            cloned_array[int(f0):int(f0+f), :] = 0
        else:
            cloned_array[int(f0):int(f0+f), :] = cloned_array.mean()
    
    return cloned_array

def apply_time_masking(cwt_array, T=80, num_masks=1, replace_with_zero=True):
    """
    Apply time masking to a CWT array.
    
    Parameters:
    - cwt_array: numpy array, the CWT representation.
    - T: int, maximum width of the time mask.
    - num_masks: int, number of time masks to apply.
    - replace_with_zero: bool, whether to replace masked parts with zero.
    """
    cloned_array = np.copy(cwt_array)
    num_time_steps = cwt_array.shape[1]
    
    for _ in range(num_masks):
        t = np.random.uniform(low=0, high=T)
        t0 = np.random.randint(0, num_time_steps - t)
        if replace_with_zero:
            cloned_array[:, int(t0):int(t0+t)] = 0
        else:
            cloned_array[:, int(t0):int(t0+t)] = cloned_array.mean()
    
    return cloned_array

time_freq_masking = lambda x: apply_time_masking(apply_frequency_masking(x))

# apply cwt augment 

def cwt_augment(cwt_array):
    if random.random() < 0.50: # 25% probability of flipping the signal
        # flip signal with 50% probability
        out = cwt_array
    else: # 75% probability of applying one of the 3 augmentations
        aug = random.choice([time_freq_masking, apply_time_masking, apply_frequency_masking])
        out = aug(cwt_array)
        # print(aug)
    return out