import numpy as np
import torch

def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip

class NormalizeVideo(object):
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(
            self.mean, self.std, self.inplace)

apply_norm = NormalizeVideo(mean=(128,128,128), std=(128,128,128))

def to_norm_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    # return clip.float().permute(3, 0, 1, 2) / 255.0 # range [0, 1]
    return apply_norm(clip.float().permute(3, 0, 1, 2)) # range [-1, 1]

def video_transform(videos):
    '''
    videos: list of PIL videos
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (B, T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (B, C, T, H, W)
    '''
    inputs = []
    for vid in videos:
        inputs.append(to_norm_tensor(vid))
    inputs = torch.stack(inputs, dim=0).float().to('cuda' if torch.cuda.is_available() else 'cpu')
    return inputs

def collate(batch):
    '''
    custom Collat funciton for collating individual fetched data samples into batches.
    '''
    frames = [b['frames'] for b in batch]
    
    body = [b['body'] for b in batch]
    face = [b['face'] for b in batch]
    rh = [b['rh'] for b in batch]
    lh = [b['lh'] for b in batch]
    
    ecg = [b['ecg'] for b in batch]

    sub_lbls = [b['sub_lbls'] for b in batch]
    super_lbls = [b['super_lbls'] for b in batch]    

    filename = [b['filename'] for b in batch]
    return {'frames': frames, 'body': body, 'face': face,
            'rh': rh, 'lh': lh, 'ecg': ecg, 'sub_lbls': sub_lbls,
            'super_lbls': super_lbls, 'filename': filename}

def generate_soft_labels(sample, num_classes):
    label = np.zeros(num_classes)
    for element in sample:
        label[element] += 1
    label /= len(sample)  # Normalize to get proportions
    return label

def get_of_indices(seconds_to_sample):
    
    max_samples_per_interval = 50
    frames_per_second = 30
    
    frames_per_second_to_sample = int(np.ceil(max_samples_per_interval/seconds_to_sample))
    
    # Generate random indices for frame sampling
    sampled_indices = []
    for second in range(1, seconds_to_sample + 1):
        start_frame = int((second - 1) * frames_per_second)
        end_frame = int(second * frames_per_second)
        sampled_indices.extend(np.random.choice(range(start_frame, end_frame),
                                                frames_per_second_to_sample,
                                                replace=False))
    # this 48 is because slowfast network's input
    x = np.random.choice(sampled_indices, 48, replace=False) 
    x.sort()
    return x

def get_pose_indices(seconds_to_sample):
    
    max_samples_per_interval = 150
    frames_per_second_to_sample = int(np.ceil(max_samples_per_interval/seconds_to_sample))
    
    frames_per_second = 30
    
    # Generate random indices for frame sampling
    sampled_indices = []
    for second in range(1, seconds_to_sample + 1):
        start_frame = int((second - 1) * frames_per_second)
        end_frame = int(second * frames_per_second)
        sampled_indices.extend(np.random.choice(range(start_frame, end_frame),
                                                frames_per_second_to_sample,
                                                replace=False))
    x = np.random.choice(sampled_indices, max_samples_per_interval, replace=False) 
    x.sort()
    return x

# x = get_of_indices(10)
# y = get_pose_indices(10)

def values_fromreport(report):
    p = report['weighted avg']['precision']
    r = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    return p,r, f1

