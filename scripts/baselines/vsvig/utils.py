JOINT_PAIRS_MAP_ALL = {(0, 15): {'joint_names': ('Nose', 'REye')},
                       (0, 16): {'joint_names': ('Nose', 'LEye')},
                       (1, 0): {'joint_names': ('Neck', 'Nose')},
                       (1, 2): {'joint_names': ('Neck', 'RShoulder')},
                       (1, 5): {'joint_names': ('Neck', 'LShoulder')},
                       (1, 8): {'joint_names': ('Neck', 'MidHip')},
                       (2, 3): {'joint_names': ('RShoulder', 'RElbow')},
                       # (2, 17): {'joint_names': ('RShoulder', 'REar')},
                       (3, 4): {'joint_names': ('RElbow', 'RWrist')},
                       (5, 6): {'joint_names': ('LShoulder', 'LElbow')},
                       # (5, 18): {'joint_names': ('LShoulder', 'LEar')},
                       (6, 7): {'joint_names': ('LElbow', 'LWrist')},
                       (8, 9): {'joint_names': ('MidHip', 'RHip')},
                       (8, 12): {'joint_names': ('MidHip', 'LHip')},
                       (9, 10): {'joint_names': ('RHip', 'RKnee')},
                       (10, 11): {'joint_names': ('RKnee', 'RAnkle')},
                       (11, 22): {'joint_names': ('RAnkle', 'RBigToe')},
                       (11, 24): {'joint_names': ('RAnkle', 'RHeel')},
                       (12, 13): {'joint_names': ('LHip', 'LKnee')},
                       (13, 14): {'joint_names': ('LKnee', 'LAnkle')},
                       (14, 19): {'joint_names': ('LAnkle', 'LBigToe')},
                       (14, 21): {'joint_names': ('LAnkle', 'LHeel')},
                       (15, 17): {'joint_names': ('REye', 'REar')},
                       (16, 18): {'joint_names': ('LEye', 'LEar')},
                       (19, 20): {'joint_names': ('LBigToe', 'LSmallToe')},
                       (22, 23): {'joint_names': ('RBigToe', 'RSmallToe')}}

HAND_JOINT_PAIRS_MAP_ALL = {
    (0, 1): {'joint_names': ('Wrist', 'Thumb'), 'color': (255, 0, 0)},
    (1, 2): {'joint_names': ('Thumb', 'ThumbKnuckle'), 'color': (255, 0, 0)},
    (2, 3): {'joint_names': ('ThumbKnuckle', 'ThumbJoint'), 'color': (255, 0, 0)},
    (3, 4): {'joint_names': ('ThumbJoint', 'ThumbTip'), 'color': (255, 0, 0)},
    
    (0, 5): {'joint_names': ('Wrist', 'IndexFinger'), 'color': (255, 255, 0)},
    (5, 6): {'joint_names': ('IndexFinger', 'IndexFingerKnuckle'), 'color': (255, 255, 0)},
    (6, 7): {'joint_names': ('IndexFingerKnuckle', 'IndexFingerJoint'), 'color': (255, 255, 0)},
    (7, 8): {'joint_names': ('IndexFingerJoint', 'IndexFingerTip'), 'color': (255, 255, 0)},
    
    (0, 9): {'joint_names': ('Wrist', 'MiddleFinger'), 'color': (0, 255, 128)},
    (9, 10): {'joint_names': ('MiddleFinger', 'MiddleFingerKnuckle'), 'color': (0, 255, 128)},
    (10, 11): {'joint_names': ('MiddleFingerKnuckle', 'MiddleFingerJoint'), 'color': (0, 255, 128)},
    (11, 12): {'joint_names': ('MiddleFingerJoint', 'MiddleFingerTip'), 'color': (0, 255, 128)},
    
    (0, 13): {'joint_names': ('Wrist', 'RingFinger'), 'color': (128, 255, 0)},
    (13, 14): {'joint_names': ('RingFinger', 'RingFingerKnuckle'), 'color': (128, 255, 0)},
    (14, 15): {'joint_names': ('RingFingerKnuckle', 'RingFingerJoint'), 'color': (128, 255, 0)},
    (15, 16): {'joint_names': ('RingFingerJoint', 'RingFingerTip'), 'color': (128, 255, 0)},
    
    (0, 17): {'joint_names': ('Wrist', 'PinkyFinger'), 'color': (128, 128, 0)},
    (17, 18): {'joint_names': ('PinkyFinger', 'PinkyFingerKnuckle'), 'color': (128, 128, 0)},
    (18, 19): {'joint_names': ('PinkyFingerKnuckle', 'PinkyFingerJoint'), 'color': (128, 128, 0)},
    (19, 20): {'joint_names': ('PinkyFingerJoint', 'PinkyFingerTip'), 'color': (128, 128, 0)},
    
}

FACE_JOINT_PAIRS_MAP_ALL = {
    # Nameing is not correct here
    (0, 1): {'joint_names': ('REye', 'REyeCorner'), 'color': (255, 0, 0)},
    (1, 2): {'joint_names': ('REyeCorner', 'REyebrow'), 'color': (0, 255, 0)},
    (2, 3): {'joint_names': ('REyebrow', 'REyebrowCorner'), 'color': (0, 0, 255)},
    (3, 4): {'joint_names': ('REyebrowCorner', 'LEyebrowCorner'), 'color': (255, 255, 0)},
    (4, 5): {'joint_names': ('LEyebrowCorner', 'LEyebrow'), 'color': (255, 0, 255)},
    (5, 6): {'joint_names': ('LEyebrow', 'LEyeCorner'), 'color': (0, 255, 255)},
    (6, 7): {'joint_names': ('LEyeCorner', 'LEye'), 'color': (0, 255, 0)},
    (7, 8): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (8, 9): {'joint_names': ('REye', 'REyeCorner'), 'color': (255, 0, 0)},
    (9, 10): {'joint_names': ('REyeCorner', 'REyebrow'), 'color': (0, 255, 0)},
    (10, 11): {'joint_names': ('REyebrow', 'REyebrowCorner'), 'color': (0, 0, 255)},
    (11, 12): {'joint_names': ('REyebrowCorner', 'LEyebrowCorner'), 'color': (255, 255, 0)},
    (12, 13): {'joint_names': ('LEyebrowCorner', 'LEyebrow'), 'color': (255, 0, 255)},
    (13, 14): {'joint_names': ('LEyebrow', 'LEyeCorner'), 'color': (0, 255, 255)},
    (14, 15): {'joint_names': ('LEyeCorner', 'LEye'), 'color': (0, 255, 0)},
    (15, 16): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    
    # right eye brow
    (17, 18): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (18, 19): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (19, 20): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (20, 21): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    # left eye brow
    (22, 23): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (23, 24): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (24, 25): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (25, 26): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    # nose center line
    (27, 28): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (28, 29): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (29, 30): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},

    # nose nostrils line
    (31, 32): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (32, 33): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (33, 34): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    (34, 35): {'joint_names': ('LEye', 'LEar'), 'color': (0, 0, 255)},
    
    
    
    
    
}

# Fun joint colors (corrected)
joint_colors = {
    'Nose': (255, 0, 0),     # Red
    'REye': (0, 255, 0),     # Green
    'LEye': (0, 0, 255),     # Blue
    'RShoulder': (255, 255, 0),  # Yellow
    'LShoulder': (255, 0, 255),  # Magenta
    'Neck': (0, 255, 255),   # Cyan
    'MidHip': (128, 128, 0), # Olive
    'RElbow': (128, 0, 128), # Purple
    'RWrist': (0, 128, 128), # Teal
    'LElbow': (255, 128, 0), # Orange
    'LWrist': (0, 255, 128), # Lime
    'RHip': (128, 0, 0),     # Maroon
    'LHip': (0, 128, 0),     # Green
    'RKnee': (0, 0, 128),    # Navy
    'LKnee': (128, 128, 128),  # Gray
    'RAnkle': (192, 192, 192),  # Silver
    'LAnkle': (255, 165, 0),   # Orange (different shade)
    'REar': (0, 255, 165),   # Cyan (different shade)
    'LEar': (165, 0, 255),   # Purple (different shade)
    'RBigToe': (0, 165, 255),  # Blue (different shade)
    'RSmallToe': (255, 0, 165), # Magenta (different shade)
    'LBigToe': (255, 100, 100), # Light red
    'LSmallToe': (100, 255, 100),  # Light green
    'RHeel': (100, 100, 255),  # Light blue
    'LHeel': (150, 0, 0),     # Dark maroon
    'RSide': (0, 150, 0),     # Dark green
    'LSide': (0, 0, 150)      # Dark blue
    # Add more joints and their colors here
}
