'''
https://github.com/Mr-TalhaIlyas/OpenCV-with-CUDA-support
https://github.com/Mr-TalhaIlyas/OpenCV-CUDA-Docker
https://docs.opencv.org/3.4.1/d7/d18/classcv_1_1cuda_1_1BroxOpticalFlow.html
'''

import cv2
import numpy as np

def enhance_frame(frame):
    # any preprocessing can be done here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def calculate_brox_optical_flow(prev_img, next_img):
    """
    Calculates optical flow using the Brox algorithm.
    Note: Requires OpenCV with CUDA support.
    """
    
    prev_img = np.float32(prev_img) / 255.0
    next_img = np.float32(next_img) / 255.0

    
    gpu_prev = cv2.cuda_GpuMat()
    gpu_next = cv2.cuda_GpuMat()
    gpu_prev.upload(prev_img)
    gpu_next.upload(next_img)

    
    brox = cv2.cuda_BroxOpticalFlow_create(
        alpha=0.197,
        gamma=50.0,
        scale_factor=0.8,
        inner_iterations=5,
        outer_iterations=150,
        solver_iterations=10
    )

    flow = brox.calc(gpu_prev, gpu_next, None)

    flow_x = flow[:, :, 0].download()
    flow_y = flow[:, :, 1].download()
    flow = np.dstack((flow_x, flow_y))

    flow_image = flow_to_color(flow)
    return flow_image

def flow_to_color(flow):
    
    # magnitude and angle of 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255                      # Saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def process_video(input_video_path, output_video_path):
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return

    prev_frame = cv2.resize(prev_frame, (0, 0), fx=0.5, fy=0.5)
    prev_gray = enhance_frame(prev_frame)

    frame_width = int(prev_gray.shape[1])
    frame_height = int(prev_gray.shape[0])
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = enhance_frame(frame)

        flow_image = calculate_brox_optical_flow(prev_gray, gray)
        
        prev_gray = gray

        out.write(flow_image)

        # cv2.imshow('Optical Flow', flow_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

in_vid = '../data/TP001.mpg'
output_brox = '../data/TP001-flow-brox.mp4'


process_video(in_vid, output_brox)


