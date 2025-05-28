import cv2
import numpy as np

def enhance_frame(frame):
    # any preprocessing can be done here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def calculate_tvl1_optical_flow(prev_img, next_img):
   
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(prev_img, next_img, None)
    
    flow_image = flow_to_color(flow)
    return flow_image

def flow_to_color(flow):
    
    # magnitude and angle of 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255                      # Saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value

    # HSV to BGR
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

        flow_image = calculate_tvl1_optical_flow(prev_gray, gray)
        
        prev_gray = gray

        out.write(flow_image)

        # cv2.imshow('Optical Flow', flow_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

input_video_path = '../data/TP001.mpg'
output_tvl1 = '../data/TP001-flow-tvl1.mp4'


process_video(input_video_path, output_tvl1)


