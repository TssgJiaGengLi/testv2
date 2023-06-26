import streamlit as st
import cv2
import torch
from utils.hubconf import custom
import numpy as np
from streamlit_webrtc import WebRtcMode,webrtc_streamer, VideoTransformerBase
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from model_utils import get_yolo, color_picker_fn, get_system_stat
from trun import get_ice_servers
# from ultralytics import YOLO
new_title = '<p style="font-family:sans-serif; color:#e20074; font-size: 40px;">T-Video Analytics</p>'
st.markdown(new_title, unsafe_allow_html=True)

p_time = 0

st.sidebar.title('Settings')
# Choose the model
model_type = st.sidebar.selectbox(
    'Choose YOLO Model', ('YOLO Model', 'YOLOv8', 'YOLOv7')
)

sample_img = cv2.imread('logo.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
cap = None

if not model_type == 'YOLO Model':
    path_model_file = st.sidebar.text_input(
        f'path to {model_type} Model:',
        f'yolov8.pt'
    )
    if st.sidebar.checkbox('Load Model'):
        
        # YOLOv7 Model
        if model_type == 'YOLOv7':
            # GPU
            gpu_option = st.sidebar.radio(
                'PU Options:', ('CPU', 'GPU'))

            if not torch.cuda.is_available():
                st.sidebar.warning('CUDA Not Available, So choose CPU', icon="⚠️")
            else:
                st.sidebar.success(
                    'GPU is Available on this Device, Choose GPU for the best performance',
                    icon="✅"
                )
            # Model
            if gpu_option == 'CPU':
                model = custom(path_or_model=path_model_file)
            if gpu_option == 'GPU':
                model = custom(path_or_model=path_model_file, gpu=True)

        # YOLOv8 Model
        if model_type == 'YOLOv8':
            from ultralytics import YOLO
            model = YOLO(path_model_file)

        # Load Class names
        class_labels = model.names

        # Inference Mode
        options = st.sidebar.radio(
            'Options:', ('Webcam','Video', 'RTSP'), index=1)

        # Confidence
        confidence = st.sidebar.slider(
            'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

        # Draw thickness
        draw_thick = st.sidebar.slider(
            'Draw Thickness:', min_value=1,
            max_value=20, value=3
        )
        
        color_pick_list = []
        for i in range(len(class_labels)):
            color = color_picker_fn()
            color_pick_list.append(color)

        # Image
       
        
        # Video
        if options == 'Video':
            upload_video_file = st.sidebar.file_uploader(
                'Upload Video', type=['mp4', 'avi', 'mkv'])
            if upload_video_file is not None:
                pred = st.checkbox(f'Predict Using {model_type}')

                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(upload_video_file.read())
                cap = cv2.VideoCapture(tfile.name)
                # if pred:


        # Web-cam
        if options == 'Webcam':
            #cam_options = st.sidebar.selectbox('Webcam Channel',
            #                               ('Select Channel', '0', '1', '2', '3'))
            class VideoTransformer(VideoTransformerBase):
                def __init__(self):
                    self.model_type = 'YOLOv8'
                    self.model = YOLO(path_model_file)
                    self.confidence = confidence
                    self.color_pick_list = color_pick_list
                    self.class_labels = class_labels
                    self.draw_thick = draw_thick
                    self.p_time = time.time()
                    self.fps = 0
                    self.class_fq = {}

                def transform(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img, current_no_class = get_yolo(img, self.model_type, self.model, self.confidence, self.color_pick_list, self.class_labels, self.draw_thick)
                    # FPS
                    c_time = time.time()
                    self.fps = 1 / (c_time - self.p_time)
                    self.p_time = c_time

                    # Current number of classes
                    class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                    class_fq = json.dumps(class_fq, indent=4)
                    class_fq = json.loads(class_fq)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                    self.class_fq = df_fq


                    return img
        
            #if not cam_options == 'Select Channel':
            ctx = webrtc_streamer(
                key="example", 
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": get_ice_servers()},
                video_transformer_factory=VideoTransformer,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                )
            if ctx.video_transformer:
                stframe1 = st.empty()
                stframe2 = st.empty()
                stframe3 = st.empty()

                while True:
                    # Updating Inference results
                    get_system_stat(stframe1, stframe2, stframe3, ctx.video_transformer.fps, ctx.video_transformer.class_fq)


        # RTSP
        if options == 'RTSP':
            rtsp_url = st.sidebar.text_input(
                'RTSP URL:',
                'eg: rtsp://admin:name6666@198.162.1.58/cam/realmonitor?channel=0&subtype=0'
            )
            pred = st.checkbox(f'Predict Using {model_type}')
            cap = cv2.VideoCapture(rtsp_url)


if (cap != None) and pred:
    stframe1 = st.empty()
    stframe2 = st.empty()
    stframe3 = st.empty()
    while True:
        success, img = cap.read()
        if not success:
            st.error(
                f"{options} NOT working\nCheck {options} properly!!",
                icon="🚨"
            )
            break

        img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
        FRAME_WINDOW.image(img, channels='BGR')

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        
        # Current number of classes
        class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
        class_fq = json.dumps(class_fq, indent = 4)
        class_fq = json.loads(class_fq)
        df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
        
        # Updating Inference results
        get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)