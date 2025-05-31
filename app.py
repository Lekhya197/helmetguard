import streamlit as st
from streamlit_webrtc import webrtc_streamer
import torch
import cv2
import av
import time
import threading
import numpy as np
import pandas as pd
import requests
import os
from queue import Queue

# Page configuration
st.set_page_config(
    page_title="HelmetGuard AI - YOLOv5",
    page_icon="🛡️",
    layout="wide"
)

# Load model with caching
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_model()

# Telegram Bot Config
TELEGRAM_BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
TELEGRAM_CHAT_ID = '6674142283'

def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Telegram alert failed: {e}")
        return False

# UI Layout
st.title("🛡️ HelmetGuard AI - YOLOv5 Helmet Detection")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Confidence thresholds
    st.subheader("Confidence Threshold")
    conf_helmet = st.slider("Helmet confidence", 0.0, 1.0, 0.4, 0.01)
    conf_no_helmet = st.slider("No helmet confidence", 0.0, 1.0, 0.5, 0.01)
    
    st.markdown("---")
    
    # Mode selection
    mode = st.radio("Select Mode", ["Upload Video", "Webcam"])
    
    # Metrics display
    st.markdown("---")
    st.subheader("Detection Metrics")
    col1, col2 = st.columns(2)
    helmet_metric = col1.empty()
    no_helmet_metric = col2.empty()
    
    # Status message
    st.markdown("---")
    status_message = st.empty()
    
    # Alert placeholder (for visual alerts)
    alert_placeholder = st.empty()

# Function to draw bounding boxes
def draw_boxes(frame, results_df):
    for _, row in results_df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = f"{row['name']} {row['confidence']:.2f}"
        color = (0, 255, 0) if row['name'] == 'helmet_on' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# State management
if 'alert_sent' not in st.session_state:
    st.session_state.alert_sent = False
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0

# Main processing based on mode
if mode == "Upload Video":
    st.subheader("Upload Video for Helmet Detection")
    st.markdown("**Data and drop file here**")
    st.caption("Limit 200MB per file - MP4, MOV, AVI, MPEG4")
    
    video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"], label_visibility="collapsed")
    
    if video_file is not None:
        # Save uploaded file temporarily
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        
        # Display file info
        file_size = video_file.size / (1024 * 1024)  # in MB
        st.info(f"Uploaded: {video_file.name} ({file_size:.1f}MB)")
        
        # Process video
        st.markdown("---")
        processing_status = st.empty()
        frame_placeholder = st.empty()
        
        cap = cv2.VideoCapture(temp_video_path)
        
        if not cap.isOpened():
            st.error("❌ Could not open the uploaded video. Please try another file.")
        else:
            # Get video properties for proper timing
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / fps if fps > 0 else 0.03
            
            # Create a queue for frame processing results
            result_queue = Queue()
            
            def process_video():
                frame_count = 0
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            result_queue.put(('end', None, None, None))
                            break
                        
                        # Perform detection
                        results = model(frame)
                        df = results.pandas().xyxy[0]
                        
                        # Filter by confidence thresholds
                        df_helmet = df[(df['name'] == 'helmet_on') & (df['confidence'] >= conf_helmet)]
                        df_no_helmet = df[(df['name'] == 'no_helmet') & (df['confidence'] >= conf_no_helmet)]
                        
                        helmet_count = len(df_helmet)
                        no_helmet_count = len(df_no_helmet)
                        
                        # Draw boxes
                        frame = draw_boxes(frame, pd.concat([df_helmet, df_no_helmet]))
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        result_queue.put(('frame', frame_rgb, helmet_count, no_helmet_count))
                        frame_count += 1
                        processing_status.info(f"📹 Processing frame {frame_count}...")
                        time.sleep(frame_delay)
                finally:
                    cap.release()
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
            
            # Start video processing in a separate thread
            processing_thread = threading.Thread(target=process_video, daemon=True)
            processing_thread.start()
            
            # Main display loop
            while True:
                if not result_queue.empty():
                    result_type, frame, helmet_count, no_helmet_count = result_queue.get()
                    
                    if result_type == 'end':
                        processing_status.success("✅ Video processing complete!")
                        helmet_metric.empty()
                        no_helmet_metric.empty()
                        status_message.empty()
                        alert_placeholder.empty()
                        st.session_state.alert_sent = False
                        break
                    
                    # Update display
                    frame_placeholder.image(frame, channels="RGB")
                    
                    # Update sidebar metrics
                    with st.sidebar:
                        helmet_metric.metric("✅ Helmet On", helmet_count)
                        no_helmet_metric.metric("🚨 No Helmet", no_helmet_count)
                        
                        # Update status message
                        if no_helmet_count > 0:
                            status_message.error("❌ No helmet detected!")
                        else:
                            status_message.success("✅ All wearing helmets")
                    
                    # Check for alerts (without affecting sidebar)
                    if no_helmet_count > 0:
                        alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
                        if not st.session_state.alert_sent:
                            if send_telegram_alert(f"⚠️ ALERT: {no_helmet_count} rider(s) without helmet detected!"):
                                st.session_state.alert_sent = True
                    else:
                        alert_placeholder.empty()
                        st.session_state.alert_sent = False
                
                time.sleep(0.01)  # Small sleep to prevent busy waiting
    else:
        st.info("⬆️ Please upload a video to begin helmet detection.")

else:  # Webcam Mode
    st.subheader("Real-time Helmet Detection")
    st.info("📷 Webcam is active. Looking for helmet violations...")
    
    class VideoProcessor:
        def __init__(self):
            self.helmet_count = 0
            self.no_helmet_count = 0
            self.person_count = 0
            self.alert_triggered = False
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Perform detection
            results = model(img)
            df = results.pandas().xyxy[0]
            
            # Filter by confidence thresholds
            df_helmet = df[(df['name'] == 'helmet_on') & (df['confidence'] >= conf_helmet)]
            df_no_helmet = df[(df['name'] == 'no_helmet') & (df['confidence'] >= conf_no_helmet)]
            df_person = df[df['name'] == 'person']
            
            self.helmet_count = len(df_helmet)
            self.no_helmet_count = len(df_no_helmet)
            self.person_count = len(df_person)
            self.alert_triggered = (self.no_helmet_count > 0) or (self.person_count > 0)
            
            # Draw boxes for all detections
            img = draw_boxes(img, pd.concat([df_helmet, df_no_helmet, df_person]))
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    webrtc_ctx = webrtc_streamer(
        key="helmet-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    def update_ui():
        while True:
            if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                processor = webrtc_ctx.video_processor
                
                # Update sidebar metrics and status
                with st.sidebar:
                    helmet_metric.metric("✅ Helmet On", processor.helmet_count)
                    no_helmet_metric.metric("🚨 No Helmet", processor.no_helmet_count)
                    
                    # Update status message
                    if processor.no_helmet_count > 0:
                        status_message.error("❌ No helmet detected!")
                    else:
                        status_message.success("✅ All wearing helmets")
                
                # Check for alerts (without affecting sidebar)
                current_time = time.time()
                if processor.alert_triggered:
                    alert_message = ""
                    if processor.no_helmet_count > 0:
                        alert_message += f"⚠️ {processor.no_helmet_count} rider(s) without helmet detected! "
                    if processor.person_count > 0:
                        alert_message += f"👤 {processor.person_count} person(s) detected!"
                    
                    alert_placeholder.error(alert_message.strip())
                    
                    # Send Telegram alert (throttled to avoid spamming)
                    if (not st.session_state.alert_sent or 
                        (current_time - st.session_state.last_alert_time) > 60):  # 60 seconds cooldown
                        if send_telegram_alert(alert_message.strip()):
                            st.session_state.alert_sent = True
                            st.session_state.last_alert_time = current_time
                else:
                    alert_placeholder.empty()
                    st.session_state.alert_sent = False
            else:
                with st.sidebar:
                    helmet_metric.empty()
                    no_helmet_metric.empty()
                    status_message.info("📷 Webcam inactive.")
                st.session_state.alert_sent = False
            
            time.sleep(0.1)  # Reduced sleep time for more responsive UI
    
    ui_thread = threading.Thread(target=update_ui, daemon=True)
    ui_thread.start()
