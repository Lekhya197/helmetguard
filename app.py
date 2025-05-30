import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import torch
import cv2
import numpy as np

# Configuration
CONF_THRESHOLD = 0.25  # Confidence threshold
MODEL_PATH = 'best.pt'

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    model.conf = CONF_THRESHOLD
    model.eval()
    st.success("✅ Model loaded successfully!")
    return model

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.frame_count = 0
        self.last_log_time = 0

    def recv(self, frame):
        self.frame_count += 1
        try:
            # Convert frame to numpy array (BGR format)
            img = frame.to_ndarray(format="bgr24")
            
            # Store original dimensions for debugging
            orig_h, orig_w = img.shape[:2]
            
            # Convert to RGB (YOLOv5 expects RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Let YOLOv5 handle resizing internally
            results = self.model(img_rgb)
            
            # Get detections
            detections = results.pandas().xyxy[0]
            
            # Draw bounding boxes on original image (BGR)
            for _, det in detections.iterrows():
                x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                label = f"{det['name']} {det['confidence']:.2f}"
                color = (0, 255, 0) if det['name'] == 'helmet_on' else (0, 0, 255)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Debug logging (every 30 frames)
            if self.frame_count % 30 == 0:
                st.write(f"📊 Frame {self.frame_count} - Detections:")
                st.write(detections)
                st.write(f"Original dimensions: {orig_w}x{orig_h}")
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            return frame

# Streamlit UI
st.title("🪖 Real-time Helmet Detection")
st.markdown("""
This application detects helmets in real-time using your webcam.
- ✅ Green boxes: Proper helmet usage
- ❌ Red boxes: No helmet detected
""")

# Webcam streamer
webrtc_ctx = webrtc_streamer(
    key="helmet-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 20, "max": 30}
        },
        "audio": False
    },
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

if not webrtc_ctx.state.playing:
    st.info("⌛ Waiting for webcam to start...")
    st.warning("""
    If the webcam doesn't start:
    1. Check browser permissions
    2. Try a different browser (Chrome works best)
    3. Ensure no other app is using the webcam
    """)
