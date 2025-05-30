import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Configuration
CONF_THRESHOLD = 0.25  # Confidence threshold
MODEL_PATH = 'best.pt'

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    model.conf = CONF_THRESHOLD
    model.eval()
    return model

# Streamlit UI
st.title("🪖 Helmet Detection with YOLOv5")
st.write("Upload an image for helmet detection")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load model
    model = load_model()
    
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run inference
    results = model(img_cv)
    
    # Get detections
    detections = results.pandas().xyxy[0]
    
    # Show raw detections (for debugging)
    st.write("Detection Results:")
    st.write(detections)
    
    # Draw bounding boxes on the image
    output_image = img_array.copy()
    for _, det in detections.iterrows():
        x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
        label = f"{det['name']} {det['confidence']:.2f}"
        color = (0, 255, 0) if det['name'] == 'helmet_on' else (0, 0, 255)
        
        # Draw rectangle and label
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_image, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display the result
    st.image(output_image, caption='Detection Results', use_column_width=True)
    
    # Alert if no_helmet detected
    if 'no_helmet' in detections['name'].values:
        st.warning("🚨 No helmet detected!")
        # You can add your alert_no_helmet() function here if needed
