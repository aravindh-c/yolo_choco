import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Title
st.title("🍬 YOLO Candy Detection App")
st.markdown("Upload an image to detect candies using a trained YOLOv8 model.")

# Load YOLOv8 model
@st.cache_resource
def load_model(path='runs/detect/train/weights/best.pt'):
    return YOLO(path)

model = load_model()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If image uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save to temp file for prediction
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # Run prediction
    st.subheader("Detection Results")
    results = model.predict(temp_path, conf=0.4)  # you can adjust confidence threshold

    # Show result image with bounding boxes
    result_img = results[0].plot()  # Annotated image as numpy array
    st.image(result_img, caption='Predicted Image', use_column_width=True)

    # Display predicted class names and confidences
    st.subheader("Detected Objects")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        st.write(f"🔸 `{label}` - {conf*100:.2f}%")

    # Clean up temp file
    os.remove(temp_path)
