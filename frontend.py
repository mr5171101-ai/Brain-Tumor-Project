import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    # Ensure your trained weights file (e.g., 'best_tumor.pt') is in the same directory
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# UI Header
# -----------------------------
st.title("🧠 Brain MRI Tumor Detection using YOLOv11")
st.markdown("Upload a brain MRI scan image to detect and localize tumor regions using your fine-tuned YOLO model.")

# -----------------------------
# File Uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Scan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original MRI Scan")
        st.image(uploaded_file, use_container_width=True)

    if st.button("🔍 Detect Tumor", use_container_width=True):
        with st.spinner("Analyzing MRI scan..."):
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(
                bytearray(uploaded_file.read()),
                dtype=np.uint8
            )
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Perform YOLO prediction
            results = model.predict(
                image,
                conf=0.25
            )

            # Extract results
            res = results[0]
            detection_count = len(res.boxes)
            
            # Plot bounding boxes on image
            predicted = res.plot()
            predicted = cv2.cvtColor(
                predicted,
                cv2.COLOR_BGR2RGB
            )

            with col2:
                st.subheader("✅ Detection Result")
                st.image(
                    predicted,
                    use_container_width=True
                )

            # Display Analysis Summary
            st.success("Analysis Completed Successfully!")
            st.metric(label="Total Tumor Regions Detected", value=detection_count)
            
            # Optional: Display confidence details for each detected box
            if detection_count > 0:
                st.write("### 📋 Detection Details:")
                for i, box in enumerate(res.boxes):
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0]) * 100
                    st.write(f"- **Region {i+1}**: {cls_name} ({conf:.2f}% confidence)")
            else:
                st.info("No tumor regions detected above the confidence threshold.")