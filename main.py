import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Skull Fracture Detection",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")   # Change path if needed

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Skull Fracture Detection using YOLO")

st.markdown(
    """
    Upload a Skull X-ray image to detect fractures using the trained YOLO model.
    """
)

uploaded_file = st.file_uploader(
    "📤 Upload Skull X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original Image")
        st.image(uploaded_file, use_container_width=True)

    if st.button("🔍 Detect Fracture", use_container_width=True):

        with st.spinner("Analyzing image..."):

            file_bytes = np.asarray(
                bytearray(uploaded_file.read()),
                dtype=np.uint8
            )

            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            results = model.predict(
                image,
                conf=0.25
            )

            predicted = results[0].plot()

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

            st.success("Analysis Completed Successfully!")