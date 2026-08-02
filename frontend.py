import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io

st.set_page_config(page_title="Brain Tumor Detection")
st.title("🧠 Brain Tumor Detection")

# Cache model so it loads only once
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load karne me issue aaya. Path check karein: {e}")

uploaded_file = st.file_uploader("MRI Image Upload Karein", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect Tumor"):
        with st.spinner("Processing..."):
            # Direct YOLO prediction without FastAPI
            results = model.predict(image, conf=0.25)
            
            # Process detections
            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0])
                    })
            
            # Draw boxes
            draw = ImageDraw.Draw(image)
            for det in detections:
                box = det["bbox"]
                conf = det["confidence"]
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1] - 10), f"Tumor: {conf:.2f}", fill="red")
            
            st.image(image, caption="Detection Result", use_column_width=True)
            st.success(f"Total Detections: {len(detections)}")
