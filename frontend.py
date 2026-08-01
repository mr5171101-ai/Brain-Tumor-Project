import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

st.set_page_config(page_title="Brain Tumor Detection")
st.title("🧠 Brain Tumor Detection")

uploaded_file = st.file_uploader("MRI Image Upload Karein", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect Tumor"):
        with st.spinner("Processing..."):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # Send request to FastAPI backend
            response = requests.post(
                "http://127.0.0.1:8000/predict/",
                files={"file": ("image.jpg", img_bytes, "image/jpeg")}
            )
            
            if response.status_code == 200:
                data = response.json()
                detections = data.get("detections", [])
                
                draw = ImageDraw.Draw(image)
                for det in detections:
                    box = det["bbox"]
                    conf = det["confidence"]
                    draw.rectangle(box, outline="red", width=3)
                    draw.text((box[0], box[1] - 10), f"Tumor: {conf:.2f}", fill="red")
                
                st.image(image, caption="Detection Result", use_column_width=True)
                st.success(f"Total Detections: {len(detections)}")
            else:
                st.error("Backend response me error aaya.")