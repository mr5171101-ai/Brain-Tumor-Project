from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="Brain Tumor Detection API")

# Model path
model = YOLO("models/best.pt")

@app.post("/predict/")
async def predict_brain_tumor(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # YOLO prediction
    results = model.predict(image, conf=0.25)
    
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "bbox": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])]
            })
            
    return {"status": "success", "detections": detections}