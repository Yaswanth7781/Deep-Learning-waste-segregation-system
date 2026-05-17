import os
import sys
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from PIL import Image
import io
import cv2
import base64
import uuid
import shutil
import uvicorn
from keras.layers import Dense

app = FastAPI()

class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

# 1. Custom Waste Categorization Model
model = tf.keras.models.load_model(
    "model/waste_mobilenetv2.h5",
    custom_objects={"Dense": CustomDense},
    compile=False
)
class_names = ['hazardous', 'non_recyclable', 'organic', 'recyclable']

# 2. General ImageNet Model for Specific Object Detection
imagenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store detection history during the app's lifecycle
session_history = {
    'hazardous': [],
    'non_recyclable': [],
    'organic': [],
    'recyclable': []
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "mode": "live",
        "history": session_history
    })

def process_frame(frame):
    # Preprocess frame for prediction
    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    processed = np.expand_dims(rgb, axis=0)
    processed = tf.keras.applications.mobilenet_v2.preprocess_input(processed)
    
    # Predict Waste Category
    preds = model.predict(processed, verbose=0)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    result = class_names[class_idx]
    
    # Draw prediction on the frame
    text = f"{result} ({confidence*100:.1f}%)"
    
    if result == 'hazardous': color = (0, 0, 255) # Red
    elif result == 'non_recyclable': color = (0, 165, 255) # Orange
    elif result == 'organic': color = (0, 255, 0) # Green
    else: color = (255, 255, 0) # Cyan for recyclable
    
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    return frame, result, confidence

def get_imagenet_label(image_rgb):
    # Get specific object name using ImageNet
    resized = cv2.resize(image_rgb, (224, 224))
    processed = np.expand_dims(resized, axis=0)
    processed = tf.keras.applications.mobilenet_v2.preprocess_input(processed)
    preds = imagenet_model.predict(processed, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
    label = decoded[0][1]
    return label.replace('_', ' ').title()

@app.post("/upload")
async def upload_media(request: Request, file: UploadFile = File(...)):
    
    if file.content_type.startswith('image/'):
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Convert to cv2 format (BGR) for waste processing
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        
        # Detect specific object name
        waste_name = get_imagenet_label(np.array(image))
        
        # Detect waste category
        frame, result, confidence = process_frame(open_cv_image)
        
        # Add to history
        if waste_name not in session_history[result]:
            session_history[result].append(waste_name)
        
        # Encode frame to base64 for display
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "mode": "upload",
            "media_type": "image",
            "image_data": frame_base64,
            "prediction": result,
            "confidence": round(confidence * 100, 2),
            "history": session_history,
            "detected_name": waste_name
        })
        
    elif file.content_type.startswith('video/'):
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Scan the video to detect all items (1 frame per second, max 30 seconds)
        cap = cv2.VideoCapture(filepath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0 or fps > 120:  # Fallback if FPS is not detected properly
            fps = 30
            
        frame_count = 0
        processed_samples = 0
        max_samples = 30 # Prevent long processing times
        
        while True:
            success, frame = cap.read()
            if not success or processed_samples >= max_samples:
                break
                
            # Process 1 frame every second
            if frame_count % fps == 0:
                # Convert to RGB for ImageNet detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                waste_name = get_imagenet_label(frame_rgb)
                
                # Get the waste category
                _, result, confidence = process_frame(frame)
                
                # Add to history
                if waste_name not in session_history[result]:
                    session_history[result].append(waste_name)
                    
                processed_samples += 1
                
            frame_count += 1
            
        cap.release()
            
        return templates.TemplateResponse("index.html", {
            "request": request,
            "mode": "upload",
            "media_type": "video",
            "video_filename": filename,
            "history": session_history
        })
    else:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "mode": "upload",
            "error": "Unsupported file format. Please upload an image or video.",
            "history": session_history
        })

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, _, _ = process_frame(frame)
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_video_frames(filepath):
    camera = cv2.VideoCapture(filepath)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, _, _ = process_frame(frame)
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/uploaded_video_feed/{filename}")
def uploaded_video_feed(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        return StreamingResponse(generate_video_frames(filepath), media_type="multipart/x-mixed-replace; boundary=frame")
    return {"error": "Video not found"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
