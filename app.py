import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from PIL import Image
import io

app = FastAPI()
from keras.layers import Dense

# FIX: ignore unknown args
class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

model = tf.keras.models.load_model(
    "model/waste_mobilenetv2.h5",
    custom_objects={"Dense": CustomDense},
    compile=False
)
# Class labels (must match training folders)
class_names = ['hazardous', 'non_recyclable', 'organic', 'recyclable']

# Templates
templates = Jinja2Templates(directory="templates")

# Static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)

    # MobileNet preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    image = np.expand_dims(image, axis=0)
    return image

# Predict route
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed = preprocess_image(image)

    preds = model.predict(processed)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    result = class_names[class_idx]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result,
        "confidence": round(confidence * 100, 2)
    })