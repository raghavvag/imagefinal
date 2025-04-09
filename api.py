# -*- coding: utf-8 -*-
# api.py
import os
import torch
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from resnet50nodown import resnet50nodown, EnhancedResNet, ResidualBlock

app = FastAPI()

# Initialize model once at startup
MODEL_WEIGHTS = 'weights/gandetection_resnet50nodown_stylegan2.pth'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Modify the model loading part
try:
    # Try loading with the 'model' key first
    weights_data = torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu'))
    if isinstance(weights_data, dict) and 'model' in weights_data:
        net = resnet50nodown(device, MODEL_WEIGHTS)
    else:
        # If the weights file doesn't have a 'model' key, load directly
        model = EnhancedResNet(ResidualBlock, [3, 4, 6, 3], output_classes=1, initial_stride=1)
        model.load_state_dict(weights_data)
        model = model.to(device).eval()
        net = model
except Exception as e:
    print(f"Error loading model: {e}")
    print("Attempting alternative loading method...")
    
    # Alternative loading method
    model = EnhancedResNet(ResidualBlock, [3, 4, 6, 3], output_classes=1, initial_stride=1)
    try:
        # Try loading the weights directly without the 'model' key
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu')))
    except:
        # If that fails, try to load with strict=False
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu')), strict=False)
        print("Loaded model with non-strict matching")
    
    model = model.to(device).eval()
    net = model

THRESHOLD = 0.5  # Adjust this based on your desired sensitivity

@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    try:
        # Load and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image.load()
        
        # Get model prediction
        logit = net.process_image(image)  # Changed from apply to process_image
        probability = torch.sigmoid(torch.tensor(logit)).item()
        
        # Determine result
        is_deepfake = probability > THRESHOLD
        
        return JSONResponse({
            "is_deepfake": is_deepfake,
            "probability": probability,
            "filename": file.filename
        })
        
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    import io  # Add import for BytesIO
    uvicorn.run(app, host="0.0.0.0", port=8000)