# -*- coding: utf-8 -*-
# api.py
import os
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from resnet50nodown import resnet50nodown

app = FastAPI()

# Initialize model once at startup
MODEL_WEIGHTS = 'weights/gandetection_resnet50nodown_stylegan2.pth'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net = resnet50nodown(device, MODEL_WEIGHTS)

THRESHOLD = 0.5  # Adjust this based on your desired sensitivity

@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    try:
        # Load and process image
        image = Image.open(file.file).convert('RGB')
        image.load()
        
        # Get model prediction
        logit = net.apply(image)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)