from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from src.preprocess import preprocess_image
from src.im2latex import load_model, predict
from src.utils import load_vocab
from src.config import config
import os

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab, idx2char = load_vocab(os.path.join(config.model_dỉr, 'vocab.json'))
model = load_model(os.path.join(config.model_dir, 'model.pth'), vocab=vocab, device=device)

@app.post("/predict")
async def predict_formula(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        processed_image = preprocess_image(image)
        
        formula = predict(model, processed_image, vocab, idx2char, device)
        
        return JSONResponse(content={"formula": formula})
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"formula": r"\text{An error occurred while processing the image. Please try again.}"}, status_code=200)

@app.get("/status")
def check_status():
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "model_loaded": model is not None,
        "vocab_loaded": vocab is not None and idx2char is not None,
        "device": device
    }

@app.get("/")
def root():
    return {"message": "Handwritten Math OCR API"}