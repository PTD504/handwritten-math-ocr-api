from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from src.preprocess import preprocess_image
from src.model import load_model, predict
from src.utils import load_vocab, latex_validator
from src.config import config
import os

app = FastAPI()

# Tải mô hình và vocab khi khởi động
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab, idx2char = load_vocab(os.path.join(config.model_dỉr, 'vocab.json'))
model = load_model(os.path.join(config.model_dir, 'model.pth'), vocab=vocab, device=device)

@app.post("/predict")
async def predict_formula(file: UploadFile = File(...)):
    try:
        # Đọc ảnh từ file upload
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Tiền xử lý ảnh
        processed_image = preprocess_image(image)
        
        # Dự đoán công thức
        formula = predict(model, processed_image, vocab, idx2char, device)
        
        # Kiểm tra và sửa lỗi LaTeX
        _, result = latex_validator(formula)
        
        return JSONResponse(content={"formula": result})
    except Exception as e:
        print(f"An error occurred: {e}") # Log lỗi để debug
        return JSONResponse(content={"formula": r"\text{Đã xảy ra lỗi trong quá trình xử lý ảnh. Vui lòng thử lại.}"}, status_code=200)

@app.get("/status")
def check_status():
    """
    Kiểm tra trạng thái của API.
    
    Returns:
        dict: Thông tin trạng thái của API, bao gồm phiên bản và trạng thái mô hình/vocab.
    """
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