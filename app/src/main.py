from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
import io
import os
import time
import logging
from typing import Optional, List
import base64
from pathlib import Path

from preprocess import preprocess_image
from im2latex import load_model, predict
from utils import load_vocab
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = None
vocab = None
idx2char = None
model = None
model_load_time = None
prediction_count = 0

# Pydantic models
class PredictionResponse(BaseModel):
    formula: str
    confidence: Optional[float] = None
    processing_time: float
    timestamp: str

class StatusResponse(BaseModel):
    status: str
    api_version: str
    model_loaded: bool
    vocab_loaded: bool
    device: str
    model_load_time: Optional[float]
    total_predictions: int
    uptime: float

class HealthResponse(BaseModel):
    healthy: bool
    checks: dict

class BatchPredictionRequest(BaseModel):
    images: List[str]
    beam_size: int = 3

def initialize_model():
    global device, vocab, idx2char, model, model_load_time
    
    start_time = time.time()
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        vocab_path = os.path.join(config.model_dir, 'vocab.json')
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        vocab, idx2char = load_vocab(vocab_path)
        logger.info(f"Loaded vocabulary with {len(vocab)} tokens")
        
        model_path = os.path.join(config.model_dir, 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = load_model(model_path, vocab=vocab, device=device)
        logger.info("Model loaded successfully")
        
        model_load_time = time.time() - start_time
        logger.info(f"Model initialization completed in {model_load_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

# Validate uploaded image file
def validate_image_file(file: UploadFile) -> None:
    if hasattr(file, 'size') and file.size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=config.ERROR_MESSAGES['file_too_large']
        )
    
    if file.filename:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=config.ERROR_MESSAGES['invalid_format']
            )

def process_image_data(image_data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Error processing image data: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid image data"
        )

# Update global prediction counter
def update_prediction_count():
    global prediction_count
    prediction_count += 1

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    initialize_model()

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    return f"""
    <html>
        <head>
            <title>{config.API_TITLE}</title>
        </head>
        <body>
            <h1>{config.API_TITLE}</h1>
            <p>Version: {config.API_VERSION}</p>
            <p>Visit <a href="/docs">/docs</a> for API documentation</p>
            <p>Visit <a href="/status">/status</a> for system status</p>
        </body>
    </html>
    """

# Main prediction endpoint (original)
@app.post("/predict", response_model=PredictionResponse)
async def predict_formula(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    beam_size: int = Query(3, ge=1, le=10)
):
    start_time = time.time()
    
    try:
        validate_image_file(file)
        
        image_bytes = await file.read()
        image = process_image_data(image_bytes)
        
        processed_image = preprocess_image(image)
        
        formula, confidence = predict(processed_image, model, vocab, idx2char, device, beam_size=beam_size)
        
        processing_time = time.time() - start_time
        
        # Update counter in background
        background_tasks.add_task(update_prediction_count)
        
        return PredictionResponse(
            formula=formula,
            confidence=confidence,
            processing_time=processing_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        processing_time = time.time() - start_time
        return PredictionResponse(
            formula=config.ERROR_MESSAGES['processing_error'],
            confidence=confidence,
            processing_time=processing_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

# Batch prediction endpoint
# Predict formulas from multiple base64 encoded images
@app.post("/predict/batch")
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    start_time = time.time()
    results = []
    
    try:
        for i, base64_image in enumerate(request.images):
            try:
                # Decode base64 image
                image_data = base64.b64decode(base64_image)
                image = process_image_data(image_data)
                
                # Preprocess and predict
                processed_image = preprocess_image(image)
                
                formula, confidence_score = predict(processed_image, model, vocab, idx2char, device, beam_size=request.beam_size)
                
                if isinstance(formula, list):
                    formula = formula[0] if formula else ""
                
                results.append({
                    "index": i,
                    "formula": formula,
                    "confidence": confidence_score,
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                results.append({
                    "index": i,
                    "formula": config.ERROR_MESSAGES['processing_error'],
                    "confidence": None,
                    "success": False,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        background_tasks.add_task(lambda: setattr(globals(), 'prediction_count', 
                                                prediction_count + len(request.images)))
        
        return {
            "results": results,
            "total_images": len(request.images),
            "successful_predictions": sum(1 for r in results if r["success"]),
            "processing_time": processing_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse)
def get_status():
    import psutil
    
    return StatusResponse(
        status="healthy" if model is not None else "unhealthy",
        api_version=config.API_VERSION,
        model_loaded=model is not None,
        vocab_loaded=vocab is not None and idx2char is not None,
        device=str(device),
        model_load_time=model_load_time,
        total_predictions=prediction_count,
        uptime=time.time() - (model_load_time or 0)
    )

# Detail health check
@app.get("/health", response_model=HealthResponse)
def health_check():
    checks = {
        "model_loaded": model is not None,
        "vocab_loaded": vocab is not None and idx2char is not None,
        "device_available": device is not None,
        "model_files_exist": {
            "model.pth": os.path.exists(os.path.join(config.model_dir, 'model.pth')),
            "vocab.json": os.path.exists(os.path.join(config.model_dir, 'vocab.json'))
        }
    }
    
    healthy = all([
        checks["model_loaded"],
        checks["vocab_loaded"],
        checks["device_available"],
        all(checks["model_files_exist"].values())
    ])
    
    return HealthResponse(healthy=healthy, checks=checks)

# Model info endpoint
# Get model configuration and statistics
@app.get("/model/info")
def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_config": {
            "img_height": config.IMG_H,
            "img_width": config.IMG_W,
            "d_model": config.D_MODEL,
            "num_heads": config.NHEAD,
            "num_decoder_layers": config.NUM_DECODER_LAYERS,
            "dim_feedforward": config.DIM_FEEDFORWARD,
            "dropout": config.DROPOUT,
            "max_seq_len": config.MAX_SEQ_LEN
        },
        "vocab_info": {
            "vocab_size": len(vocab) if vocab else 0,
            "special_tokens": config.SPECIAL_TOKENS
        },
        "device": str(device),
        "model_parameters": sum(p.numel() for p in model.parameters()) if model else 0
    }

@app.get("/vocab")
def get_vocabulary(limit: int = Query(50, ge=1, le=1000)):
    if vocab is None:
        raise HTTPException(status_code=503, detail="Vocabulary not loaded")
    
    vocab_items = list(vocab.items())[:limit]
    
    return {
        "total_tokens": len(vocab),
        "displayed_tokens": len(vocab_items),
        "tokens": vocab_items,
        "special_tokens": config.SPECIAL_TOKENS
    }

# Metrics endpoint
@app.get("/metrics")
def get_metrics():
    import psutil
    
    return {
        "predictions": {
            "total": prediction_count,
            "rate": prediction_count / (time.time() - (model_load_time or 0)) if model_load_time else 0
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        },
        "uptime": time.time() - (model_load_time or 0)
    }

@app.get("/api/status")
def legacy_status():
    return {
        "status": "healthy",
        "api_version": config.API_VERSION,
        "model_loaded": model is not None,
        "vocab_loaded": vocab is not None and idx2char is not None,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)