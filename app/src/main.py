from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks, Depends, status, Request
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

# FIREBASE
import json
import firebase_admin
from firebase_admin import credentials, auth
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

# RATE LIMITER 
from rate_limiter import (
    init_rate_limiter, 
    get_rate_limiter, 
    apply_rate_limit,
    rate_limit_decorator,
    ConcurrentRequestTracker,
    RateLimitConfig
)

# Load environment variables
load_dotenv()

from preprocess import preprocess_image
from im2latex import load_model, predict
from utils import load_vocab
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FIREBASE ADMIN SDK
firebase_admin_initialized = False

firebase_service_account_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY_JSON')

if firebase_service_account_json:
    try:
        cred_dict = json.loads(firebase_service_account_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        firebase_admin_initialized = True
        logger.info("Firebase Admin SDK initialized from FIREBASE_SERVICE_ACCOUNT_KEY_JSON environment variable.")
    except json.JSONDecodeError:
        logger.error("Error: FIREBASE_SERVICE_ACCOUNT_KEY_JSON is not a valid JSON string. Check your .env or Cloud Run config.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK from FIREBASE_SERVICE_ACCOUNT_KEY_JSON: {e}")
else:
    logger.warning("Firebase Admin SDK not initialized. Authentication will not work.")
    logger.warning("Please set FIREBASE_SERVICE_ACCOUNT_KEY_JSON in your .env file (local) or as an environment variable (Cloud Run).")

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
    allow_origins=["*"], # Allow all origins for development; restrict in production
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

def update_prediction_count():
    global prediction_count
    prediction_count += 1

@app.on_event("startup")
async def startup_event():
    app.state.start_time = time.time()
    
    # Initialize rate limiter
    try:
        redis_url = os.getenv('REDIS_URL')
        rate_limit_config = RateLimitConfig(
            requests_per_minute=int(os.getenv('RATE_LIMIT_PER_MINUTE', '10')),
            requests_per_hour=int(os.getenv('RATE_LIMIT_PER_HOUR', '100')),
            requests_per_day=int(os.getenv('RATE_LIMIT_PER_DAY', '500')),
            concurrent_requests=int(os.getenv('CONCURRENT_REQUESTS', '5')),
            authenticated_multiplier=float(os.getenv('AUTH_MULTIPLIER', '2.0')),
            anonymous_daily_limit=int(os.getenv('ANON_DAILY_LIMIT', '50')),
            block_duration=int(os.getenv('BLOCK_DURATION', '300'))
        )
        
        init_rate_limiter(redis_url, rate_limit_config)
        logger.info("Rate limiter initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize rate limiter: {e}")
    
    try:
        initialize_model()
    except Exception as e:
        logger.error(f"Failed to initialize model on startup: {e}. API may not function correctly.")

# Middleware to apply rate limiting globally
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for health checks and static endpoints
    skip_paths = ["/health", "/status", "/", "/docs", "/redoc", "/openapi.json"]
    if request.url.path in skip_paths:
        return await call_next(request)
    
    # Check if we have authorization header to determine user type
    user_data = None
    auth_header = request.headers.get("authorization")
    
    if auth_header and auth_header.startswith("Bearer ") and firebase_admin_initialized:
        try:
            token = auth_header.split(" ")[1]
            decoded_token = auth.verify_id_token(token)
            user_data = decoded_token
        except Exception:
            # If token verification fails, treat as anonymous user
            user_data = None
    
    # Apply rate limiting
    try:
        rate_limit_response = await apply_rate_limit(request, user_data)
        if rate_limit_response:
            return rate_limit_response
    except Exception as e:
        logger.error(f"Rate limiting middleware error: {e}")
    
    return await call_next(request)
        
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

# User authentication dependency
security = HTTPBearer()

async def get_current_firebase_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify Firebase ID Token from the Authorization Bearer header.
    Return decoded user information if the token is valid.
    """
    if not firebase_admin_initialized:
        logger.error("Attempted token verification when Firebase Admin SDK was not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is not ready to verify authentication tokens. Please try again later."
        )
    
    try:
        token = credentials.credentials
        
        # Verify Firebase ID Token
        decoded_token = auth.verify_id_token(token)
        logger.info(f"Firebase ID Token verified for user UID: {decoded_token.get('uid')}, Email: {decoded_token.get('email')}")
        return decoded_token
        
    except Exception as e:
        logger.error(f"Firebase ID Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/verify-firebase-login")
async def verify_firebase_login_endpoint(request: Request, token_data: dict):
    """
    Verify Firebase ID Token received directly from the frontend.
    """
    id_token = token_data.get("token")
    if not id_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Firebase ID Token is missing")
    
    if not firebase_admin_initialized:
        logger.error("Attempted /verify-firebase-login when Firebase Admin SDK was not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is not ready to verify authentication tokens. Please try again later."
        )
    
    try:
        decoded_token = auth.verify_id_token(id_token)
        logger.info(f"Backend Firebase ID Token verification successful for user: {decoded_token.get('email') or decoded_token.get('uid')}")
        return {"message": "Firebase ID Token verified successfully", "user_uid": decoded_token['uid'], "email": decoded_token.get('email')}
    except Exception as e:
        logger.error(f"Firebase ID Token verification failed on /verify-firebase-login: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Firebase ID Token verification failed: {e}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_formula(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file containing a mathematical formula."),
    current_user: dict = Depends(get_current_firebase_user)
):
    """
    Predicts the LaTeX formula from an uploaded image.
    Requires a valid Firebase ID Token in the Authorization header.
    Rate limiting is applied based on user authentication status.
    """
    global prediction_count
    start_time = time.time()

    user_identifier = current_user.get('email') or current_user.get('uid') or "Unknown User"
    logger.info(f"Prediction request from user: {user_identifier}, Provider: {current_user.get('firebase', {}).get('sign_in_provider')}")

    # Get rate limiter and client ID for concurrent request tracking
    try:
        rate_limiter = get_rate_limiter()
        client_id, is_authenticated = rate_limiter.get_client_id(request, current_user)
        
        # Use concurrent request tracker
        async with ConcurrentRequestTracker(client_id):
            
            if model is None:
                logger.error("Model is not loaded. Attempting to re-initialize.")
                try:
                    initialize_model()
                except Exception as e:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Model not initialized or failed to re-initialize: {e}")

            try:
                validate_image_file(file)
                
                image_bytes = await file.read()
                if not image_bytes:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

                image = process_image_data(image_bytes)
                processed_image = preprocess_image(image)
                
                formula, confidence = predict(model, processed_image, vocab, idx2char, device)
                
                processing_time = time.time() - start_time
                
                background_tasks.add_task(update_prediction_count)
                
                logger.info(f"Prediction successful for {file.filename} by {user_identifier}. Time: {processing_time:.4f}s")
                
                return PredictionResponse(
                    formula=formula,
                    confidence=confidence,
                    processing_time=processing_time,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Prediction error for user {user_identifier}: {e}", exc_info=True)
                processing_time = time.time() - start_time
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Internal server error during prediction: {config.ERROR_MESSAGES['processing_error']} - {e}"
                )
    
    except HTTPException as e:
        if e.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            logger.warning(f"Concurrent request limit exceeded for user {user_identifier}")
        raise

@app.post("/predict/batch")
async def predict_batch(
    request: Request,
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_firebase_user)
):
    """
    Predicts LaTeX formulas from multiple base64 encoded images.
    Requires a valid Firebase ID Token in the Authorization header.
    Rate limiting is applied based on user authentication status.
    """
    start_time = time.time()
    results = []
    
    user_identifier = current_user.get('email') or current_user.get('uid') or "Unknown User"
    logger.info(f"Batch prediction request from user: {user_identifier}, {len(batch_request.images)} images.")

    # Get rate limiter and client ID for concurrent request tracking
    try:
        rate_limiter = get_rate_limiter()
        client_id, is_authenticated = rate_limiter.get_client_id(request, current_user)
        
        # Use concurrent request tracker
        async with ConcurrentRequestTracker(client_id):
            
            if model is None:
                logger.error("Model is not loaded for batch prediction. Attempting to re-initialize.")
                try:
                    initialize_model()
                except Exception as e:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Model not initialized for batch prediction: {e}")

            try:
                for i, base64_image in enumerate(batch_request.images):
                    try:
                        image_data = base64.b64decode(base64_image)
                        if not image_data:
                            raise ValueError("Base64 string resulted in empty image data.")

                        image = process_image_data(image_data)
                        processed_image = preprocess_image(image)
                        formula, confidence_score = predict(model, processed_image, vocab, idx2char, device) 
                        
                        if isinstance(formula, list):
                            formula = formula[0] if formula else ""
                        
                        results.append({
                            "index": i,
                            "formula": formula,
                            "confidence": confidence_score,
                            "success": True
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing image {i} in batch for user {user_identifier}: {e}", exc_info=True)
                        results.append({
                            "index": i,
                            "formula": config.ERROR_MESSAGES['processing_error'],
                            "confidence": None,
                            "success": False,
                            "error": str(e)
                        })
                
                processing_time = time.time() - start_time
                background_tasks.add_task(lambda: setattr(globals(), 'prediction_count', 
                                                         prediction_count + len(batch_request.images)))
                
                logger.info(f"Batch prediction completed for user {user_identifier}. Total images: {len(batch_request.images)}, Success: {sum(1 for r in results if r['success'])}")
                
                return {
                    "results": results,
                    "total_images": len(batch_request.images),
                    "successful_predictions": sum(1 for r in results if r["success"]),
                    "processing_time": processing_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Batch prediction error for user {user_identifier}: {e}", exc_info=True)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during batch prediction: {str(e)}")
    
    except HTTPException as e:
        if e.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            logger.warning(f"Concurrent request limit exceeded for user {user_identifier}")
        raise

@app.get("/status", response_model=StatusResponse)
def get_status():
    import psutil
    
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    
    return StatusResponse(
        status="healthy" if model is not None else "unhealthy",
        api_version=config.API_VERSION,
        model_loaded=model is not None,
        vocab_loaded=vocab is not None and idx2char is not None,
        device=str(device),
        model_load_time=model_load_time,
        total_predictions=prediction_count,
        uptime=uptime
    )

@app.get("/health", response_model=HealthResponse)
def health_check():
    # Check rate limiter status
    rate_limiter_healthy = True
    try:
        get_rate_limiter()
    except Exception:
        rate_limiter_healthy = False
    
    checks = {
        "model_loaded": model is not None,
        "vocab_loaded": vocab is not None and idx2char is not None,
        "device_available": device is not None,
        "firebase_admin_initialized": firebase_admin_initialized,
        "rate_limiter_initialized": rate_limiter_healthy,
        "model_files_exist": {
            "model.pth": os.path.exists(os.path.join(config.model_dir, 'model.pth')),
            "vocab.json": os.path.exists(os.path.join(config.model_dir, 'vocab.json'))
        }
    }
    
    healthy = all([
        checks["model_loaded"],
        checks["vocab_loaded"],
        checks["device_available"],
        checks["firebase_admin_initialized"],
        checks["rate_limiter_initialized"],
        all(checks["model_files_exist"].values())
    ])
    
    return HealthResponse(healthy=healthy, checks=checks)

@app.get("/rate-limit/status")
async def get_rate_limit_status(request: Request, current_user: dict = Depends(get_current_firebase_user)):
    """
    Get current rate limit status for the authenticated user
    """
    try:
        rate_limiter = get_rate_limiter()
        client_id, is_authenticated = rate_limiter.get_client_id(request, current_user)
        limits = rate_limiter.get_rate_limits(is_authenticated)
        
        # Get current usage (this is a simplified version)
        current_time = int(time.time())
        minute_key = f"{client_id}:minute:{current_time // 60}"
        hour_key = f"{client_id}:hour:{current_time // 3600}"
        day_key = f"{client_id}:day:{current_time // 86400}"
        
        current_minute = await rate_limiter.storage.get_count(minute_key)
        current_hour = await rate_limiter.storage.get_count(hour_key)
        current_day = await rate_limiter.storage.get_count(day_key)
        
        return {
            "client_id": client_id,
            "is_authenticated": is_authenticated,
            "limits": limits,
            "current_usage": {
                "minute": current_minute,
                "hour": current_hour,
                "day": current_day
            },
            "remaining": {
                "minute": max(0, limits["requests_per_minute"] - current_minute),
                "hour": max(0, limits["requests_per_hour"] - current_hour),
                "day": max(0, limits["requests_per_day"] - current_day)
            },
            "concurrent_requests": rate_limiter.active_requests.get(client_id, 0),
            "max_concurrent": rate_limiter.config.concurrent_requests
        }
        
    except Exception as e:
        logger.error(f"Error getting rate limit status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving rate limit status"
        )

@app.get("/model/info")
def get_model_info():
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    
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
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vocabulary not loaded")
    
    vocab_items = list(vocab.items())[:limit]
    
    return {
        "total_tokens": len(vocab),
        "displayed_tokens": len(vocab_items),
        "tokens": vocab_items,
        "special_tokens": config.SPECIAL_TOKENS
    }

@app.get("/metrics")
def get_metrics():
    import psutil
    
    uptime_seconds = time.time() - (app.state.start_time if hasattr(app.state, 'start_time') else time.time())
    
    # Get rate limiter metrics
    rate_limiter_metrics = {}
    try:
        rate_limiter = get_rate_limiter()
        rate_limiter_metrics = {
            "active_concurrent_requests": len(rate_limiter.active_requests),
            "total_concurrent_requests": sum(rate_limiter.active_requests.values()),
            "max_concurrent_per_client": rate_limiter.config.concurrent_requests
        }
    except Exception:
        rate_limiter_metrics = {"error": "Rate limiter not available"}

    return {
        "predictions": {
            "total": prediction_count,
            "rate_per_second": prediction_count / uptime_seconds if uptime_seconds > 0 else 0
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        },
        "rate_limiter": rate_limiter_metrics,
        "uptime_seconds": uptime_seconds
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