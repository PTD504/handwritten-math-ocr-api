import os
import time
import logging
from typing import Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, status, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
import uvicorn

# Google Cloud imports
# pip install google-cloud-logging
from google.cloud import logging as cloud_logging

# ML/Image processing imports
import torch
from PIL import Image
import io
import base64

# Project imports
from preprocess import preprocess_image
from im2latex import load_model, predict
from utils import load_vocab
from config import config
from rate_limiter import (
    init_rate_limiter,
    get_rate_limiter,
    apply_rate_limit,
    RateLimitConfig,
    ConcurrentRequestTracker
)

# Pydantic models for API requests and responses
from models import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse, StatusResponse, HealthResponse, ErrorResponse

# LOGGING CONFIGURATION

def setup_logging():
    """Setup Cloud Logging with structured logging"""
    try:
        # Initialize Cloud Logging client
        if os.getenv('GOOGLE_CLOUD_PROJECT'):
            cloud_logging_client = cloud_logging.Client()
            cloud_logging_client.setup_logging()
            
        # Configure structured logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        
        # Create logger
        logger = logging.getLogger("model_api")
        logger.info("Logging configured successfully")
        return logger
    except Exception as e:
        # Fallback to basic logging if Cloud Logging fails
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("model_api")
        logger.warning(f"Failed to setup Cloud Logging, using basic logging: {e}")
        return logger

logger = setup_logging()

# APP CONFIGURATION

class AppConfig:
    """Application configuration"""
    
    def __init__(self):
        # Basic configuration
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # API Configuration
        self.api_key = os.getenv('MODEL_API_KEY')
        if not self.api_key and self.environment == 'production':
            logger.warning("MODEL_API_KEY not found in environment")
        
        # CORS Configuration
        cors_origins_str = os.getenv('CORS_ORIGINS', '')
        self.cors_origins = [origin.strip() for origin in cors_origins_str.split(',') if origin.strip()]
        
        # If no CORS origins specified, use restrictive defaults
        if not self.cors_origins:
            if self.environment == 'development':
                self.cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "http://127.0.0.1:8000"]
            else:
                self.cors_origins = []  # No origins allowed by default in production
        
        # Trusted hosts
        trusted_hosts_str = os.getenv('TRUSTED_HOSTS', '')
        self.trusted_hosts = [host.strip() for host in trusted_hosts_str.split(',') if host.strip()]
        
        # Rate limiting configuration
        self.rate_limit_config = RateLimitConfig(
            requests_per_minute=int(os.getenv('RATE_LIMIT_PER_MINUTE', '20')),
            requests_per_hour=int(os.getenv('RATE_LIMIT_PER_HOUR', '200')),
            requests_per_day=int(os.getenv('RATE_LIMIT_PER_DAY', '1000')),
            concurrent_requests=int(os.getenv('CONCURRENT_REQUESTS', '10')),
            authenticated_multiplier=float(os.getenv('AUTH_MULTIPLIER', '3.0')),
            anonymous_daily_limit=int(os.getenv('ANON_DAILY_LIMIT', '100')),
            block_duration=int(os.getenv('BLOCK_DURATION', '300'))
        )
        
        # Redis configuration for rate limiting
        self.redis_url = os.getenv('REDIS_URL')
        
        logger.info(f"Configuration loaded for environment: {self.environment}")
        logger.info(f"CORS origins: {self.cors_origins}")
        logger.info(f"Trusted hosts: {self.trusted_hosts}")

app_config = AppConfig()

# GLOBAL VARIABLES

device = None
vocab = None
idx2char = None
model = None
model_load_time = None
prediction_count = 0
app_start_time = time.time()

# AUTHENTICATION

def verify_api_key(request: Request):
    """Verify API key from request headers"""
    if not app_config.api_key:
        return True  # No API key configured, allow all requests
    
    auth_header = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key"
        )
    
    # Handle Bearer token format
    if auth_header.startswith("Bearer "):
        provided_key = auth_header.split(" ")[1]
    else:
        provided_key = auth_header
    
    if provided_key != app_config.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    
    return True

# MODEL INITIALIZATION

def initialize_model():
    """Initialize the ML model and vocabulary"""
    global device, vocab, idx2char, model, model_load_time
    
    start_time = time.time()
    
    try:
        # Setup device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load vocabulary
        vocab_path = os.path.join(config.model_dir, 'vocab.json')
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        vocab, idx2char = load_vocab(vocab_path)
        logger.info(f"Loaded vocabulary with {len(vocab)} tokens")
        
        # Load model
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

# UTILITY FUNCTIONS

def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file"""
    if hasattr(file, 'size') and file.size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE} bytes"
        )
    
    if file.filename:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
            )

def process_image_data(image_data: bytes) -> Image.Image:
    """Process raw image data into PIL Image"""
    try:
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Error processing image data: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid image data"
        )

def process_base64_image(base64_data: str) -> Image.Image:
    """Process base64 encoded image"""
    try:
        image_data = base64.b64decode(base64_data)
        return process_image_data(image_data)
    except Exception as e:
        logger.error(f"Error processing base64 image: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image data"
        )

def update_prediction_count():
    """Update global prediction counter"""
    global prediction_count
    prediction_count += 1

def get_user_data_from_request(request: Request) -> Dict[str, Any]:
    """Extract user data for rate limiting"""
    user_data = {"is_authenticated": False}
    
    if app_config.api_key:
        auth_header = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if auth_header:
            provided_key = auth_header.split(" ")[1] if auth_header.startswith("Bearer ") else auth_header
            if provided_key == app_config.api_key:
                user_data["is_authenticated"] = True
                user_data["uid"] = "authenticated_user"
    
    return user_data

# LIFESPAN MANAGEMENT

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting up application...")
    
    # Initialize rate limiter
    try:
        init_rate_limiter(app_config.redis_url, app_config.rate_limit_config)
        logger.info("Rate limiter initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize rate limiter: {e}")
    
    # Initialize ML model
    try:
        initialize_model()
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# FASTAPI APPLICATION

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url="/docs" if app_config.debug else None,
    redoc_url="/redoc" if app_config.debug else None,
    lifespan=lifespan
)

# MIDDLEWARE

# Trusted Host Middleware
if app_config.trusted_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=app_config.trusted_hosts
    )

# CORS Middleware
if app_config.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"]
    )

# Rate Limiting Middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Global rate limiting middleware"""
    # Skip rate limiting for health checks and docs
    skip_paths = ["/health", "/status", "/", "/docs", "/redoc", "/openapi.json"]
    if request.url.path in skip_paths:
        return await call_next(request)
    
    # Apply rate limiting
    try:
        user_data = get_user_data_from_request(request)
        rate_limit_response = await apply_rate_limit(request, user_data)
        if rate_limit_response:
            return rate_limit_response
    except Exception as e:
        logger.error(f"Rate limiting middleware error: {e}")
    
    return await call_next(request)

# Request ID Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to response headers"""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# ERROR HANDLERS

@app.exception_handler(HTTPException)
async def http_exception_handler_custom(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code} error: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            detail=exc.detail,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        ).dict()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.error(f"Validation error: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error",
            detail=f"Request validation failed: {exc.errors()}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        ).dict()
    )

# ROUTES

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information"""
    return f"""
    <html>
        <head>
            <title>{config.API_TITLE}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .status {{ color: green; }}
                .info {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{config.API_TITLE}</h1>
                <p class="status">✅ API is running</p>
                <div class="info">
                    <p><strong>Version:</strong> {config.API_VERSION}</p>
                    <p><strong>Environment:</strong> {app_config.environment}</p>
                    <p><strong>Model Status:</strong> {'✅ Loaded' if model else '❌ Not Loaded'}</p>
                </div>
                <p><a href="/docs">📚 API Documentation</a></p>
                <p><a href="/status">📊 System Status</a></p>
            </div>
        </body>
    </html>
    """

@app.post("/predict", response_model=PredictionResponse)
async def predict_formula(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file containing mathematical formula"),
    auth_check: bool = Depends(verify_api_key)
):
    """
    Predict LaTeX formula from uploaded image file.
    
    - **file**: Image file (PNG, JPG, JPEG, GIF, BMP)
    - **X-API-Key**: API key in header (if configured)
    """
    start_time = time.time()
    
    try:
        # Get rate limiter for concurrent request tracking
        rate_limiter = get_rate_limiter()
        user_data = get_user_data_from_request(request)
        client_id, is_authenticated = rate_limiter.get_client_id(request, user_data)
        
        async with ConcurrentRequestTracker(client_id):
            # Validate model is loaded
            if model is None:
                logger.error("Model not loaded, attempting to reinitialize")
                try:
                    initialize_model()
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Model initialization failed: {str(e)}"
                    )
            
            # Validate and process image
            validate_image_file(file)
            image_bytes = await file.read()
            
            if not image_bytes:
                raise HTTPException(
                    status_code=400,
                    detail="Empty file uploaded"
                )
            
            image = process_image_data(image_bytes)
            processed_image = preprocess_image(image)
            
            # Make prediction
            formula, confidence = predict(model, processed_image, vocab, idx2char, device)
            
            processing_time = time.time() - start_time
            
            # Update metrics
            background_tasks.add_task(update_prediction_count)
            
            logger.info(f"Prediction successful for {file.filename}. Time: {processing_time:.4f}s")
            
            return PredictionResponse(
                formula=formula,
                confidence=confidence,
                processing_time=processing_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request,
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    auth_check: bool = Depends(verify_api_key)
):
    """
    Predict LaTeX formulas from multiple base64 encoded images.
    
    - **images**: List of base64 encoded image strings (max 10)
    - **X-API-Key**: API key in header (if configured)
    """
    start_time = time.time()
    results = []
    
    try:
        # Get rate limiter for concurrent request tracking
        rate_limiter = get_rate_limiter()
        user_data = get_user_data_from_request(request)
        client_id, is_authenticated = rate_limiter.get_client_id(request, user_data)
        
        async with ConcurrentRequestTracker(client_id):
            # Validate model is loaded
            if model is None:
                logger.error("Model not loaded for batch prediction")
                try:
                    initialize_model()
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Model initialization failed: {str(e)}"
                    )
            
            # Process each image
            for i, base64_image in enumerate(batch_request.images):
                try:
                    image = process_base64_image(base64_image)
                    processed_image = preprocess_image(image)
                    formula, confidence = predict(model, processed_image, vocab, idx2char, device)
                    
                    if isinstance(formula, list):
                        formula = formula[0] if formula else ""
                    
                    results.append({
                        "index": i,
                        "formula": formula,
                        "confidence": confidence,
                        "success": True
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing image {i}: {e}")
                    results.append({
                        "index": i,
                        "formula": "",
                        "confidence": None,
                        "success": False,
                        "error": str(e)
                    })
            
            processing_time = time.time() - start_time
            successful_predictions = sum(1 for r in results if r["success"])
            
            # Update metrics
            background_tasks.add_task(lambda: globals().update({"prediction_count": prediction_count + len(batch_request.images)}))
            
            logger.info(f"Batch prediction completed. Total: {len(batch_request.images)}, Success: {successful_predictions}")
            
            return BatchPredictionResponse(
                results=results,
                total_images=len(batch_request.images),
                successful_predictions=successful_predictions,
                processing_time=processing_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status information"""
    uptime = time.time() - app_start_time
    
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
async def health_check():
    """Comprehensive health check endpoint"""
    # Check components
    rate_limiter_healthy = True
    try:
        get_rate_limiter()
    except Exception:
        rate_limiter_healthy = False
    
    model_files_exist = {
        "model.pth": os.path.exists(os.path.join(config.model_dir, 'model.pth')),
        "vocab.json": os.path.exists(os.path.join(config.model_dir, 'vocab.json'))
    }
    
    checks = {
        "model_loaded": model is not None,
        "vocab_loaded": vocab is not None and idx2char is not None,
        "device_available": device is not None,
        "rate_limiter_initialized": rate_limiter_healthy,
        "model_files_exist": model_files_exist,
        "environment": app_config.environment
    }
    
    healthy = all([
        checks["model_loaded"],
        checks["vocab_loaded"],
        checks["device_available"],
        checks["rate_limiter_initialized"],
        all(model_files_exist.values())
    ])
    
    return HealthResponse(
        healthy=healthy,
        checks=checks,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

# DEVELOPMENT SERVER

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=app_config.debug,
        log_level="info"
    )