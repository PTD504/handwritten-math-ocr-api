from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator

class PredictionRequest(BaseModel):
    """Request model for image prediction"""
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    
    class Config:
        schema_extra = {
            "example": {
                "image_data": "base64_encoded_image_string"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    formula: str = Field(..., description="Predicted LaTeX formula")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "formula": "\\frac{x^2}{2}",
                "confidence": 0.95,
                "processing_time": 0.45,
                "timestamp": "2024-01-15 10:30:45"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    images: List[str] = Field(..., min_items=1, max_items=10, description="List of base64 encoded images")
    
    @validator('images')
    def validate_images(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 images allowed per batch')
        return v

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    results: List[Dict[str, Any]] = Field(..., description="List of prediction results")
    total_images: int = Field(..., description="Total number of images processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    processing_time: float = Field(..., description="Total processing time")
    timestamp: str = Field(..., description="Batch processing timestamp")

class StatusResponse(BaseModel):
    """System status response"""
    status: str = Field(..., description="System status")
    api_version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    vocab_loaded: bool = Field(..., description="Whether vocabulary is loaded")
    device: str = Field(..., description="Computing device")
    model_load_time: Optional[float] = Field(None, description="Model loading time")
    total_predictions: int = Field(..., description="Total predictions made")
    uptime: float = Field(..., description="System uptime in seconds")

class HealthResponse(BaseModel):
    """Health check response"""
    healthy: bool = Field(..., description="Overall health status")
    checks: Dict[str, Any] = Field(..., description="Individual health checks")
    timestamp: str = Field(..., description="Health check timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: str = Field(..., description="Error timestamp")