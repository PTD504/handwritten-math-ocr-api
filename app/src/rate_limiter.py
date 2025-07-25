import time
import asyncio
from typing import Dict, Optional, Tuple
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
import redis
import logging
from datetime import datetime, timedelta
import hashlib
import json
from functools import wraps
import os
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RateLimitType(Enum):
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"

@dataclass
class RateLimitConfig:
    requests_per_minute: int = 10
    requests_per_hour: int = 100
    requests_per_day: int = 500
    concurrent_requests: int = 5
    burst_threshold: int = 20
    burst_window: int = 60
    block_duration: int = 300
    
    # Special limits for authenticated vs anonymous users
    authenticated_multiplier: float = 2.0
    anonymous_daily_limit: int = 50

class RateLimitStorage:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        self.in_memory_storage = {}
        self.setup_storage(redis_url)
    
    def setup_storage(self, redis_url: Optional[str]):
        """Setup storage backend (Redis or in-memory)"""
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis connected successfully for rate limiting")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory storage: {e}")
                self.redis_client = None
        else:
            logger.info("Using in-memory storage for rate limiting")
    
    async def get_count(self, key: str) -> int:
        """Get current count for a key"""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                return int(value) if value else 0
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                return 0
        else:
            # Clean up expired entries on read for in-memory to keep it simple
            if key in self.in_memory_storage and time.time() > self.in_memory_storage[key].get('expires', 0):
                del self.in_memory_storage[key]
                return 0
            return self.in_memory_storage.get(key, {}).get('count', 0)
    
    async def increment(self, key: str, ttl: int) -> int:
        """Increment counter with TTL"""
        if self.redis_client:
            try:
                pipeline = self.redis_client.pipeline()
                pipeline.incr(key)
                pipeline.expire(key, ttl)
                results = pipeline.execute()
                return results[0]
            except Exception as e:
                logger.error(f"Redis increment error: {e}")
                return 1
        else:
            current_time = time.time()
            if key not in self.in_memory_storage or current_time > self.in_memory_storage[key]['expires']:
                self.in_memory_storage[key] = {'count': 0, 'expires': current_time + ttl}
            
            self.in_memory_storage[key]['count'] += 1
            return self.in_memory_storage[key]['count']
    
    async def set_block(self, key: str, duration: int):
        """Set a block for a key"""
        block_key = f"block:{key}"
        if self.redis_client:
            try:
                self.redis_client.setex(block_key, duration, "blocked")
            except Exception as e:
                logger.error(f"Redis set block error: {e}")
        else:
            self.in_memory_storage[block_key] = {
                'blocked': True,
                'expires': time.time() + duration
            }
    
    async def is_blocked(self, key: str) -> bool:
        """Check if a key is blocked"""
        block_key = f"block:{key}"
        if self.redis_client:
            try:
                return bool(self.redis_client.get(block_key))
            except Exception as e:
                logger.error(f"Redis is_blocked error: {e}")
                return False
        else:
            blocked_data = self.in_memory_storage.get(block_key)
            if blocked_data and blocked_data.get('blocked'):
                return time.time() < blocked_data.get('expires', 0)
            return False
    
    async def cleanup_expired(self):
        """Clean up expired entries (for in-memory storage)"""
        if not self.redis_client:
            current_time = time.time()
            # Clean up both rate limit keys and block keys
            keys_to_delete = [
                key for key, data in self.in_memory_storage.items()
                if isinstance(data, dict) and data.get('expires', 0) < current_time
            ]
            for key in keys_to_delete:
                del self.in_memory_storage[key]

class RateLimiter:
    def __init__(self, config: RateLimitConfig, storage: RateLimitStorage):
        self.config = config
        self.storage = storage
        self.active_requests = {}
        
        # Start cleanup task for in-memory storage
        if not storage.redis_client:
            asyncio.create_task(self._cleanup_task())
    
    async def _cleanup_task(self):
        while True:
            try:
                await self.storage.cleanup_expired()
                await asyncio.sleep(60)  # Cleanup every minute
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(60)
    
    def get_client_id(self, request: Request, user_data: Optional[dict] = None) -> Tuple[str, bool]:
        """
        Get client identifier and whether user is authenticated.
        For Model API, `user_data` can be a simplified dict indicating internal service.
        """
        # Check if user_data indicates an authenticated internal service call
        if user_data and user_data.get('uid') == "internal_service" and user_data.get('isAnonymous') is False:
            return f"service:{user_data['uid']}", True
        
        # Fallback to IP + User-Agent for anonymous/unrecognized calls
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        client_hash = hashlib.md5(f"{client_ip}:{user_agent}".encode()).hexdigest()
        return f"ip:{client_hash}", False # Treat as anonymous if not explicitly an internal service

    def get_rate_limits(self, is_authenticated: bool) -> Dict[str, int]:
        """Get rate limits based on user type"""
        base_limits = {
            "requests_per_minute": self.config.requests_per_minute,
            "requests_per_hour": self.config.requests_per_hour,
            "requests_per_day": self.config.requests_per_day,
        }
        
        if is_authenticated:
            # Authenticated users (now includes internal services) get higher limits
            return {
                key: int(value * self.config.authenticated_multiplier)
                for key, value in base_limits.items()
            }
        else:
            # Anonymous users (including unknown external IPs) get stricter daily limits
            base_limits["requests_per_day"] = min(
                base_limits["requests_per_day"],
                self.config.anonymous_daily_limit
            )
            return base_limits
    
    async def check_rate_limit(self, request: Request, user_data: Optional[dict] = None) -> Optional[JSONResponse]:
        """Check if request should be rate limited"""
        client_id, is_authenticated = self.get_client_id(request, user_data)
        
        # Check if client is blocked
        if await self.storage.is_blocked(client_id):
            logger.warning(f"Blocked client attempted request: {client_id}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": "Client is temporarily blocked due to excessive requests",
                    "retry_after": self.config.block_duration
                }
            )
        
        # Get rate limits for this user type
        limits = self.get_rate_limits(is_authenticated)
        
        # Check different time windows
        current_time = int(time.time())
        checks = [
            (f"{client_id}:minute:{current_time // 60}", limits["requests_per_minute"], 60),
            (f"{client_id}:hour:{current_time // 3600}", limits["requests_per_hour"], 3600),
            (f"{client_id}:day:{current_time // 86400}", limits["requests_per_day"], 86400),
        ]
        
        for key, limit, ttl in checks:
            count = await self.storage.increment(key, ttl)
            
            if count > limit:
                # Check if this is burst traffic (potential abuse)
                if count > limit * self.config.burst_threshold / self.config.requests_per_minute: # Dynamic burst threshold
                    await self.storage.set_block(client_id, self.config.block_duration)
                    logger.warning(f"Client blocked for abuse: {client_id}, count: {count}, limit: {limit}")
                
                # Calculate retry after
                retry_after = ttl - (current_time % ttl)
                
                logger.warning(f"Rate limit exceeded for {client_id}: {count}/{limit} in {ttl}s window")
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": f"Too many requests. Limit: {limit} per {ttl // 60} minutes",
                        "retry_after": retry_after,
                        "limit": limit,
                        "remaining": max(0, limit - count),
                        "reset": current_time + retry_after
                    }
                )
        
        return None
    
    async def track_concurrent_request(self, client_id: str) -> bool:
        """Track concurrent requests"""
        if client_id not in self.active_requests:
            self.active_requests[client_id] = 0
        
        if self.active_requests[client_id] >= self.config.concurrent_requests:
            logger.warning(f"Concurrent request limit exceeded for {client_id}")
            return False
        
        self.active_requests[client_id] += 1
        return True
    
    async def release_concurrent_request(self, client_id: str):
        """Release concurrent request tracking"""
        if client_id in self.active_requests:
            self.active_requests[client_id] = max(0, self.active_requests[client_id] - 1)
            if self.active_requests[client_id] == 0:
                del self.active_requests[client_id]

# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None

def init_rate_limiter(redis_url: Optional[str] = None, config: Optional[RateLimitConfig] = None):
    global _rate_limiter
    
    if config is None:
        config = RateLimitConfig()
    
    storage = RateLimitStorage(redis_url)
    _rate_limiter = RateLimiter(config, storage)
    
    logger.info("Rate limiter initialized successfully")
    return _rate_limiter

def get_rate_limiter() -> RateLimiter:
    if _rate_limiter is None:
        raise RuntimeError("Rate limiter not initialized. Call init_rate_limiter() first.")
    return _rate_limiter

async def apply_rate_limit(request: Request, user_data: Optional[dict] = None) -> Optional[JSONResponse]:
    """Apply rate limiting to a request"""
    try:
        rate_limiter = get_rate_limiter()
        return await rate_limiter.check_rate_limit(request, user_data)
    except Exception as e:
        logger.error(f"Rate limiting error: {e}")
        return None

def rate_limit_decorator(func):
    """Decorator to apply rate limiting to endpoint functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request and user data from arguments
        request = None
        user_data = None
        
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        # For the Model API, user_data might be simple, e.g., {'uid': 'internal_service', 'isAnonymous': False}
        # if an internal API key is used.
        # We need to make sure this logic is aligned with how `main.py` is passing `user_data` to middleware.
        
        # For now, `rate_limit_middleware` handles `user_data`
        # so this decorator might be less critical if middleware is always on.
        
        # Original: For decorator usage, it would look for current_user if passed
        # for key, value in kwargs.items():
        #     if key == "request":
        #         request = value
        #     elif key == "current_user": # This is what was used for Firebase in main.py
        #         user_data = value 
        
        if request:
            # We need to pass the "simulated" user_data (from API key check) to apply_rate_limit
            # In the new main.py, the middleware `rate_limit_middleware` is already doing this,
            # so the decorator might be redundant if all endpoints go through that middleware.
            # If you want to use the decorator for specific endpoints *instead of* the middleware,
            # you'd need to adapt how `user_data` is determined here.
            pass # The middleware handles it.
        
        return await func(*args, **kwargs)
    return wrapper

class ConcurrentRequestTracker:
    """Context manager for tracking concurrent requests"""
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.rate_limiter = get_rate_limiter()
    
    async def __aenter__(self):
        can_proceed = await self.rate_limiter.track_concurrent_request(self.client_id)
        if not can_proceed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many concurrent requests"
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.rate_limiter.release_concurrent_request(self.client_id)