"""API Manager with robust connectivity and caching mechanisms"""
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional

import requests
from extensions import db

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker implementation for API calls"""
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if (datetime.utcnow() - self.last_failure_time).total_seconds() > self.reset_timeout:
                self.state = "half-open"
                return True
            return False
            
        return True  # half-open state allows one test request

    def record_success(self) -> None:
        """Record successful execution"""
        if self.state == "half-open":
            self.state = "closed"
        self.failures = 0
        self.last_failure_time = None

    def record_failure(self) -> None:
        """Record failed execution"""
        self.failures += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"

class APIManager:
    """Manages API connections with caching and fallback mechanisms"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes default TTL
        self.circuit_breakers = {}
        self.retry_delays = [1, 2, 4, 8, 16]  # Exponential backoff

    def _get_circuit_breaker(self, api_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for an API"""
        if api_name not in self.circuit_breakers:
            self.circuit_breakers[api_name] = CircuitBreaker()
        return self.circuit_breakers[api_name]

    def _get_cache_key(self, api_name: str, endpoint: str, params: Dict) -> str:
        """Generate cache key from API details"""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{api_name}:{endpoint}:{param_str}"

    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if valid"""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl:
                return data
            del self.cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: Dict) -> None:
        """Cache API response"""
        self.cache[cache_key] = (response, datetime.utcnow())

    def call_api(self, api_name: str, endpoint: str, method: str = 'GET',
                params: Optional[Dict] = None, data: Optional[Dict] = None,
                headers: Optional[Dict] = None, use_cache: bool = True) -> Dict:
        """Make API call with caching, circuit breaking, and retry logic"""
        
        params = params or {}
        headers = headers or {}
        circuit_breaker = self._get_circuit_breaker(api_name)
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for {api_name}")
            raise Exception(f"Service {api_name} is currently unavailable")
            
        # Check cache
        cache_key = self._get_cache_key(api_name, endpoint, params)
        if use_cache:
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached

        # Make API request with retries
        last_exception = None
        for delay in self.retry_delays:
            try:
                response = requests.request(
                    method=method,
                    url=endpoint,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=30
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Cache successful response
                if use_cache:
                    self._cache_response(cache_key, result)
                    
                circuit_breaker.record_success()
                return result
                
            except Exception as e:
                last_exception = e
                logger.error(f"API call failed: {str(e)}")
                time.sleep(delay)
                
        # All retries failed
        circuit_breaker.record_failure()
        raise last_exception or Exception("API call failed after retries")

def with_fallback(fallback_function: Callable) -> Callable:
    """Decorator to provide fallback for failed API calls"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                return fallback_function(*args, **kwargs)
        return wrapper
    return decorator

# Global API manager instance
api_manager = APIManager()
