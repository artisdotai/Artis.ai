"""Configuration for LLM services with proper error handling and retry logic"""
import os
import logging
from datetime import timedelta
import psutil

logger = logging.getLogger(__name__)

# LLM Service Configuration
LLM_CONFIG = {
    'openai': {
        'api_key': os.environ.get('OPENAI_API_KEY'),
        'model': 'gpt-4',  # Latest stable model
        'max_retries': 3,
        'timeout': 30,
        'rate_limit': {
            'max_requests': 50,
            'time_window': timedelta(minutes=1)
        }
    },
    'cohere': {
        'api_key': os.environ.get('COHERE_API_KEY'),
        'max_retries': 3,
        'timeout': 30,
        'rate_limit': {
            'max_requests': 40,
            'time_window': timedelta(minutes=1)
        }
    },
    'huggingface': {
        'api_key': os.environ.get('HUGGINGFACE_API_KEY'),
        'max_retries': 3,
        'timeout': 30,
        'rate_limit': {
            'max_requests': 30,
            'time_window': timedelta(minutes=1)
        }
    }
}

def validate_llm_config() -> bool:
    """Validate LLM configuration and API keys"""
    missing_keys = []
    for service, config in LLM_CONFIG.items():
        if not config.get('api_key'):
            missing_keys.append(service)
            logger.warning(f"Missing API key for {service}")

    if missing_keys:
        logger.warning(f"Missing API keys for services: {', '.join(missing_keys)}")
        return False
    return True

def get_system_metrics() -> dict:
    """Get system resource metrics"""
    try:
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        return {}

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    pass

class LLMServiceUnavailable(Exception):
    """Exception raised when LLM service is unavailable"""
    pass

class InvalidAPIResponse(Exception):
    """Exception raised when API response is invalid"""
    pass