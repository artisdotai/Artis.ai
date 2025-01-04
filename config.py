"""Application configuration with enhanced error handling and connection management"""
import os
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())
    DEBUG = True
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,  # Enable connection health checks
        "pool_recycle": 300,    # Recycle connections every 5 minutes
        "pool_timeout": 30,     # Connection timeout after 30 seconds
        "max_overflow": 5,      # Maximum number of connections above pool size
        "pool_size": 5,         # Base pool size
    }

    # Service Timeouts
    SERVICE_TIMEOUT = 30
    SERVICE_MAX_RETRIES = 3
    SERVICE_RETRY_DELAY = 5

    # Monitoring
    MONITORING_INTERVAL = 15  # seconds
    OPTIMIZATION_INTERVAL = 300  # seconds
    METRICS_RETENTION = 1000  # number of metrics to retain

    # API Rate Limits
    API_RATE_LIMITS = {
        'openai': {
            'requests_per_minute': 50,
            'tokens_per_minute': 10000
        },
        'cohere': {
            'requests_per_minute': 40,
            'tokens_per_minute': 8000
        },
        'huggingface': {
            'requests_per_minute': 30,
            'tokens_per_minute': 5000
        }
    }

    @staticmethod
    def init_app(app):
        """Initialize application configuration"""
        try:
            # Verify environment variables
            required_vars = ['DATABASE_URL', 'OPENAI_API_KEY', 'COHERE_API_KEY', 'HUGGINGFACE_API_KEY']
            missing_vars = [var for var in required_vars if not os.environ.get(var)]
            
            if missing_vars:
                logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            
            # Configure logging
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            logger.info("Application configuration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing application configuration: {str(e)}")
            return False

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Tighter connection pooling for production
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_timeout": 20,
        "max_overflow": 3,
        "pool_size": 5
    }

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
