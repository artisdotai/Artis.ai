"""
Configuration settings for the Eliza Framework implementation
"""
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Flask Configuration
    FLASK_SECRET_KEY: str = os.environ.get('FLASK_SECRET_KEY', 'dev_key_replace_in_production')

    # API Keys
    OPENAI_API_KEY: str = os.environ.get('OPENAI_API_KEY')
    COHERE_API_KEY: str = os.environ.get('COHERE_API_KEY')
    HUGGINGFACE_API_KEY: str = os.environ.get('HUGGINGFACE_API_KEY')
    SOLANA_API_KEY: str = os.environ.get('SOLANA_API_KEY')
    BITQUERY_API_KEY: str = os.environ.get('BITQUERY_API_KEY')
    ALCHEMY_API_KEY: str = os.environ.get('ALCHEMY_API_KEY')

    # Chain Configuration
    SUPPORTED_CHAINS = ['ETH', 'BSC', 'POLYGON', 'ARB', 'AVAX', 'SOLANA']

    # Risk Management Parameters
    DEFAULT_MAX_POSITION_SIZE = 0.05  # 5% of portfolio
    DEFAULT_STOP_LOSS = 0.02  # 2%
    DEFAULT_TAKE_PROFIT = 0.05  # 5%

    # Performance Monitoring
    MONITORING_INTERVAL = 30  # seconds
    HEALTH_CHECK_INTERVAL = 120  # seconds

    # Database Configuration
    SQLALCHEMY_DATABASE_URI: str = os.environ.get('DATABASE_URL')

    @staticmethod
    def validate():
        """Validate required configuration parameters"""
        required_keys = [
            'OPENAI_API_KEY',
            'SOLANA_API_KEY',
            'DATABASE_URL'
        ]

        missing_keys = [key for key in required_keys 
                       if not os.environ.get(key)]

        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")