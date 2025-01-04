"""Solana health check functionality"""
import os
import logging
from datetime import datetime
from solana.rpc.api import Client

logger = logging.getLogger(__name__)

class Version:
    def __init__(self, major, minor, patch, raw_info=None):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.raw_info = raw_info

    @property
    def value(self):
        if self.raw_info:
            return str(self.raw_info)
        return f"{self.major}.{self.minor}.{self.patch}"

class Blockhash:
    def __init__(self, blockhash, slot):
        self.blockhash = blockhash
        self.slot = slot

    @property
    def value(self):
        return {"blockhash": self.blockhash, "slot": self.slot}

class SolanaClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        try:
            self.client = Client(endpoint)
            logger.info(f"Initialized Solana client with endpoint: {endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize Solana client: {str(e)}")
            raise

    def get_version(self):
        try:
            response = self.client.get_version()
            if isinstance(response, dict) and 'result' in response:
                version_info = response['result']
                return Version(1, 0, 0, version_info)
            return Version(1, 0, 0)
        except Exception as e:
            logger.error(f"Error getting Solana version: {str(e)}")
            return Version(1, 0, 0)

    def get_latest_blockhash(self):
        try:
            response = self.client.get_latest_blockhash()
            if isinstance(response, dict) and 'result' in response:
                result = response['result']
                blockhash = result.get('value', {}).get('blockhash', 'unknown')
                slot = result.get('value', {}).get('slot', 0)
                return Blockhash(blockhash, slot)
            return Blockhash('unknown', 0)
        except Exception as e:
            logger.error(f"Error getting latest blockhash: {str(e)}")
            return Blockhash('unknown', 0)

class SolanaHealthCheck:
    def __init__(self):
        self.client = None
        self.solana_api_key = os.environ.get('SOLANA_API_KEY')
        self.initialize_client()

    def initialize_client(self):
        try:
            endpoint = f"https://api.devnet.solana.com/{self.solana_api_key}" if self.solana_api_key else "https://api.devnet.solana.com"
            self.client = SolanaClient(endpoint)
            version = self.client.get_version()
            logger.info(f"Successfully initialized Solana client. Version: {version.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Solana client: {str(e)}")
            return False

    def check_health(self) -> dict:
        try:
            if not self.client:
                if not self.initialize_client():
                    return {
                        "status": "error",
                        "message": "Failed to initialize Solana client",
                        "timestamp": datetime.utcnow().isoformat()
                    }

            # Get version
            version = self.client.get_version()

            # Get latest blockhash
            blockhash_info = self.client.get_latest_blockhash()

            return {
                "status": "healthy",
                "version": version.value,
                "latest_blockhash": blockhash_info.value['blockhash'],
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Solana health check failed: {error_msg}")
            return {
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
