"""Spydefi bot API connector for social trading signals integration"""
import logging
import os
import time
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SpydefiConnector:
    def __init__(self):
        """Initialize Spydefi connector with caching and rate limiting"""
        self.base_url = "https://api.spydefi.io/v1"
        self.api_key = os.getenv("SPYDEFI_API_KEY")
        self.use_mock_data = True  # Set to False when API key is available
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
        self.last_request_time = None
        self.min_request_interval = 1.0  # seconds
        self.max_retries = 3
        self.retry_delay = 2.0  # seconds

    def get_social_signals(self, token_address: str, chain: str = 'solana') -> Dict[str, Any]:
        """Get social trading signals from Spydefi with enhanced error handling"""
        cache_key = f"social_signals_{chain}_{token_address}"

        if not self.use_mock_data:
            # Check cache
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.utcnow() - timestamp < self.cache_duration:
                    logger.debug(f"Using cached social signals for {token_address}")
                    return cached_data

            try:
                endpoint = f"/social/signals/{chain}/{token_address}"
                response = self._make_api_request_with_retry(endpoint)

                if response and 'signals' in response:
                    self.cache[cache_key] = (response, datetime.utcnow())
                    return response

            except Exception as e:
                logger.error(f"Error fetching Spydefi social signals: {str(e)}")
                return self._generate_mock_signals(token_address)

        return self._generate_mock_signals(token_address)

    def get_trending_signals(self, chain: str = 'solana', limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending trading signals from Spydefi"""
        cache_key = f"trending_signals_{chain}"
        
        if not self.use_mock_data:
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.utcnow() - timestamp < self.cache_duration:
                    return cached_data

            try:
                endpoint = f"/social/trending/{chain}"
                response = self._make_api_request_with_retry(endpoint, params={'limit': limit})
                
                if response and 'trending' in response:
                    self.cache[cache_key] = (response['trending'], datetime.utcnow())
                    return response['trending']
                    
            except Exception as e:
                logger.error(f"Error fetching Spydefi trending signals: {str(e)}")
                return self._generate_mock_trending_signals(limit)
        
        return self._generate_mock_trending_signals(limit)

    def get_social_metrics(self, token_address: str) -> Dict[str, Any]:
        """Get social metrics for a token"""
        cache_key = f"social_metrics_{token_address}"

        if not self.use_mock_data:
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.utcnow() - timestamp < self.cache_duration:
                    return cached_data

            try:
                endpoint = f"/social/metrics/{token_address}"
                response = self._make_api_request_with_retry(endpoint)

                if response and 'metrics' in response:
                    self.cache[cache_key] = (response, datetime.utcnow())
                    return response

            except Exception as e:
                logger.error(f"Error fetching social metrics: {str(e)}")
                return self._generate_mock_metrics(token_address)

        return self._generate_mock_metrics(token_address)

    def _make_api_request_with_retry(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make API request with retry logic and rate limiting"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                return self._make_api_request(endpoint, params)
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    logger.error(f"Failed after {self.max_retries} retries: {str(e)}")
                    raise
                logger.warning(f"Retry {retry_count}/{self.max_retries} after error: {str(e)}")
                time.sleep(self.retry_delay * retry_count)

    def _make_api_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make rate-limited API request to Spydefi"""
        if not self.api_key:
            raise ValueError("Spydefi API key not configured")

        # Implement rate limiting
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

        url = f"{self.base_url}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.get(url, headers=headers, params=params)
        self.last_request_time = time.time()

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:  # Rate limit exceeded
            logger.warning("Rate limit exceeded, implementing exponential backoff")
            raise requests.exceptions.RequestException("Rate limit exceeded")
        else:
            logger.error(f"Spydefi API error: {response.status_code} - {response.text}")
            raise requests.exceptions.RequestException(f"Spydefi API error: {response.status_code}")

    def clear_cache(self):
        """Clear the cache manually if needed"""
        self.cache.clear()
        logger.info("Spydefi connector cache cleared")

    def _generate_mock_signals(self, token_address: str) -> Dict[str, Any]:
        """Generate realistic mock social trading signals"""
        import random

        signals = []
        for _ in range(random.randint(3, 8)):
            signal_type = random.choice(['BUY', 'SELL'])
            confidence = random.uniform(0.6, 0.95)
            timestamp = datetime.utcnow() - timedelta(hours=random.randint(1, 48))
            
            signal = {
                'type': signal_type,
                'confidence': confidence,
                'timestamp': timestamp.isoformat(),
                'trader_id': f"TRADER_{random.randint(1000, 9999)}",
                'followers': random.randint(1000, 50000),
                'success_rate': random.uniform(0.7, 0.95),
                'predicted_roi': random.uniform(10, 100),
                'sentiment_score': random.uniform(0.6, 0.9)
            }
            signals.append(signal)

        return {
            'token_address': token_address,
            'signals': signals,
            'aggregated_sentiment': random.uniform(0.6, 0.9),
            'signal_strength': random.uniform(0.5, 1.0),
            'total_reach': random.randint(50000, 500000),
            'updated_at': datetime.utcnow().isoformat()
        }

    def _generate_mock_trending_signals(self, limit: int) -> List[Dict[str, Any]]:
        """Generate realistic mock trending signals"""
        import random

        trending = []
        for i in range(limit):
            signal = {
                'token_address': f"SOLTEST{i+1}" * 8,
                'signal_type': random.choice(['BUY', 'SELL']),
                'confidence_score': random.uniform(0.7, 0.95),
                'trader_count': random.randint(3, 15),
                'total_reach': random.randint(50000, 500000),
                'avg_roi_prediction': random.uniform(20, 100),
                'momentum_score': random.uniform(0.6, 0.9),
                'trend_strength': random.uniform(1.2, 2.5),
                'timestamp': datetime.utcnow().isoformat()
            }
            trending.append(signal)

        return trending

    def _generate_mock_metrics(self, token_address: str) -> Dict[str, Any]:
        """Generate realistic mock social metrics"""
        import random

        return {
            'token_address': token_address,
            'metrics': {
                'social_score': random.uniform(60, 95),
                'total_mentions': random.randint(1000, 10000),
                'unique_traders': random.randint(100, 1000),
                'sentiment_ratio': random.uniform(0.6, 0.9),
                'engagement_rate': random.uniform(0.02, 0.15),
                'trader_confidence': random.uniform(0.7, 0.95),
                'trend_direction': random.choice(['bullish', 'neutral', 'bearish']),
                'trend_strength': random.uniform(0.5, 1.0)
            },
            'updated_at': datetime.utcnow().isoformat()
        }