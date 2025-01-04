"""Solana token monitoring with enhanced safety checks and mock data support"""
import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class SolanaTokenMonitor:
    """Monitor Solana tokens with safety validation"""

    def __init__(self, use_mock_data=True):
        """Initialize Solana token monitor"""
        self.base_url = "https://public-api.solscan.io"
        self.initialized = True
        self.use_mock_data = use_mock_data
        self.safety_thresholds = {
            'min_safety_score': 80,
            'min_liquidity': 1000,
            'min_holder_count': 100,
            'min_volume_24h': 5000
        }
        logger.info("Initialized Solana token monitor")

    def _generate_mock_safety_data(self, token_address: str) -> Dict[str, Any]:
        """Generate mock safety data for testing"""
        safety_score = random.randint(60, 100)
        liquidity = random.uniform(500, 2000000)
        holder_count = random.randint(50, 10000)
        volume_24h = random.uniform(1000, 1000000)

        return {
            'safety_score': safety_score,
            'liquidity': liquidity,
            'holder_count': holder_count,
            'volume_24h': volume_24h,
            'contract_verified': random.choice([True, False]),
            'mint_enabled': random.choice([True, False]),
            'liquidity_locked': random.choice([True, False]),
            'has_raydium_pool': random.choice([True, False]),
            'honeypot_risk': random.choice(['low', 'medium', 'high']),
            'ownership_renounced': random.choice([True, False])
        }

    def validate_token_safety(self, token_address: str) -> Dict[str, Any]:
        """Validate token safety using multiple sources"""
        try:
            if self.use_mock_data:
                mock_data = self._generate_mock_safety_data(token_address)
                token_info = {
                    'holder_count': mock_data['holder_count'],
                    'volume24h': mock_data['volume_24h'],
                    'liquidity': mock_data['liquidity']
                }
                raydium_info = {
                    'pool_exists': mock_data['has_raydium_pool'],
                    'liquidity_locked': mock_data['liquidity_locked']
                }
                sniffer_info = {
                    'safety_score': mock_data['safety_score'],
                    'contract_verified': mock_data['contract_verified'],
                    'mint_enabled': mock_data['mint_enabled'],
                    'honeypot_risk': mock_data['honeypot_risk']
                }
            else:
                # Get token information from Solscan
                token_info = self._get_token_info(token_address)
                if not token_info:
                    return {'error': 'Failed to fetch token information', 'passed': False}

                # Check Raydium pool
                raydium_info = self._check_raydium_pool(token_address)

                # Get SolanaSniffer safety score
                sniffer_info = self._check_solana_sniffer(token_address)

            # Combine safety checks
            safety_result = {
                'timestamp': datetime.utcnow().isoformat(),
                'token_address': token_address,
                'passed': True,
                'warnings': [],
                'risk_factors': [],
                'metrics': {
                    'holder_count': token_info.get('holder_count', 0),
                    'total_volume_24h': token_info.get('volume24h', 0),
                    'liquidity': token_info.get('liquidity', 0),
                    'raydium_pool_exists': raydium_info.get('pool_exists', False),
                    'liquidity_locked': raydium_info.get('liquidity_locked', False),
                    'safety_score': sniffer_info.get('safety_score', 0),
                    'contract_verified': sniffer_info.get('contract_verified', False),
                    'mint_enabled': sniffer_info.get('mint_enabled', True)
                }
            }

            # Apply safety checks
            if token_info.get('holder_count', 0) < self.safety_thresholds['min_holder_count']:
                safety_result['warnings'].append('Low holder count')

            if token_info.get('volume24h', 0) < self.safety_thresholds['min_volume_24h']:
                safety_result['warnings'].append('Low trading volume')

            if token_info.get('liquidity', 0) < self.safety_thresholds['min_liquidity']:
                safety_result['risk_factors'].append('Insufficient liquidity')
                safety_result['passed'] = False

            if not raydium_info.get('liquidity_locked', False):
                safety_result['risk_factors'].append('Liquidity not locked')
                safety_result['passed'] = False

            if sniffer_info.get('safety_score', 0) < self.safety_thresholds['min_safety_score']:
                safety_result['risk_factors'].append('Low safety score')
                safety_result['passed'] = False

            if sniffer_info.get('mint_enabled', True):
                safety_result['risk_factors'].append('Minting is enabled')
                safety_result['passed'] = False

            if not sniffer_info.get('contract_verified', False):
                safety_result['risk_factors'].append('Contract not verified')

            return safety_result

        except Exception as e:
            logger.error(f"Error validating token safety: {str(e)}")
            if self.use_mock_data:
                # Return mock data even in case of error
                mock_data = self._generate_mock_safety_data(token_address)
                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'token_address': token_address,
                    'passed': mock_data['safety_score'] >= self.safety_thresholds['min_safety_score'],
                    'warnings': [],
                    'risk_factors': ['Mock data - for testing only'],
                    'metrics': mock_data
                }
            return {'error': str(e), 'passed': False}

    def _get_token_info(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get token information from Solscan"""
        try:
            response = requests.get(
                f"{self.base_url}/token/{token_address}"
            )

            if response.status_code != 200:
                logger.error(f"Solscan API error: {response.status_code} - {response.text}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"Error fetching token info: {str(e)}")
            return None

    def _check_raydium_pool(self, token_address: str) -> Dict[str, Any]:
        """Check Raydium pool information"""
        try:
            # Use Raydium public API to check pool existence and liquidity lock
            response = requests.get(
                f"https://api.raydium.io/v2/main/pool/{token_address}"
            )

            if response.status_code != 200:
                return {
                    'pool_exists': False,
                    'liquidity_locked': False,
                    'error': f"Raydium API error: {response.status_code}"
                }

            pool_data = response.json()
            return {
                'pool_exists': True,
                'liquidity_locked': pool_data.get('liquidity_locked', False),
                'pool_size': pool_data.get('liquidity', 0),
                'volume_24h': pool_data.get('volume24h', 0)
            }

        except Exception as e:
            logger.error(f"Error checking Raydium pool: {str(e)}")
            return {
                'pool_exists': False,
                'liquidity_locked': False,
                'error': str(e)
            }

    def _check_solana_sniffer(self, token_address: str) -> Dict[str, Any]:
        """Check token safety using SolanaSniffer public API"""
        try:
            # Using SolanaSniffer public API endpoint
            response = requests.get(
                f"https://api.solanasniffer.com/v1/token/{token_address}/safety"
            )

            if response.status_code != 200:
                return {
                    'safety_score': 0,
                    'contract_verified': False,
                    'mint_enabled': True,
                    'error': f"SolanaSniffer API error: {response.status_code}"
                }

            data = response.json()
            return {
                'safety_score': data.get('safety_score', 0),
                'contract_verified': data.get('contract_verified', False),
                'mint_enabled': data.get('mint_enabled', True),
                'honeypot_risk': data.get('honeypot_risk', 'high'),
                'ownership_renounced': data.get('ownership_renounced', False)
            }

        except Exception as e:
            logger.error(f"Error checking SolanaSniffer: {str(e)}")
            return {
                'safety_score': 0,
                'contract_verified': False,
                'mint_enabled': True,
                'error': str(e)
            }

    def update_safety_thresholds(self, new_thresholds: Dict[str, float]) -> bool:
        """Update safety validation thresholds"""
        try:
            for key, value in new_thresholds.items():
                if key in self.safety_thresholds:
                    self.safety_thresholds[key] = float(value)

            logger.info(f"Updated safety thresholds: {self.safety_thresholds}")
            return True
        except Exception as e:
            logger.error(f"Error updating safety thresholds: {str(e)}")
            return False