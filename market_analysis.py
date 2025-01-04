"""
Market Analysis Module
Responsible for analyzing tokens across multiple chains and providing scoring metrics
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import requests
from web3 import Web3
import pandas as pd
import numpy as np
from technical_analysis import TechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.supported_chains = {
            'bsc': {'rpc': 'https://bsc-dataseed.binance.org/'},
            'eth': {'rpc': 'https://mainnet.infura.io/v3/your-infura-key'},
            'polygon': {'rpc': 'https://polygon-rpc.com'},
            'arb': {'rpc': 'https://arb1.arbitrum.io/rpc'},
            'avax': {'rpc': 'https://api.avax.network/ext/bc/C/rpc'},
            'solana': {'rpc': 'https://api.mainnet-beta.solana.com'}
        }
        
        # Initialize Web3 connections
        self.web3_connections = {}
        for chain, config in self.supported_chains.items():
            if chain != 'solana':  # Solana uses different connection method
                try:
                    self.web3_connections[chain] = Web3(Web3.HTTPProvider(config['rpc']))
                except Exception as e:
                    logger.error(f"Failed to initialize Web3 for {chain}: {str(e)}")

        # Scoring weights
        self.weights = {
            'liquidity': 0.3,
            'volume': 0.2,
            'price_action': 0.15,
            'holder_metrics': 0.15,
            'sentiment': 0.2
        }

    async def analyze_token(self, token_address: str, chain: str) -> Dict[str, Any]:
        """Analyze a token and return comprehensive metrics"""
        try:
            logger.info(f"Analyzing token {token_address} on {chain}")
            
            # Basic token metrics
            metrics = await self._get_token_metrics(token_address, chain)
            if not metrics:
                return {'error': 'Failed to get token metrics'}

            # Technical analysis
            technical_scores = self.technical_analyzer.analyze_token(token_address, chain)
            
            # Sentiment analysis
            sentiment_scores = self.sentiment_analyzer.analyze_token(token_address)
            
            # Calculate final score
            final_score = self._calculate_token_score(metrics, technical_scores, sentiment_scores)
            
            return {
                'token_address': token_address,
                'chain': chain,
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': metrics,
                'technical_analysis': technical_scores,
                'sentiment': sentiment_scores,
                'final_score': final_score,
                'recommendation': self._generate_recommendation(final_score)
            }

        except Exception as e:
            logger.error(f"Error analyzing token {token_address}: {str(e)}")
            return {'error': str(e)}

    async def _get_token_metrics(self, token_address: str, chain: str) -> Optional[Dict[str, Any]]:
        """Get basic token metrics like liquidity, volume, holders etc."""
        try:
            if chain == 'solana':
                return await self._get_solana_metrics(token_address)
            
            web3 = self.web3_connections.get(chain)
            if not web3:
                return None

            # Get token contract
            token_contract = web3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=self._get_token_abi()
            )

            # Fetch basic metrics
            metrics = {
                'total_supply': token_contract.functions.totalSupply().call(),
                'holders': self._get_holder_count(token_address, chain),
                'liquidity': await self._get_liquidity(token_address, chain),
                'volume_24h': await self._get_24h_volume(token_address, chain),
                'market_cap': await self._get_market_cap(token_address, chain)
            }

            return metrics

        except Exception as e:
            logger.error(f"Error getting token metrics: {str(e)}")
            return None

    def _calculate_token_score(
        self, 
        metrics: Dict[str, Any],
        technical_scores: Dict[str, Any],
        sentiment_scores: Dict[str, Any]
    ) -> float:
        """Calculate final token score based on all metrics"""
        try:
            # Normalize metrics
            normalized_metrics = {
                'liquidity': self._normalize_value(metrics['liquidity'], 0, 1000000),
                'volume': self._normalize_value(metrics['volume_24h'], 0, 500000),
                'holder_score': self._normalize_value(metrics['holders'], 0, 10000),
                'price_action': technical_scores.get('overall_score', 0.5),
                'sentiment': sentiment_scores.get('overall_score', 0.5)
            }

            # Calculate weighted score
            final_score = sum(
                normalized_metrics[key] * self.weights[key] 
                for key in self.weights
                if key in normalized_metrics
            )

            return round(final_score * 10, 2)  # Convert to 0-10 scale

        except Exception as e:
            logger.error(f"Error calculating token score: {str(e)}")
            return 0.0

    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to 0-1 range"""
        try:
            if value <= min_val:
                return 0.0
            if value >= max_val:
                return 1.0
            return (value - min_val) / (max_val - min_val)
        except Exception:
            return 0.0

    def _generate_recommendation(self, score: float) -> Dict[str, Any]:
        """Generate trading recommendation based on score"""
        if score >= 8.0:
            return {
                'action': 'STRONG_BUY',
                'confidence': 'HIGH',
                'risk_level': 'LOW'
            }
        elif score >= 6.5:
            return {
                'action': 'BUY',
                'confidence': 'MEDIUM',
                'risk_level': 'MEDIUM'
            }
        elif score >= 5.0:
            return {
                'action': 'HOLD',
                'confidence': 'MEDIUM',
                'risk_level': 'MEDIUM'
            }
        else:
            return {
                'action': 'AVOID',
                'confidence': 'HIGH',
                'risk_level': 'HIGH'
            }

    @staticmethod
    def _get_token_abi():
        """Return standard ERC20 ABI"""
        return [
            {
                "constant": True,
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"name": "", "type": "uint256"}],
                "payable": False,
                "stateMutability": "view",
                "type": "function"
            },
            # Add other necessary ABI elements
        ]

    async def _get_liquidity(self, token_address: str, chain: str) -> float:
        """Get token liquidity from DEX"""
        # Implementation for getting liquidity from relevant DEX
        return 0.0

    async def _get_24h_volume(self, token_address: str, chain: str) -> float:
        """Get 24h trading volume"""
        # Implementation for getting 24h volume
        return 0.0

    async def _get_market_cap(self, token_address: str, chain: str) -> float:
        """Calculate market cap"""
        # Implementation for calculating market cap
        return 0.0

    def _get_holder_count(self, token_address: str, chain: str) -> int:
        """Get number of token holders"""
        # Implementation for getting holder count
        return 0

    async def _get_solana_metrics(self, token_address: str) -> Dict[str, Any]:
        """Get metrics for Solana tokens"""
        # Implementation for Solana metrics
        return {}
