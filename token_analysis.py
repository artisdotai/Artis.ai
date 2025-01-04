from web3 import Web3
from solana.rpc.api import Client
import base58
from datetime import datetime
import requests
import logging
from models import TokenMetrics, db
from sentiment_analyzer import SentimentAnalyzer
from technical_analysis import TechnicalAnalyzer
import time

logger = logging.getLogger(__name__)

class TokenAnalyzer:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider('https://bsc-dataseed.binance.org/'))
        self.solana_client = Client("https://api.mainnet-beta.solana.com")
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.last_api_call = 0
        self.api_call_delay = 1.0  # Delay between API calls in seconds

        # Add Raydium DEX configuration
        self.raydium_config = {
            'program_id': '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',
            'amm_authority': '5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1',
            'pool_program_id': '27haf8L6oxUeXrHrgEgsexjSY5hbVUWEmvv9Nyxg8vQv'
        }

    def _get_raydium_pool_info(self, token_address):
        """Get Raydium pool information for a token"""
        try:
            # In production, this would query the Raydium pool
            # For now, return simulated data
            if token_address.startswith('So'):  # Solana address format
                return {
                    'liquidity': 75000,
                    'volume_24h': 35000,
                    'price': 1.0
                }
            return None
        except Exception as e:
            logger.error(f"Error getting Raydium pool info: {str(e)}")
            return None

    def get_token_metrics(self, token_address):
        try:
            current_time = time.time()
            if current_time - self.last_api_call < self.api_call_delay:
                time.sleep(self.api_call_delay - (current_time - self.last_api_call))
            self.last_api_call = time.time()

            # Check if it's a Solana token
            if token_address.startswith('So'):  # Solana address format
                pool_info = self._get_raydium_pool_info(token_address)
                if pool_info:
                    social_score = self.sentiment_analyzer.analyze_token(token_address)
                    technical_indicators = self.technical_analyzer.get_indicators(token_address)

                    metrics = {
                        'liquidity': pool_info['liquidity'],
                        'volume_24h': pool_info['volume_24h'],
                        'social_score': social_score,
                        'technical_indicators': technical_indicators,
                        'potential_score': self._calculate_potential_score(
                            pool_info['liquidity'],
                            pool_info['volume_24h'],
                            social_score,
                            technical_indicators
                        ),
                        'chain': 'solana'
                    }

                    self._update_token_metrics(token_address, metrics)
                    return metrics

            if token_address in [
                '0x0000000000000000000000000000000000000000',
                '0x0000000000000000000000000000000000000001',
                '0x0000000000000000000000000000000000000002',
                '0x0000000000000000000000000000000000000003',
                '0x0000000000000000000000000000000000000004',
                '0x0000000000000000000000000000000000000005'  
            ]:
                return self._get_test_token_metrics(token_address)

            response = requests.get(
                f"https://api.coingecko.com/api/v3/simple/token_price/binance-smart-chain",
                params={
                    "contract_addresses": token_address,
                    "vs_currencies": "usd",
                    "include_24hr_vol": True,
                    "include_market_cap": True
                }
            )

            if response.status_code == 429:
                logger.warning("CoinGecko API rate limit reached, using test data")
                return self._get_test_token_metrics(token_address)

            data = response.json()
            liquidity = self._get_pair_liquidity(token_address)
            social_score = self.sentiment_analyzer.analyze_token(token_address)
            technical_indicators = self.technical_analyzer.get_indicators(token_address)
            volume_24h = data[token_address].get('usd_24h_vol', 0)
            potential_score = self._calculate_potential_score(
                liquidity,
                volume_24h,
                social_score,
                technical_indicators
            )

            metrics = {
                'liquidity': liquidity,
                'volume_24h': volume_24h,
                'social_score': social_score,
                'technical_indicators': technical_indicators,
                'potential_score': potential_score
            }

            self._update_token_metrics(token_address, metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing token {token_address}: {str(e)}")
            return self._get_test_token_metrics(token_address)

    def _get_test_token_metrics(self, token_address):
        test_configs = {
            '0x0000000000000000000000000000000000000000': {
                'chain': 'bsc',
                'liquidity': 100000,
                'volume_24h': 50000,
                'social_score': 7.5,
                'potential_score': 8.0,
                'symbol': 'TEST1',
                'name': 'Test Token 1 (BSC)'
            },
            '0x0000000000000000000000000000000000000001': {
                'chain': 'eth',
                'liquidity': 200000,
                'volume_24h': 100000,
                'social_score': 8.0,
                'potential_score': 8.5,
                'symbol': 'TEST2',
                'name': 'Test Token 2 (ETH)'
            },
            '0x0000000000000000000000000000000000000002': {
                'chain': 'polygon',
                'liquidity': 75000,
                'volume_24h': 35000,
                'social_score': 7.0,
                'potential_score': 7.8,
                'symbol': 'TEST3',
                'name': 'Test Token 3 (POLYGON)'
            },
            '0x0000000000000000000000000000000000000003': {
                'chain': 'arb',
                'liquidity': 80000,
                'volume_24h': 40000,
                'social_score': 7.2,
                'potential_score': 7.9,
                'symbol': 'TEST4',
                'name': 'Test Token 4 (ARB)'
            },
            'SoLTest1111111111111111111111111111111111111': {
                'chain': 'solana',
                'liquidity': 70000,
                'volume_24h': 35000,
                'social_score': 7.0,
                'potential_score': 7.7,
                'symbol': 'TEST6',
                'name': 'Test Token 6 (SOLANA)'
            }
        }

        config = test_configs.get(token_address, test_configs['SoLTest1111111111111111111111111111111111111'])

        return {
            'liquidity': config['liquidity'],
            'volume_24h': config['volume_24h'],
            'social_score': config['social_score'],
            'potential_score': config['potential_score'],
            'chain': config['chain'],
            'symbol': config['symbol'],
            'name': config['name'],
            'technical_indicators': {
                'rsi': {'value': 50, 'signal': 'neutral'}, 
                'macd': {'interpretation': 'neutral'}, 
                'bollinger': {'signal': 'neutral'}, 
                'vwap': {'signal': 'neutral'}, 
                'trend_strength': {'strength': 'neutral'}
            }
        }

    def find_opportunities(self):
        opportunities = []
        tokens = self._get_trending_tokens()

        for token in tokens:
            metrics = self.get_token_metrics(token['address'])
            if metrics and metrics['potential_score'] > 7.0:
                opportunities.append({
                    'address': token['address'],
                    'chain': metrics.get('chain', 'bsc'),  
                    'metrics': metrics
                })

            self._update_token_metrics(token['address'], metrics)

        return opportunities

    def _get_trending_tokens(self):
        try:
            response = requests.get(
                "https://api.dextools.io/v1/trending",
                params={"chain": "bsc"},
                headers={"X-API-Key": "demo"}
            )

            if response.status_code == 200:
                data = response.json()
                return [
                    {
                        'address': token['address'],
                        'symbol': token['symbol'],
                        'name': token['name']
                    }
                    for token in data.get('data', [])[:10]
                ]

            logger.warning("Using test tokens due to API unavailability")
            return [
                {
                    'address': '0x0000000000000000000000000000000000000000',
                    'symbol': 'TEST1',
                    'name': 'Test Token 1 (BSC)'
                },
                {
                    'address': '0x0000000000000000000000000000000000000001',
                    'symbol': 'TEST2',
                    'name': 'Test Token 2 (ETH)'
                },
                {
                    'address': '0x0000000000000000000000000000000000000002',
                    'symbol': 'TEST3',
                    'name': 'Test Token 3 (POLYGON)'
                },
                {
                    'address': '0x0000000000000000000000000000000000000003',
                    'symbol': 'TEST4',
                    'name': 'Test Token 4 (ARB)'
                },
                {
                    'address': 'SoLTest1111111111111111111111111111111111111',
                    'symbol': 'TEST6',
                    'name': 'Test Token 6 (SOLANA)'
                }
            ]
        except Exception as e:
            logger.error(f"Error fetching trending tokens: {str(e)}")
            return []

    def _get_pair_liquidity(self, token_address):
        try:
            # Check if it's a Solana token
            if token_address.startswith('So'):
                pool_info = self._get_raydium_pool_info(token_address)
                return pool_info['liquidity'] if pool_info else 0

            return 100000  # Default test value for other chains
        except Exception as e:
            logger.error(f"Error getting liquidity for {token_address}: {str(e)}")
            return 0

    def _calculate_potential_score(self, liquidity, volume, social_score, technical_indicators):
        if liquidity == 0 or not technical_indicators:
            return 0

        volume_score = min(volume / liquidity, 10) * 0.15
        liquidity_score = min(liquidity / 100000, 10) * 0.15
        social_score = min(social_score, 10) * 0.2
        technical_score = self._calculate_technical_score(technical_indicators)

        return volume_score + liquidity_score + social_score + technical_score

    def _calculate_technical_score(self, indicators):
        try:
            score = 0
            max_score = 5  
            if 'rsi' in indicators and indicators['rsi']:
                rsi = indicators['rsi']['value']
                if indicators['rsi']['signal'] == 'oversold':
                    score += 0.75  
                elif indicators['rsi']['signal'] == 'overbought':
                    score += 0.15  
                else:
                    score += 0.45  

            if 'macd' in indicators and indicators['macd']:
                if indicators['macd']['interpretation'] == 'strong_buy':
                    score += 0.75
                elif indicators['macd']['interpretation'] == 'buy':
                    score += 0.6
                elif indicators['macd']['interpretation'] == 'sell':
                    score += 0.3
                elif indicators['macd']['interpretation'] == 'strong_sell':
                    score += 0.15

            if 'bollinger' in indicators and indicators['bollinger']:
                if indicators['bollinger']['signal'] == 'strong_buy':
                    score += 0.5
                elif indicators['bollinger']['signal'] == 'buy':
                    score += 0.4
                elif indicators['bollinger']['signal'] == 'neutral':
                    score += 0.25
                else:
                    score += 0.1

            if 'vwap' in indicators and indicators['vwap']:
                if indicators['vwap']['signal'] == 'buy':
                    score += 0.25
                elif indicators['vwap']['signal'] == 'neutral':
                    score += 0.15
                else:
                    score += 0.05

            if 'trend_strength' in indicators and indicators['trend_strength']:
                if indicators['trend_strength']['strength'] == 'strong':
                    score += 0.25
                elif indicators['trend_strength']['strength'] == 'moderate':
                    score += 0.15
                else:
                    score += 0.05

            return (score / 2) * max_score  

        except Exception as e:
            logger.error(f"Error calculating technical score: {str(e)}")
            return 0

    def _update_token_metrics(self, address, metrics):
        try:
            token_metrics = TokenMetrics.query.filter_by(token_address=address).first()
            if not token_metrics:
                token_metrics = TokenMetrics(token_address=address)

            token_metrics.liquidity = metrics['liquidity']
            token_metrics.volume_24h = metrics['volume_24h']
            token_metrics.social_score = metrics['social_score']
            token_metrics.potential_score = metrics['potential_score']
            token_metrics.technical_indicators = str(metrics['technical_indicators']) 

            db.session.add(token_metrics)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error updating token metrics: {str(e)}")
            db.session.rollback()