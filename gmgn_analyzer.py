"""GMGN Swap API integration with improved error handling"""
from api_manager import api_manager, with_fallback
import logging
import os
import random
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Any
import requests
from extensions import db
from models import TokenMetrics

logger = logging.getLogger(__name__)

class GMGNAPIError(Exception):
    """Custom exception for GMGN API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class GMGNAnalyzer:
    """GMGN Swap API integration with enhanced error handling"""

    def __init__(self, use_mock_data: bool = True):
        """Initialize GMGN analyzer with robust error handling"""
        try:
            self.base_url = "https://gmgn-swap.xyz/api"
            self.use_mock_data = use_mock_data
            self.initialized = True

            # Configure logging
            logging.basicConfig(level=logging.INFO)
            logger.info("Initialized GMGN analyzer with enhanced error handling")

        except Exception as e:
            logger.error(f"Failed to initialize GMGN analyzer: {str(e)}")
            self.initialized = False
            raise

    def _make_api_request(self, endpoint: str, method: str = 'GET', 
                         params: Optional[Dict] = None) -> Dict:
        """Make API request using the API manager"""
        try:
            return api_manager.call_api(
                api_name='gmgn',
                endpoint=f"{self.base_url}/{endpoint.lstrip('/')}",
                method=method,
                params=params
            )
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise GMGNAPIError(f"Request failed: {str(e)}")

    @with_fallback(fallback_function=lambda *args, **kwargs: [])
    def get_historical_data(self, days: int = 30) -> List[Dict]:
        """Fetch historical data with enhanced error handling"""
        try:
            if not self.initialized:
                raise GMGNAPIError("Analyzer not properly initialized")

            if self.use_mock_data:
                logger.info("Using mock data for historical data")
                return self._generate_mock_token_data(count=20)

            response = self._make_api_request(
                'tokens/history',
                params={
                    'start_time': int((datetime.utcnow() - timedelta(days=days)).timestamp()),
                    'chain': 'solana'
                }
            )

            tokens_data = response.get('tokens', [])
            self._process_token_data(tokens_data)
            return tokens_data

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return self._generate_mock_token_data(count=10)

    def _process_token_data(self, tokens_data: List[Dict]) -> None:
        """Process and store token data with error handling"""
        try:
            for token_data in tokens_data:
                try:
                    enriched_data = self._enrich_token_data(token_data)
                    self._store_token_metrics(enriched_data)
                except Exception as token_error:
                    logger.error(f"Error processing token {token_data.get('address', 'unknown')}: {str(token_error)}")
                    continue

            db.session.commit()
            logger.info(f"Successfully processed {len(tokens_data)} tokens")

        except Exception as e:
            logger.error(f"Error in bulk token processing: {str(e)}")
            db.session.rollback()

    def _store_token_metrics(self, token_data: Dict) -> None:
        """Store token metrics with error handling"""
        try:
            token_metrics = TokenMetrics(
                address=token_data['address'],
                chain='solana',
                timestamp=datetime.utcnow(),
                price=token_data.get('price'),
                volume=token_data.get('volume24h'),
                liquidity=token_data.get('liquidity'),
                holder_count=token_data.get('holders'),
                safety_score=token_data.get('safety_score'),
                liquidity_locked=token_data.get('is_liquidity_locked'),
                mint_enabled=token_data.get('is_mint_enabled'),
                contract_verified=token_data.get('is_contract_verified'),
                raydium_pool_exists=token_data.get('has_raydium_pool'),
                holder_analysis=token_data.get('holder_analysis'),
                risk_score=self._calculate_risk_score(token_data),
                potential_score=self._calculate_potential_score(token_data)
            )

            db.session.merge(token_metrics)

        except Exception as e:
            logger.error(f"Error storing token metrics: {str(e)}")
            raise

    def _enrich_token_data(self, token_data: Dict) -> Dict:
        """Enrich token data with enhanced error handling"""
        try:
            enriched_data = token_data.copy()

            # Calculate price changes
            if 'price_history' in token_data:
                try:
                    price_changes = self._calculate_price_changes(token_data['price_history'])
                    enriched_data.update(price_changes)
                except Exception as price_error:
                    logger.error(f"Error calculating price changes: {str(price_error)}")
                    enriched_data.update({'price_change_24h': 0, 'price_trend': 'unknown'})

            # Calculate market cap
            try:
                if token_data.get('price') and token_data.get('total_supply'):
                    enriched_data['market_cap'] = float(token_data['price']) * float(token_data['total_supply'])
            except Exception as market_cap_error:
                logger.error(f"Error calculating market cap: {str(market_cap_error)}")
                enriched_data['market_cap'] = 0

            # Analyze holder distribution
            if token_data.get('holder_metrics'):
                try:
                    holder_analysis = self._analyze_holder_distribution(token_data['holder_metrics'])
                    enriched_data['holder_analysis'] = holder_analysis
                except Exception as holder_error:
                    logger.error(f"Error analyzing holder distribution: {str(holder_error)}")
                    enriched_data['holder_analysis'] = {}

            return enriched_data

        except Exception as e:
            logger.error(f"Error enriching token data: {str(e)}")
            return token_data

    def _calculate_risk_score(self, token_data: Dict) -> float:
        """Calculate risk score with error handling"""
        try:
            risk_factors = []

            # Check key risk factors
            if not token_data.get('is_liquidity_locked', False):
                risk_factors.append(0.3)
            if token_data.get('is_mint_enabled', True):
                risk_factors.append(0.25)
            if not token_data.get('is_contract_verified', False):
                risk_factors.append(0.2)
            if not token_data.get('has_raydium_pool', False):
                risk_factors.append(0.15)

            # Additional risk factors
            holder_analysis = token_data.get('holder_analysis', {})
            if holder_analysis.get('holder_concentration', 0) > 0.5:
                risk_factors.append(0.1)

            return min(1.0, sum(risk_factors))

        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 0.75

    def analyze_token_potential(self, token_address: str) -> Dict[str, Any]:
        """Analyze token potential with comprehensive error handling"""
        try:
            token_details = self.get_token_details(token_address)
            if not token_details:
                return {'error': 'Failed to fetch token details'}

            historical_metrics = self._get_historical_metrics()

            analysis_result = {
                'timestamp': datetime.utcnow().isoformat(),
                'token_address': token_address,
                'metrics': {
                    'current_price': token_details.get('price'),
                    'current_volume': token_details.get('volume24h'),
                    'current_liquidity': token_details.get('liquidity'),
                    'holder_count': token_details.get('holders'),
                    'price_change_24h': token_details.get('price_change_24h'),
                    'holder_distribution': token_details.get('holder_analysis', {})
                },
                'risk_assessment': {
                    'safety_score': token_details.get('safety_score', 0),
                    'risk_level': self._calculate_risk_level(token_details)
                },
                'comparison': self._compare_with_historical(token_details, historical_metrics),
                'potential_assessment': {
                    'success_probability': self._calculate_success_probability(token_details),
                    'growth_potential': self._calculate_growth_potential(token_details, historical_metrics)
                }
            }

            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing token potential: {str(e)}")
            return {'error': str(e)}

    def _get_historical_metrics(self) -> List[TokenMetrics]:
        """Get historical metrics with error handling"""
        try:
            return TokenMetrics.query.filter(
                TokenMetrics.chain == 'solana',
                TokenMetrics.potential_score >= 0.7
            ).all()
        except Exception as e:
            logger.error(f"Error fetching historical metrics: {str(e)}")
            return []

    def _compare_with_historical(self, token_details: Dict, 
                               historical_metrics: List[TokenMetrics]) -> Dict:
        """Compare with historical data with error handling"""
        try:
            if not historical_metrics:
                return {
                    'avg_successful_price': 0,
                    'avg_successful_volume': 0,
                    'avg_successful_liquidity': 0
                }

            return {
                'avg_successful_price': sum(t.price for t in historical_metrics) / len(historical_metrics),
                'avg_successful_volume': sum(t.volume for t in historical_metrics) / len(historical_metrics),
                'avg_successful_liquidity': sum(t.liquidity for t in historical_metrics) / len(historical_metrics)
            }

        except Exception as e:
            logger.error(f"Error comparing with historical data: {str(e)}")
            return {}

    def _calculate_success_probability(self, token_data: Dict) -> float:
        """Calculate success probability with error handling"""
        try:
            factors = []

            if token_data.get('safety_score', 0) > 70:
                factors.append(0.3)
            if token_data.get('volume24h', 0) > 10000:
                factors.append(0.2)
            if token_data.get('holders', 0) > 200:
                factors.append(0.2)
            if token_data.get('is_contract_verified', False):
                factors.append(0.15)
            if token_data.get('has_raydium_pool', False):
                factors.append(0.15)

            return min(1.0, sum(factors))

        except Exception as e:
            logger.error(f"Error calculating success probability: {str(e)}")
            return 0.0


    def _generate_mock_token_data(self, count: int = 10) -> List[Dict]:
        """Generate realistic mock token data for testing"""
        try:
            mock_tokens = []
            current_time = datetime.utcnow()

            for i in range(count):
                # Generate realistic token address for Solana
                address = ''.join(random.choices('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz', k=44))

                # Generate realistic metrics
                price = random.uniform(0.00001, 0.1)
                volume = random.uniform(5000, 100000)
                liquidity = random.uniform(10000, 500000)
                holders = random.randint(150, 5000)

                # Calculate market cap
                total_supply = random.uniform(1000000, 100000000)
                market_cap = price * total_supply

                # Generate realistic mock data
                token = {
                    'address': address,
                    'name': f'Mock Token {i+1}',
                    'symbol': f'MOCK{i+1}',
                    'price': price,
                    'volume24h': volume,
                    'liquidity': liquidity,
                    'holders': holders,
                    'market_cap': market_cap,
                    'total_supply': total_supply,
                    'safety_score': random.randint(70, 95),
                    'is_liquidity_locked': random.choice([True, True, False]),  # Bias towards safer tokens
                    'is_mint_enabled': random.choice([False, False, True]),  # Bias towards safer tokens
                    'is_contract_verified': True,  # All test tokens verified for safety
                    'has_raydium_pool': True,  # All test tokens have Raydium pool
                    'launch_time': (current_time - timedelta(hours=random.randint(1, 24))).isoformat(),
                    'price_history': self._generate_mock_price_history(price),
                    'holder_metrics': {
                        'total_holders': holders,
                        'top_10_holdings': holders * random.uniform(0.15, 0.35),  # Realistic distribution
                        'unique_holders': holders * random.uniform(0.9, 0.98)  # Account for duplicates
                    }
                }

                # Add volatility metrics
                token['price_change_24h'] = random.uniform(-15, 30)
                token['volume_change_24h'] = random.uniform(-20, 40)

                mock_tokens.append(token)

            logger.info(f"Generated {len(mock_tokens)} mock tokens with realistic metrics")
            return mock_tokens

        except Exception as e:
            logger.error(f"Error generating mock token data: {str(e)}")
            return []

    def _generate_mock_price_history(self, current_price: float) -> List[Dict]:
        """Generate realistic price history data"""
        try:
            history = []
            base_price = current_price * random.uniform(0.7, 1.3)
            timestamps = [
                datetime.utcnow() - timedelta(hours=x)
                for x in range(24, -1, -1)
            ]

            for timestamp in timestamps:
                # Add some realistic volatility
                price_change = random.uniform(-0.15, 0.15)
                price = base_price * (1 + price_change)
                base_price = price  # Use this as the base for next iteration

                history.append({
                    'timestamp': int(timestamp.timestamp()),
                    'price': price
                })

            return history

        except Exception as e:
            logger.error(f"Error generating mock price history: {str(e)}")
            return []

    def _calculate_price_changes(self, price_history: List[Dict]) -> Dict:
        """Calculate price change percentages"""
        try:
            if not price_history or len(price_history) < 2:
                return {}

            current_price = float(price_history[-1]['price'])
            day_ago_price = float(price_history[0]['price'])

            price_change_24h = ((current_price - day_ago_price) / day_ago_price) * 100 if day_ago_price else 0

            return {
                'price_change_24h': price_change_24h,
                'price_trend': 'up' if price_change_24h > 0 else 'down'
            }
        except Exception as e:
            logger.error(f"Error calculating price changes: {str(e)}")
            return {}

    def _analyze_holder_distribution(self, holder_metrics: Dict) -> Dict:
        """Analyze holder distribution patterns"""
        try:
            total_holders = holder_metrics.get('total_holders', 0)
            if not total_holders:
                return {}

            # Calculate concentration metrics
            top_10_holders = holder_metrics.get('top_10_holdings', 0)
            concentration_ratio = (top_10_holders / total_holders) if total_holders else 0

            return {
                'holder_concentration': concentration_ratio,
                'distribution_quality': 'good' if concentration_ratio < 0.5 else 'concerning',
                'unique_holders': total_holders
            }
        except Exception as e:
            logger.error(f"Error analyzing holder distribution: {str(e)}")
            return {}

    @lru_cache(maxsize=100)
    def get_token_details(self, token_address: str) -> Optional[Dict]:
        """Get detailed information for a specific token"""
        try:
            if self.use_mock_data:
                # Generate consistent mock data for a specific address
                mock_data = self._generate_mock_token_data(count=1)[0]
                mock_data['address'] = token_address
                return self._enrich_token_data(mock_data)

            response = self._make_api_request(f"tokens/{token_address}")
            return self._enrich_token_data(response)

        except GMGNAPIError as api_error:
            logger.error(f"API error fetching token details: {str(api_error)}")
            mock_data = self._generate_mock_token_data(count=1)[0]
            mock_data['address'] = token_address
            return self._enrich_token_data(mock_data)
        except Exception as e:
            logger.error(f"Error fetching token details: {str(e)}")
            return None

    def _calculate_risk_score(self, token_data: Dict) -> float:
        """Calculate risk score based on token metrics"""
        try:
            risk_factors = []

            # Check key risk factors
            if not token_data.get('is_liquidity_locked', False):
                risk_factors.append(0.3)  # 30% risk for unlocked liquidity
            if token_data.get('is_mint_enabled', True):
                risk_factors.append(0.25)  # 25% risk for enabled minting
            if not token_data.get('is_contract_verified', False):
                risk_factors.append(0.2)  # 20% risk for unverified contract
            if not token_data.get('has_raydium_pool', False):
                risk_factors.append(0.15)  # 15% risk for no Raydium pool

            # Additional risk factors
            holder_analysis = token_data.get('holder_analysis', {})
            if holder_analysis.get('holder_concentration', 0) > 0.5:
                risk_factors.append(0.1)  # 10% risk for high holder concentration

            # Calculate final risk score (0-1, higher means riskier)
            risk_score = sum(risk_factors)
            return min(1.0, risk_score)

        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 0.75  # Default to high risk on error

    def _calculate_potential_score(self, token_data: Dict) -> float:
        """Calculate potential score based on metrics"""
        try:
            potential_factors = []

            # Positive factors
            if token_data.get('is_liquidity_locked', False):
                potential_factors.append(0.3)  # 30% potential for locked liquidity
            if token_data.get('volume24h', 0) > 10000:  # 10k USD volume
                potential_factors.append(0.2)  # 20% potential for good volume
            if token_data.get('holders', 0) > 100:
                potential_factors.append(0.2)  # 20% potential for good holder count
            if token_data.get('is_contract_verified', False):
                potential_factors.append(0.15)  # 15% potential for verified contract
            if token_data.get('has_raydium_pool', False):
                potential_factors.append(0.15)  # 15% potential for Raydium presence

            # Additional potential factors
            price_changes = token_data.get('price_change_24h', 0)
            if price_changes > 10:  # 10% price increase
                potential_factors.append(0.1)  # 10% potential for positive price momentum

            holder_analysis = token_data.get('holder_analysis', {})
            if holder_analysis.get('distribution_quality') == 'good':
                potential_factors.append(0.1)  # 10% potential for good holder distribution

            # Calculate final potential score (0-1, higher means more potential)
            potential_score = sum(potential_factors)
            return min(1.0, potential_score)

        except Exception as e:
            logger.error(f"Error calculating potential score: {str(e)}")
            return 0.3  # Default to low potential on error

    def analyze_token_potential(self, token_address: str) -> Dict[str, Any]:
        """Analyze token potential based on historical patterns"""
        try:
            token_details = self.get_token_details(token_address)
            if not token_details:
                return {'error': 'Failed to fetch token details'}

            # Compare with historical successful tokens
            historical_metrics = self._get_historical_metrics()

            analysis_result = {
                'timestamp': datetime.utcnow().isoformat(),
                'token_address': token_address,
                'metrics': {
                    'current_price': token_details.get('price'),
                    'current_volume': token_details.get('volume24h'),
                    'current_liquidity': token_details.get('liquidity'),
                    'holder_count': token_details.get('holders'),
                    'price_change_24h': token_details.get('price_change_24h'),
                    'holder_distribution': token_details.get('holder_analysis', {})
                },
                'comparison': self._compare_with_historical(token_details, historical_metrics),
                'risk_assessment': {
                    'safety_score': token_details.get('safety_score', 0),
                    'risk_level': self._calculate_risk_level(token_details)
                },
                'potential_assessment': {
                    'success_probability': self._calculate_success_probability(token_details),
                    'growth_potential': self._calculate_growth_potential(token_details, historical_metrics)
                }
            }

            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing token potential: {str(e)}")
            return {'error': str(e)}

    def _calculate_risk_level(self, token_data: Dict) -> str:
        """Calculate risk level based on token metrics"""
        try:
            risk_factors = []

            if token_data.get('safety_score', 0) < 80:
                risk_factors.append('low_safety_score')
            if not token_data.get('is_liquidity_locked', False):
                risk_factors.append('unlocked_liquidity')
            if token_data.get('is_mint_enabled', True):
                risk_factors.append('mint_enabled')

            # Additional risk factors
            holder_analysis = token_data.get('holder_analysis', {})
            if holder_analysis.get('distribution_quality') == 'concerning':
                risk_factors.append('concentrated_holdings')

            if len(risk_factors) >= 2:
                return 'high'
            elif risk_factors:
                return 'medium'
            return 'low'

        except Exception as e:
            logger.error(f"Error calculating risk level: {str(e)}")
            return 'unknown'

    def _calculate_growth_potential(self, token_data: Dict, historical_metrics: List[TokenMetrics]) -> float:
        """Calculate growth potential compared to historical successful tokens"""
        try:
            if not historical_metrics:
                return 0.5

            # Weight different factors
            weights = {
                'holder_growth': 0.3,
                'volume_growth': 0.3,
                'liquidity_ratio': 0.2,
                'safety_score': 0.2
            }

            scores = {
                'holder_growth': min(1.0, token_data.get('holder_growth_rate', 0) / 0.1),  # 10% daily growth as baseline
                'volume_growth': min(1.0, token_data.get('volume_growth_rate', 0) / 0.2),  # 20% daily growth as baseline
                'liquidity_ratio': min(1.0, token_data.get('liquidity', 0) / token_data.get('market_cap', 1)),
                'safety_score': token_data.get('safety_score', 0) / 100
            }

            potential = sum(score * weights[factor] for factor, score in scores.items())
            return max(0, min(1, potential))

        except Exception as e:
            logger.error(f"Error calculating growth potential: {str(e)}")
            return 0.5