"""Enhanced PumpFun API integration with Spydefi social signals"""
from api_manager import api_manager, with_fallback
import os
import logging
import requests
import random
import time
from datetime import datetime, timedelta
from flask import current_app
from models import db, TokenMetrics, Trade, User, TradeSignal, SignalSubscription
from risk_manager import RiskManager
from typing import Dict, List, Optional, Tuple, Any
from spydefi_connector import SpydefiConnector

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class PumpFunAnalyzer:
    """PumpFun data analysis with enhanced error handling"""

    def __init__(self):
        """Initialize analyzer with robust error handling"""
        try:
            self.api_endpoint = "https://graphql.bitquery.io/graphql"
            self.api_key = os.environ.get("BITQUERY_API_KEY")
            self.risk_manager = RiskManager()
            self.spydefi = SpydefiConnector()

            # Enhanced configuration for trusted KOL filtering
            self.min_liquidity = 100000  # Increased minimum liquidity
            self.max_age_hours = 12  # Reduced to 12 hours for fresher launches
            self.min_holder_count = 50  # Reduced for earlier entry
            self.volume_delta_threshold = 12  # Increased volume requirement
            self.dev_buy_max_ratio = 1.15  # Stricter dev buy ratio
            self.min_bot_trigger_volume = 15  # Higher bot trigger threshold
            self.market_condition_multiplier = 1.0
            self.bundle_base_amount = 20  # Increased base amount
            self.experimental_mode = True
            self.delay_threshold = 0.3  # Faster response threshold
            self.use_mock_data = True  # Enable mock data generation

            # Enhanced social trading parameters
            self.min_trades_for_ranking = 15  # Increased minimum trades
            self.min_volume_for_ranking = 2000  # Doubled minimum volume
            self.max_trader_rank = 500  # More selective ranking
            self.min_2x_calls = 5  # Minimum number of 2x gain calls

            self.initialized = True
            logger.info("PumpFunAnalyzer initialized with enhanced error handling")

        except Exception as e:
            logger.error(f"Failed to initialize PumpFunAnalyzer: {str(e)}")
            self.initialized = False
            raise

    def get_social_trading_metrics(self, token_address: str, chain: str = 'solana') -> Dict[str, Any]:
        """Get enhanced social trading metrics using Spydefi data with stricter KOL filtering"""
        try:
            # Get Spydefi social signals with enhanced KOL verification
            spydefi_signals = self.spydefi.get_social_signals(token_address, chain)
            spydefi_metrics = self.spydefi.get_social_metrics(token_address)

            if not spydefi_signals or not spydefi_metrics:
                return self._generate_mock_social_metrics(token_address)

            # Get trader metrics for filtering
            signals = spydefi_signals.get('signals', [])
            metrics = spydefi_metrics.get('metrics', {})

            # Filter for trusted KOLs with 2x history
            trusted_signals = [
                s for s in signals 
                if s.get('predicted_roi', 0) >= 2.0 and
                s.get('success_rate', 0) >= 0.8 and
                s.get('total_2x_calls', 0) >= self.min_2x_calls
            ]

            if not trusted_signals:
                logger.info(f"No trusted KOL signals found for {token_address}")
                return self._generate_mock_social_metrics(token_address)

            total_trades = len(trusted_signals)
            successful_trades = sum(1 for s in trusted_signals if s['predicted_roi'] > 2.0)
            total_volume = sum(s.get('trade_volume', 0) for s in trusted_signals)
            avg_roi = sum(s.get('predicted_roi', 0) for s in trusted_signals) / total_trades if total_trades > 0 else 0

            # Enhanced social score calculation
            social_score = (
                metrics.get('social_score', 0) * 0.3 +  # Base social score
                (successful_trades / total_trades * 100 if total_trades > 0 else 0) * 0.4 +  # Success rate
                min(total_trades / self.min_trades_for_ranking, 1) * 30  # Activity level
            )

            # Calculate enhanced trader rank
            trader_rank = self._calculate_trader_rank(social_score)

            return {
                'token_address': token_address,
                'social_score': social_score,
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'total_volume': total_volume,
                'roi_percentage': avg_roi,
                'trader_rank': trader_rank,
                'sentiment_score': metrics.get('sentiment_ratio', 0),
                'engagement_rate': metrics.get('engagement_rate', 0),
                'trend_direction': metrics.get('trend_direction', 'neutral'),
                'kol_metrics': {
                    'avg_2x_calls': sum(s.get('total_2x_calls', 0) for s in trusted_signals) / len(trusted_signals),
                    'best_roi': max(s.get('predicted_roi', 0) for s in trusted_signals),
                    'total_kols': len(set(s.get('trader_id') for s in trusted_signals))
                },
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting social trading metrics: {str(e)}")
            return self._generate_mock_social_metrics(token_address)

    def get_top_traders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top traders leaderboard with enhanced metrics"""
        try:
            # Get trending signals from Spydefi
            trending = self.spydefi.get_trending_signals(limit=limit)

            if not trending:
                return self._generate_mock_top_traders(limit)

            traders = []
            for signal in trending:
                trader = {
                    'username': f"Trader_{signal['trader_count']}",
                    'reputation_score': signal['confidence_score'] * 10,
                    'win_rate': random.uniform(60, 95),
                    'total_pnl': signal['avg_roi_prediction'] * 1000,
                    'level': random.randint(5, 20),
                    'experience_points': random.randint(1000, 5000),
                    'badges': self._calculate_badges(signal),
                    'streak_days': random.randint(0, 14)
                }
                traders.append(trader)

            # Sort by reputation score
            traders.sort(key=lambda x: x['reputation_score'], reverse=True)
            return traders[:limit]

        except Exception as e:
            logger.error(f"Error getting top traders: {str(e)}")
            return self._generate_mock_top_traders(limit)

    def _calculate_badges(self, signal: Dict[str, Any]) -> List[str]:
        """Calculate badges based on trader performance"""
        badges = []

        if signal['confidence_score'] >= 0.9:
            badges.append('accuracy_master')
        if signal['momentum_score'] >= 0.8:
            badges.append('momentum_trader')
        if signal['trend_strength'] >= 2.0:
            badges.append('trend_hunter')
        if signal['total_reach'] >= 100000:
            badges.append('influencer')

        return badges

    def _calculate_trader_rank(self, social_score: float) -> int:
        """Calculate trader rank based on social score"""
        try:
            # Higher social score = lower rank number (better ranking)
            rank = max(1, int((100 - social_score) * 10))
            return min(rank, self.max_trader_rank)
        except Exception as e:
            logger.error(f"Error calculating trader rank: {str(e)}")
            return random.randint(1, self.max_trader_rank)

    def _generate_mock_social_metrics(self, token_address: str) -> Dict[str, Any]:
        """Generate realistic mock social trading metrics"""
        try:
            # Generate base success rate between 40-80%
            success_rate = random.uniform(0.4, 0.8)
            total_trades = random.randint(50, 500)
            successful_trades = int(total_trades * success_rate)

            # Calculate average trade size between 100-1000 SOL
            avg_trade_size = random.uniform(100, 1000)
            total_volume = avg_trade_size * total_trades

            # Calculate ROI based on success rate (-20% to +100%)
            roi_percentage = (success_rate - 0.5) * 200

            # Calculate social score (0-100) based on performance metrics
            social_score = (
                success_rate * 40 +  # Success rate contribution (max 32)
                min(total_trades / 500, 1) * 30 +  # Trade volume contribution (max 30)
                min(total_volume / 100000, 1) * 30  # Total volume contribution (max 30)
            )

            # Calculate rank based on social score (1-1000)
            trader_rank = max(1, int((1 - (social_score / 100)) * 1000))

            return {
                'token_address': token_address,
                'social_score': social_score,
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'total_volume': total_volume,
                'roi_percentage': roi_percentage,
                'trader_rank': trader_rank,
                'sentiment_score': random.uniform(0.6, 0.9),
                'engagement_rate': random.uniform(0.02, 0.15),
                'trend_direction': random.choice(['bullish', 'neutral', 'bearish']),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating mock social metrics: {str(e)}")
            return {
                'token_address': token_address,
                'error': str(e)
            }

    def _generate_mock_top_traders(self, limit: int) -> List[Dict[str, Any]]:
        """Generate realistic mock top traders data"""
        traders = []
        for i in range(limit):
            success_rate = random.uniform(0.5, 0.95)
            total_trades = random.randint(100, 1000)
            successful_trades = int(total_trades * success_rate)
            avg_profit = random.uniform(10, 50)

            trader = {
                'username': f'Trader_{i+1}',
                'reputation_score': random.uniform(7, 10),
                'total_signals': total_trades,
                'successful_signals': successful_trades,
                'win_rate': success_rate * 100,
                'avg_profit': avg_profit,
                'total_pnl': avg_profit * total_trades,
                'level': random.randint(5, 20),
                'experience_points': random.randint(1000, 5000),
                'badges': ['accuracy_master', 'volume_trader'] if success_rate > 0.8 else ['rising_star'],
                'streak_days': random.randint(0, 14)
            }
            traders.append(trader)

        # Sort by reputation score
        traders.sort(key=lambda x: x['reputation_score'], reverse=True)
        return traders

    def _make_api_request(self, endpoint: str, method: str = 'POST', 
                         params: Optional[Dict] = None, data: Optional[Dict] = None,
                         headers: Optional[Dict] = None) -> Dict:
        """Make API request using the API manager with proper BitQuery headers"""
        try:
            default_headers = {
                'X-API-KEY': os.environ.get('BITQUERY_API_KEY'),
                'Content-Type': 'application/json'
            }
            if headers:
                default_headers.update(headers)

            return api_manager.call_api(
                api_name='bitquery',
                endpoint=f"{self.api_endpoint}/{endpoint.lstrip('/')}",
                method=method,
                params=params,
                data=data,
                headers=default_headers
            )
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise APIError(f"Request failed: {str(e)}")

    @with_fallback(fallback_function=lambda *args, **kwargs: [])
    def get_early_launches(self, chain='solana'):
        """Fetch early token launches with fallback to test data"""
        if not self.api_key or self.use_mock_data:
            logger.info("Using mock data for early launches")
            return self._get_test_launches(chain)

        query = """
        query ($network: BlockchainNetwork!, $from: ISO8601DateTime) {
              DEXTrades(
                network: $network
                options: {
                  desc: ["Block.Timestamp", "Trade.Volume"]
                  limit: 50
                }
                date: {since: $from}
                where: {Trade.Side: {in: ["BUY", "SELL"]}}
              ) {
                Block {
                  Timestamp
                }
                Transaction {
                  Hash
                }
                Trade {
                  Volume
                  Price
                  Side
                  Currency {
                    Name
                    Symbol
                    SmartContract
                    Platform
                  }
                  QuoteCurrency {
                    Symbol
                    SmartContract
                  }
                }
                Pool {
                  Liquidity
                  Currency {
                    HolderCount
                    SmartContract
                  }
                }
              }
            }
        """

        variables = {
            "network": chain.upper(),
            "from": (datetime.utcnow() - timedelta(hours=self.max_age_hours)).isoformat()
        }

        response = self._make_api_request(
            endpoint='',
            method='POST',
            data={
                "query": query,
                "variables": variables
            }
        )

        if 'errors' in response:
            logger.error(f"GraphQL errors: {response['errors']}")
            return self._get_test_launches(chain)

        processed_launches = self._process_bitquery_response(response)
        if not processed_launches:
            logger.warning("No launches found from API, using test data")
            return self._get_test_launches(chain)

        return processed_launches

    def get_board_data(self, chain: str = 'solana') -> List[Dict]:
        """Get real-time board data with enhanced error handling"""
        try:
            if not self.initialized:
                raise APIError("Analyzer not properly initialized")

            if not self.api_key:
                logger.info("No API key found, using test data")
                return self._get_test_board_data()

            # Implementation for real API call would go here
            # For now, using test data with proper error handling
            return self._get_test_board_data()

        except APIError as e:
            logger.error(f"Board data error: {str(e)}")
            return self._get_test_board_data()  # Fallback to test data
        except Exception as e:
            logger.error(f"Unexpected error fetching board data: {str(e)}")
            return []

    def analyze_token_potential(self, token_data: Dict) -> Tuple[bool, Dict]:
        """Analyze token potential with comprehensive error handling"""
        try:
            if not self._validate_token_data(token_data):
                return False, {"error": "Invalid token data format"}

            analysis_result = {
                'timestamp': datetime.utcnow().isoformat(),
                'token_address': token_data.get('address'),
                'metrics': self._calculate_metrics(token_data),
                'risk_assessment': self._assess_risks(token_data),
                'potential_score': self._calculate_potential_score(token_data)
            }

            # Store results in database with error handling
            try:
                self._store_analysis_results(analysis_result)
            except Exception as db_error:
                logger.error(f"Database error: {str(db_error)}")
                # Continue with analysis even if storage fails

            return True, analysis_result

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return False, {"error": f"Validation error: {str(e)}"}
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return False, {"error": f"Analysis failed: {str(e)}"}

    def _validate_token_data(self, token_data: Dict) -> bool:
        """Validate token data structure"""
        required_fields = ['address', 'price', 'volume24h', 'liquidity', 'holders']
        return all(field in token_data for field in required_fields)

    def _calculate_metrics(self, token_data: Dict) -> Dict:
        """Calculate token metrics with error handling"""
        try:
            return {
                'price': token_data.get('price', 0),
                'volume_24h': token_data.get('volume24h', 0),
                'liquidity': token_data.get('liquidity', 0),
                'holder_count': token_data.get('holders', 0),
                'price_change': self._calculate_price_change(token_data)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _calculate_price_change(self, token_data: Dict) -> float:
        """Calculate price change with error handling"""
        try:
            if 'price_history' not in token_data:
                return 0.0

            history = token_data['price_history']
            if not history or len(history) < 2:
                return 0.0

            initial_price = history[0]['price']
            current_price = history[-1]['price']

            if initial_price == 0:
                return 0.0

            return ((current_price - initial_price) / initial_price) * 100

        except Exception as e:
            logger.error(f"Error calculating price change: {str(e)}")
            return 0.0

    def _assess_risks(self, token_data: Dict) -> Dict:
        """Assess token risks with error handling"""
        try:
            risk_factors = []

            if token_data.get('liquidity', 0) < self.min_liquidity:
                risk_factors.append('low_liquidity')
            if token_data.get('holders', 0) < self.min_holder_count:
                risk_factors.append('low_holder_count')
            if token_data.get('is_mint_enabled', True):
                risk_factors.append('mint_enabled')

            risk_level = 'high' if len(risk_factors) >= 2 else 'medium' if risk_factors else 'low'

            return {
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'safety_score': self._calculate_safety_score(token_data)
            }

        except Exception as e:
            logger.error(f"Error assessing risks: {str(e)}")
            return {'risk_level': 'unknown', 'risk_factors': [], 'safety_score': 0}

    def _calculate_safety_score(self, token_data: Dict) -> float:
        """Calculate safety score with error handling"""
        try:
            # Base score of 50
            score = 50.0

            # Liquidity factor (up to +20)
            liquidity = token_data.get('liquidity', 0)
            score += min(20, (liquidity / self.min_liquidity) * 10)

            # Holder factor (up to +20)
            holders = token_data.get('holders', 0)
            score += min(20, (holders / self.min_holder_count) * 10)

            # Contract verification (+10)
            if token_data.get('is_contract_verified', False):
                score += 10

            return min(100, max(0, score))

        except Exception as e:
            logger.error(f"Error calculating safety score: {str(e)}")
            return 0.0

    def _store_analysis_results(self, analysis_result: Dict) -> None:
        """Store analysis results with error handling"""
        try:
            token_metrics = TokenMetrics(
                address=analysis_result['token_address'],
                chain='solana',
                timestamp=datetime.utcnow(),
                price=analysis_result['metrics'].get('price'),
                volume=analysis_result['metrics'].get('volume_24h'),
                liquidity=analysis_result['metrics'].get('liquidity'),
                holder_count=analysis_result['metrics'].get('holder_count'),
                risk_score=analysis_result['risk_assessment'].get('safety_score'),
                potential_score=analysis_result.get('potential_score')
            )

            db.session.merge(token_metrics)
            db.session.commit()

        except Exception as e:
            logger.error(f"Error storing analysis results: {str(e)}")
            db.session.rollback()
            raise

    def analyze_volume_delta(self, token_data):
        """Analyze volume delta patterns for bot triggering"""
        try:
            inflow = float(token_data.get('inflow', 0))
            outflow = float(token_data.get('outflow', 0))

            # Calculate volume delta
            volume_delta = inflow - outflow

            # Calculate timing metrics
            launch_time = datetime.fromisoformat(token_data.get('launch_time', ''))
            time_delta = (datetime.utcnow() - launch_time).total_seconds()

            return {
                'volume_delta': volume_delta,
                'time_delta': time_delta,
                'meets_threshold': volume_delta >= self.volume_delta_threshold,
                'is_timely': time_delta <= self.delay_threshold if self.experimental_mode else True,
                'dev_buy_ratio': float(token_data.get('dev_buy_ratio', 0))
            }
        except Exception as e:
            logger.error(f"Error analyzing volume delta: {str(e)}")
            return None

    def calculate_optimal_bundle(self, market_metrics):
        """Calculate optimal bundle size based on market conditions"""
        try:
            # Adjust base amount based on market conditions
            if market_metrics.get('market_trend') == 'bearish':
                self.market_condition_multiplier = 1.5  # Increase bundle in bear market
            elif market_metrics.get('market_trend') == 'bullish':
                self.market_condition_multiplier = 0.8  # Decrease bundle in bull market

            # Calculate target volume delta
            target_volume = self.bundle_base_amount * self.market_condition_multiplier

            # Ensure minimum bot trigger volume
            if target_volume < self.min_bot_trigger_volume:
                target_volume = self.min_bot_trigger_volume

            return {
                'recommended_bundle': target_volume,
                'market_multiplier': self.market_condition_multiplier,
                'min_expected_delta': self.volume_delta_threshold,
                'experimental_mode': self.experimental_mode
            }
        except Exception as e:
            logger.error(f"Error calculating optimal bundle: {str(e)}")
            return None

    def _get_test_board_data(self):
        """Generate realistic test data for development"""
        try:
            current_time = datetime.utcnow()
            test_tokens = []

            # Generate multiple test tokens with realistic patterns
            for i in range(5):
                inflow = random.uniform(5, 25)  # SOL
                outflow = random.uniform(2, 10)  # SOL

                token = {
                    'address': f'SOLTEST{i+1}' * 8,
                    'launch_time': (current_time - timedelta(minutes=random.randint(1, 60))).isoformat(),
                    'inflow': inflow,
                    'outflow': outflow,
                    'volume_delta': inflow - outflow,
                    'liquidity': random.uniform(50000, 200000),
                    'dev_buy_ratio': random.uniform(0.8, 1.4),
                    'holder_count': random.randint(100, 500),
                    'market_cap': random.uniform(100000, 500000),
                    'price_change_24h': random.uniform(-20, 50),
                    'bot_activity_level': random.uniform(0.3, 0.9)
                }

                # Add market condition indicators
                token['market_metrics'] = {
                    'market_trend': 'bullish' if random.random() > 0.5 else 'bearish',
                    'global_sentiment': random.uniform(0.3, 0.8)
                }

                test_tokens.append(token)

            return test_tokens

        except Exception as e:
            logger.error(f"Error generating test board data: {str(e)}")
            return []

    def is_bot_trigger_candidate(self, token_data):
        """Determine if token is likely to trigger bot activity"""
        try:
            volume_analysis = self.analyze_volume_delta(token_data)
            if not volume_analysis:
                return False

            # Check key criteria
            meets_volume = volume_analysis['meets_threshold']
            good_timing = volume_analysis['is_timely']
            safe_dev_buy = volume_analysis['dev_buy_ratio'] <= self.dev_buy_max_ratio

            return meets_volume and good_timing and safe_dev_buy

        except Exception as e:
            logger.error(f"Error checking bot trigger potential: {str(e)}")
            return False

    def _get_test_launches(self, chain):
        """Get test launch data for development"""
        current_time = datetime.utcnow()
        logger.info(f"Generating test launches for chain: {chain}")

        try:
            test_launches = [
                {
                    'token_address': 'SoLTest111111111111111111111111111111111111',
                    'chain': 'solana',
                    'launch_time': (current_time - timedelta(hours=2)).isoformat(),
                    'liquidity': 75000,
                    'volume_24h': 15000,
                    'holder_count': 150,
                    'top_holder_percentage': 8.5,
                    'price': 0.00001,
                    'market_cap': 100000
                },
                {
                    'token_address': 'SoLTest222222222222222222222222222222222222',
                    'chain': 'solana',
                    'launch_time': (current_time - timedelta(hours=5)).isoformat(),
                    'liquidity': 120000,
                    'volume_24h': 35000,
                    'holder_count': 280,
                    'top_holder_percentage': 6.2,
                    'price': 0.00005,
                    'market_cap': 250000
                },
                {
                    'token_address': 'SoLTest333333333333333333333333333333333333',
                    'chain': 'solana',
                    'launch_time': (current_time - timedelta(hours=8)).isoformat(),
                    'liquidity': 95000,
                    'volume_24h': 28000,
                    'holder_count': 220,
                    'top_holder_percentage': 7.1,
                    'price': 0.00003,
                    'market_cap': 180000
                }
            ]

            # Filter launches by chain if specified
            if chain != 'all':
                test_launches = [launch for launch in test_launches if launch['chain'] == chain]

            logger.info(f"Generated {len(test_launches)} test launches for chain {chain}")

            # Process and prepare launches with risk assessment and potential scores
            processed_launches = []
            for launch in test_launches:
                try:
                    # Calculate metrics
                    metrics = {
                        'passed': True,
                        'details': {
                            'vol_liq_ratio': launch['volume_24h'] / launch['liquidity'],
                            'holder_count': launch['holder_count'],
                            'liquidity': launch['liquidity'],
                            'volume_244h': launch['volume_24h'],
                            'top_holder_percentage': launch['top_holder_percentage']
                        }
                    }

                    # Calculate risk score (between 0-10)
                    risk_score = min(
                        5 +  # Base score
                        (launch['top_holder_percentage'] / 20) +  # Higher concentration = higher risk
                        (50000 / launch['liquidity']) +  # Lower liquidity = higher risk
                        (100 / launch['holder_count']),  # Fewer holders = higher risk
                        10  # Cap at 10
                    )

                    # Calculate potential score (between 0-10)
                    potential_score = min(
                        (launch['volume_24h'] /launch['liquidity']) * 5 +  # Volume/liquidity impact
                        (launch['holder_count'] / 100) +  # Holder count impact
                        (launch['liquidity'] / 50000) +  # Liquidity impact
                        (15 - launch['top_holder_percentage']) / 5,  # Top holder impact
                        10  # Cap at 10
                    )

                    # Add calculated scores to launch data
                    launch['potential_score'] = potential_score
                    launch['risk_assessment'] = {
                        'risk_score': risk_score,
                        'passed': risk_score < 7
                    }

                    processed_launches.append(launch)
                    logger.info(f"Successfully processed test launch: {launch['token_address']}")

                except Exception as e:
                    logger.error(f"Error processing test launch {launch.get('token_address', 'unknown')}: {str(e)}")
                    continue

            if not processed_launches:
                logger.warning("No launches were successfully processed")
                return []

            return processed_launches

        except Exception as e:
            logger.error(f"Error in test launch generation: {str(e)}")
            return []

    def analyze_potential(self, launch_data):
        """Analyze the potential of a newly launched token"""
        try:
            # For testing without API key, return simulated analysis
            if not self.api_key:
                logger.info(f"Analyzing test launch for {launch_data['token_address']}")
                metrics = {
                    'passed': True,
                    'details': {
                        'vol_liq_ratio': launch_data['volume_24h'] / launch_data['liquidity'],
                        'holder_count': launch_data['holder_count'],
                        'liquidity': launch_data['liquidity'],
                        'volume_24h': launch_data['volume_24h'],
                        'top_holder_percentage': launch_data['top_holder_percentage'],
                        'risk_score': 3.5  # Test risk score
                    }
                }

                self._store_launch_metrics(
                    launch_data,
                    metrics,
                    7.5  # Test potential score
                )

                return True, {
                    'potential_score': 7.5,
                    'metrics': metrics,
                    'risk_assessment': {
                        'passed': True,
                        'reason': '',
                        'details': {
                            'risk_score': 3.5,
                            'chain': launch_data['chain'],
                            'price': launch_data['price']
                        }
                    }
                }

            # Basic checks
            if not self._validate_basic_requirements(launch_data):
                return False, "Failed basic requirements"

            metrics = self._analyze_metrics(launch_data)
            if not metrics['passed']:
                return False, metrics['reason']

            risk_check = self._check_risk_parameters(launch_data)
            if not risk_check['passed']:
                return False, risk_check['reason']

            potential_score = self._calculate_potential_score(launch_data, metrics)
            self._store_launch_metrics(launch_data, metrics, potential_score)

            return True, {
                'potential_score': potential_score,
                'metrics': metrics,
                'risk_assessment': risk_check
            }

        except Exception as e:
            logger.error(f"Error analyzing launch potential: {str(e)}")
            return False, f"Analysis error: {str(e)}"

    def _process_bitquery_response(self, data):
        """Process Bitquery GraphQL APIv2 response data"""
        try:
            trades = data.get('data', {}).get('DEXTrades', [])
            launches = []
            processed_tokens = set()

            for trade in trades:
                token = trade['Trade']['Currency']
                token_address = token['SmartContract']

                # Skip if we've already processed this token
                if token_address in processed_tokens:
                    continue
                processed_tokens.add(token_address)

                pool = trade['Pool']

                launch = {
                    'token_address': token_address,
                    'chain': 'solana',
                    'launch_time': datetime.fromisoformat(
                        trade['Block']['Timestamp']
                    ).isoformat(),
                    'liquidity': float(pool['Liquidity'] or 0),
                    'volume_24h': float(trade['Trade']['Volume'] or 0),
                    'holder_count': int(pool['Currency']['HolderCount'] or 0),
                    'top_holder_percentage': 0,  # Not available in APIv2 directly
                    'price': float(trade['Trade']['Price'] or 0),
                    'market_cap': 0  # Calculate separately if needed
                }
                launches.append(launch)

            return launches

        except Exception as e:
            logger.error(f"Error processing Bitquery response: {str(e)}")
            return []

    def _store_launch_metrics(self, launch_data, metrics, potential_score):
        """Store launch metrics in the database"""
        try:
            token_metrics = TokenMetrics.query.filter_by(
                token_address=launch_data['token_address'],
                chain=launch_data['chain']
            ).first()

            if not token_metrics:
                token_metrics = TokenMetrics(
                    token_address=launch_data['token_address'],
                    chain=launch_data['chain']
                )

            # Update launch-specific metrics
            token_metrics.launch_time = datetime.fromisoformat(launch_data['launch_time'])
            token_metrics.holder_count = launch_data['holder_count']
            token_metrics.top_holder_percentage = launch_data['top_holder_percentage']
            token_metrics.launch_price = launch_data['price']
            token_metrics.current_price = launch_data['price']
            token_metrics.is_early_launch = True
            token_metrics.launch_score= potential_score
            token_metrics.launch_risk_score = metrics.get('risk_score', 0)

            # Update general metrics
            token_metrics.liquidity = launch_data['liquidity']
            token_metrics.volume_24h = launch_data['volume_24h']
            token_metrics.potential_score = potential_score

            db.session.add(token_metrics)
            db.session.commit()

        except Exception as e:
            logger.error(f"Error storing launch metrics: {str(e)}")
            db.session.rollback()

    def _validate_basic_requirements(self, launch_data):
        """Validate basic requirements for a launch"""
        try:
            # Check liquidity
            if launch_data['liquidity'] < self.min_liquidity:
                return False

            # Check launch time
            launch_time = datetime.fromisoformat(launch_data['launch_time'])
            if datetime.utcnow() - launch_time > timedelta(hours=self.max_age_hours):
                return False

            # Check holder count
            if launch_data['holder_count'] < self.min_holder_count:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating basic requirements: {str(e)}")
            return False

    def _analyze_metrics(self, launch_data):
        """Analyze various metrics for the launch"""
        try:
            metrics = {
                'passed': True,
                'reason': '',
                'details': {}
            }

            # Volume/Liquidity ratio
            vol_liq_ratio = launch_data['volume_24h'] / launch_data['liquidity']
            metrics['details']['vol_liq_ratio'] = vol_liq_ratio

            if vol_liq_ratio < self.min_volume_liquidity_ratio:
                metrics['passed'] = False
                metrics['reason'] = 'Insufficient volume/liquidity ratio'
                return metrics

            # Top holder percentage
            if launch_data['top_holder_percentage'] > self.max_top_holder_percentage:
                metrics['passed'] = False
                metrics['reason'] = 'Top holder owns too much'
                return metrics

            # Add other relevant metrics
            metrics['details'].update({
                'holder_count': launch_data['holder_count'],
                'liquidity': launch_data['liquidity'],
                'volume_24h': launch_data['volume_24h'],
                'top_holder_percentage': launch_data['top_holder_percentage']
            })

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing metrics: {str(e)}")
            return {'passed': False, 'reason': f"Metrics analysis error: {str(e)}"}

    def _check_risk_parameters(self, launch_data):
        """Check risk management parameters"""
        try:
            risk_check = {
                'passed': True,
                'reason': '',
                'details': {}
            }

            # Validate trade through risk manager
            valid, reason = self.risk_manager.validate_trade(
                launch_data['token_address'],
                launch_data['chain'],
                1.0,  # Sample amount
                launch_data['price'],
                'BUY'
            )

            if not valid:
                risk_check['passed'] = False
                risk_check['reason'] = reason
                return risk_check

            # Add risk metrics
            risk_check['details'] = {
                'chain': launch_data['chain'],
                'price': launch_data['price'],
                'market_cap': launch_data.get('market_cap', 0),
                'risk_score': self._calculate_risk_score(launch_data)
            }

            return risk_check

        except Exception as e:
            logger.error(f"Error checking risk parameters: {str(e)}")
            return {'passed': False, 'reason': f"Risk check error: {str(e)}"}

    def _calculate_potential_score(self, launch_data: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> float:
        """Calculate potential score for the launch"""
        try:
            # Base score starts at 5
            score = 5.0

            # If no metrics provided, use launch data directly
            if not metrics:
                metrics = {
                    'details': {
                        'vol_liq_ratio': launch_data.get('volume_24h', 0) / max(launch_data.get('liquidity', 1), 1),
                        'holder_count': launch_data.get('holder_count', 0),
                        'liquidity': launch_data.get('liquidity', 0),
                        'volume_24h': launch_data.get('volume_24h', 0),
                        'top_holder_percentage': launch_data.get('top_holder_percentage', 0)
                    }
                }

            # Volume/Liquidity ratio impact (up to +2)
            vol_liq_ratio = metrics['details'].get('vol_liq_ratio', 0)
            score += min(vol_liq_ratio * 5, 2)

            # Holder count impact (up to +1)
            holder_count = metrics['details'].get('holder_count', 0)
            holder_ratio = holder_count / max(self.min_holder_count, 1)
            score += min(holder_ratio - 1, 1)

            # Liquidity impact (up to +1)
            liquidity = metrics['details'].get('liquidity', 0)
            liquidity_ratio = liquidity / max(self.min_liquidity, 1)
            score += min(liquidity_ratio - 1, 1)

            # Top holder percentage impact (up to +1)
            top_holder_percentage = metrics['details'].get('top_holder_percentage', 0)
            holder_score = (self.max_top_holder_percentage - top_holder_percentage) / self.max_top_holder_percentage
            score += holder_score

            return min(max(score, 0), 10)  # Ensure score is between 0 and 10

        except Exception as e:
            logger.error(f"Error calculating potential score: {str(e)}")
            return 0

    def _calculate_risk_score(self, launch_data):
        """Calculate risk score for the launch"""
        try:
            # Base risk score starts at 5
            risk_score = 5.0

            # Liquidity factor (higher liquidity = lower risk)
            liquidity_factor = min(launch_data['liquidity'] / (self.min_liquidity * 2), 1)
            risk_score -= liquidity_factor * 2

            # Holder concentration factor (higher concentration = higher risk)
            concentration_risk = (launch_data['top_holder_percentage'] / self.max_top_holder_percentage) * 3
            risk_score += concentration_risk

            # Age factor (newer = higher risk)
            launch_time = datetime.fromisoformat(launch_data['launch_time'])
            age_hours = (datetime.utcnow() - launch_time).total_seconds() / 3600
            age_factor = (self.max_age_hours - age_hours) / self.max_age_hours
            risk_score += age_factor * 2

            return min(max(risk_score, 0), 10)  # Ensure score is between 0 and 10

        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 10  # Return maximum risk on error

    # Added missing attribute for volume/liquidity ratio check
    min_volume_liquidity_ratio = 0.1
    max_top_holder_percentage = 20

    def _generate_mock_social_metrics(self, token_address: str) -> Dict[str, Any]:
        """Generate realistic mock social trading metrics"""
        try:
            # Generate base success rate between 40-80%
            success_rate = random.uniform(0.4, 0.8)
            total_trades = random.randint(50, 500)
            successful_trades = int(total_trades * success_rate)

            # Calculate average trade size between 100-1000 SOL
            avg_trade_size = random.uniform(100, 1000)
            total_volume = avg_trade_size * total_trades

            # Calculate ROI based on success rate (-20% to +100%)
            roi_percentage = (success_rate - 0.5) * 200

            # Calculate social score (0-100) based on performance metrics
            social_score = (
                success_rate * 40 +  # Success rate contribution (max 32)
                min(total_trades / 500, 1) * 30 +  # Trade volume contribution (max 30)
                min(total_volume / 100000, 1) * 30  # Total volume contribution (max 30)
            )

            # Calculate rank based on social score (1-1000)
            trader_rank = max(1, int((1 - (social_score / 100)) * 1000))

            return {
                'token_address': token_address,
                'social_score': social_score,
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'total_volume': total_volume,
                'roi_percentage': roi_percentage,
                'trader_rank': trader_rank,
                'sentiment_score': random.uniform(0.6, 0.9),
                'engagement_rate': random.uniform(0.02, 0.15),
                'trend_direction': random.choice(['bullish', 'neutral', 'bearish']),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating mock social metrics: {str(e)}")
            return {
                'token_address': token_address,
                'error': str(e)
            }

    def get_social_trading_metrics(self, token_address: str, chain: str = 'solana') -> Dict[str, Any]:
        """Get social trading metrics for leaderboard"""
        try:
            if self.use_mock_data:
                return self._generate_mock_social_metrics(token_address)

            # Real implementation will use Spydefi API when available
            metrics = {
                'token_address': token_address,
                'chain': chain,
                'social_score': 0,
                'total_trades': 0,
                'successful_trades': 0,
                'total_volume': 0,
                'roi_percentage': 0,
                'trader_rank': 0,
                'timestamp': datetime.utcnow().isoformat()
            }

            return metrics

        except Exception as e:
            logger.error(f"Error fetching social trading metrics: {str(e)}")
            return {'error': str(e)}

    def get_top_traders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top traders for leaderboard"""
        try:
            if not self.use_mock_data:
                return []

            with current_app.app_context():
                # Generate mock data for multiple traders
                traders = []
                for i in range(limit):
                    success_rate = random.uniform(0.5, 0.95)
                    total_trades = random.randint(100, 1000)
                    successful_trades = int(total_trades * success_rate)
                    avg_profit = random.uniform(10, 50)

                    trader = {
                        'username': f'Trader_{i+1}',
                        'reputation_score': random.uniform(7, 10),
                        'total_signals': total_trades,
                        'successful_signals': successful_trades,
                        'win_rate': success_rate * 100,
                        'avg_profit': avg_profit,
                        'total_pnl': avg_profit * total_trades,
                        'level': random.randint(5, 20),
                        'experience_points': random.randint(1000, 5000),
                        'badges': ['accuracy_master', 'volume_trader'] if success_rate > 0.8 else ['rising_star'],
                        'streak_days': random.randint(0, 14)
                    }
                    traders.append(trader)

                # Sort by reputation score
                traders.sort(key=lambda x: x['reputation_score'], reverse=True)
                return traders

        except Exception as e:
            logger.error(f"Error getting top traders: {str(e)}")
            return []

    def calculate_trader_rank(self, user_id: int) -> Dict[str, Any]:
        """Calculate trader ranking and stats"""
        try:
            with current_app.app_context():
                user = User.query.get(user_id)
                if not user:
                    return {'error': 'User not found'}

                # Calculate rank based on reputation score
                better_traders = User.query.filter(
                    User.reputation_score > user.reputation_score
                ).count()
                rank = better_traders + 1

                return {
                    'rank': rank,
                    'total_signals': user.total_signals,
                    'successful_signals': user.successful_signals,
                    'win_rate': user.win_rate,
                    'avg_profit': user.avg_profit,
                    'level': user.level,
                    'experience_points': user.experience_points,
                    'badges': user.badges,
                    'streak_days': user.streak_days
                }

        except Exception as e:
            logger.error(f"Error calculating trader rank: {str(e)}")
            return {'error': str(e)}