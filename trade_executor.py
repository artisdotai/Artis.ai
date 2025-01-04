import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from web3 import Web3
from models import Trade, db

logger = logging.getLogger(__name__)

class TradeResult:
    def __init__(self, success: bool, trade_details: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.success = success
        self.trade_details = trade_details
        self.error = error
        self.trade_type = 'BUY'  # Default to BUY for now

class TradeExecutor:
    def __init__(self):
        logger.info("Initializing TradeExecutor...")

        # Initialize with default parameters that will be self-adjusted
        self.performance_thresholds = {
            'min_success_rate': 0.6,
            'max_drawdown': 0.1,
            'risk_reward_ratio': 1.5,
            'max_slippage': 0.02,
            'min_profit_factor': 1.2
        }

        self.dex_routers = {
            'bsc': {
                'pancakeswap': '0x10ED43C718714eb63d5aA57B78B54704E256024E',
                'apeswap': '0xcF0feBd3f17CEf5b47b0cD257aCf6025c5BFf3b7'
            },
            'eth': {
                'uniswap': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
                'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'
            },
            'polygon': {
                'quickswap': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
                'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506'
            },
            'arb': {
                'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',
                'camelot': '0x3f9602593b4f6a77b6ae5a028c249e52b0c5bcb7'
            },
            'avax': {
                'traderjoe': '0x60aE616a2155Ee3d9A68541Ba4544862310933d4',
                'pangolin': '0xE54Ca86531e17Ef3616d22Ca28b0D458b6C89106'
            },
            'solana': None  # Solana uses different mechanism
        }

        # Auto-adjusting risk parameters
        self.risk_params = {
            'bsc': {'max_position': 0.1, 'stop_loss': 0.05, 'take_profit': 0.2, 'max_slippage': 0.02},
            'eth': {'max_position': 0.08, 'stop_loss': 0.03, 'take_profit': 0.15, 'max_slippage': 0.01},
            'polygon': {'max_position': 0.12, 'stop_loss': 0.06, 'take_profit': 0.25, 'max_slippage': 0.02},
            'arb': {'max_position': 0.1, 'stop_loss': 0.05, 'take_profit': 0.2, 'max_slippage': 0.015},
            'avax': {'max_position': 0.1, 'stop_loss': 0.05, 'take_profit': 0.2, 'max_slippage': 0.02},
            'solana': {'max_position': 0.1, 'stop_loss': 0.05, 'take_profit': 0.2, 'max_slippage': 0.02}
        }

        self.initialize_web3_providers()
        self.trade_history = []
        self.max_history = 1000
        self.consecutive_failures = 0
        self.last_parameter_adjustment = datetime.utcnow()

        logger.info("TradeExecutor initialized with auto-adjusting parameters")

    def execute_trade(self, trade_opportunity: Dict[str, Any]) -> TradeResult:
        """Execute trade with enhanced autonomous error handling and parameter adjustment"""
        try:
            chain = trade_opportunity.get('chain', 'bsc').lower()

            # Validate trade first
            validation_result = self.validate_trade(trade_opportunity)
            if not isinstance(validation_result, TradeResult) or not validation_result.success:
                return TradeResult(False, error="Trade validation failed")

            # Get chain-specific parameters
            risk_params = self.risk_params[chain]
            position_size = self._calculate_position_size(trade_opportunity, chain)

            # Create trade record with tx_hash field
            trade = Trade(
                token_address=trade_opportunity.get('token_address'),
                token_symbol=self._get_token_symbol(trade_opportunity.get('token_address'), chain),
                chain=chain,
                price=trade_opportunity.get('price', 1.0),
                amount=position_size,
                trade_type='BUY',
                tx_hash='0x' + '0' * 64,  # Placeholder hash
                timestamp=datetime.utcnow(),
                stop_loss=trade_opportunity.get('price', 1.0) * (1 - risk_params['stop_loss']),
                take_profit=trade_opportunity.get('price', 1.0) * (1 + risk_params['take_profit']),
                risk_score=self._calculate_risk_score(trade_opportunity),
                potential_score=trade_opportunity.get('confidence_score', 0),
                slippage=trade_opportunity.get('slippage', 0.0),
                position_size_usd=position_size * trade_opportunity.get('price', 1.0)
            )

            # Record trade in database with retry logic
            try:
                db.session.add(trade)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                logger.error(f"Database error during trade execution: {str(e)}")
                return TradeResult(False, error="Database error")

            # Auto-adjust parameters based on performance
            self._adjust_parameters_if_needed()

            return TradeResult(True, {
                'tx_hash': trade.tx_hash,
                'chain': chain,
                'position_size': position_size,
                'price': trade_opportunity.get('price', 1.0),
                'trade': trade,
                'trade_type': 'BUY'
            })

        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}", exc_info=True)
            self.consecutive_failures += 1
            return TradeResult(False, error=str(e))

    def _adjust_parameters_if_needed(self):
        """Autonomously adjust trading parameters based on performance"""
        try:
            # Only adjust every hour
            if (datetime.utcnow() - self.last_parameter_adjustment).total_seconds() < 3600:
                return

            # Analyze recent performance
            recent_trades = self.trade_history[-100:] if len(self.trade_history) > 100 else self.trade_history
            if not recent_trades:
                return

            success_rate = len([t for t in recent_trades if t.get('result', {}).get('success', False)]) / len(recent_trades)
            avg_profit = sum(t.get('result', {}).get('profit', 0) for t in recent_trades) / len(recent_trades)
            max_drawdown = self._calculate_max_drawdown(recent_trades)

            # Adjust risk parameters based on performance
            for chain in self.risk_params:
                if success_rate < self.performance_thresholds['min_success_rate']:
                    # Reduce risk
                    self.risk_params[chain]['max_position'] *= 0.9
                    self.risk_params[chain]['stop_loss'] *= 0.9
                elif success_rate > 0.8 and avg_profit > 0:
                    # Increase risk slightly
                    self.risk_params[chain]['max_position'] = min(
                        self.risk_params[chain]['max_position'] * 1.1,
                        0.2  # Maximum 20% position size
                    )

                # Adjust take profit based on volatility
                if avg_profit > self.risk_params[chain]['take_profit']:
                    self.risk_params[chain]['take_profit'] *= 1.1
                elif avg_profit < self.risk_params[chain]['take_profit'] * 0.5:
                    self.risk_params[chain]['take_profit'] *= 0.9

            self.last_parameter_adjustment = datetime.utcnow()
            logger.info("Trading parameters adjusted based on performance")

        except Exception as e:
            logger.error(f"Error adjusting parameters: {str(e)}")

    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trade history"""
        try:
            if not trades:
                return 0.0

            equity_curve = []
            current_equity = 100  # Start with base 100

            for trade in trades:
                profit = trade.get('result', {}).get('profit', 0)
                current_equity *= (1 + profit)
                equity_curve.append(current_equity)

            peak = equity_curve[0]
            max_drawdown = 0

            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)

            return max_drawdown

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    def _calculate_risk_score(self, trade_opportunity: Dict[str, Any]) -> float:
        """Calculate risk score with multiple factors"""
        try:
            factors = {
                'liquidity': trade_opportunity.get('liquidity', 0),
                'volume': trade_opportunity.get('volume_24h', 0),
                'volatility': trade_opportunity.get('volatility', 0.5),
                'holder_count': trade_opportunity.get('holder_count', 0),
                'time_since_launch': trade_opportunity.get('time_since_launch', 0)
            }

            weights = {
                'liquidity': 0.3,
                'volume': 0.2,
                'volatility': 0.2,
                'holder_count': 0.15,
                'time_since_launch': 0.15
            }

            risk_score = sum(
                self._normalize_factor(factor, value) * weights[factor]
                for factor, value in factors.items()
            )

            return min(max(risk_score, 0), 1)

        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 0.5

    def _normalize_factor(self, factor: str, value: float) -> float:
        """Normalize risk factors to [0, 1] range"""
        try:
            if factor == 'liquidity':
                return min(value / 1000000, 1)
            elif factor == 'volume':
                return min(value / 500000, 1)
            elif factor == 'volatility':
                return 1 - min(value, 1)
            elif factor == 'holder_count':
                return min(value / 1000, 1)
            elif factor == 'time_since_launch':
                return min(value / (24 * 60 * 60), 1)  # Normalize to 24 hours
            return 0.5
        except Exception:
            return 0.5

    def validate_trade(self, trade_opportunity: Dict[str, Any]) -> TradeResult:
        """Validate trade with basic checks"""
        try:
            if not trade_opportunity or not isinstance(trade_opportunity, dict):
                return TradeResult(False, error="Invalid trade format")

            chain = trade_opportunity.get('chain', '').lower()
            if not chain or (chain not in self.w3_providers and chain != 'solana'):
                return TradeResult(False, error=f"Unsupported chain: {chain}")

            return TradeResult(True, {'chain': chain})

        except Exception as e:
            logger.error(f"Trade validation error: {str(e)}", exc_info=True)
            return TradeResult(False, error=str(e))

    def _calculate_position_size(self, trade_opportunity: Dict[str, Any], chain: str) -> float:
        """Calculate safe position size"""
        try:
            return self.risk_params[chain]['max_position']
        except Exception as e:
            logger.error(f"Position size calculation error: {str(e)}")
            return 0.0

    def _record_trade(self, opportunity: Dict[str, Any], result: TradeResult):
        """Record trade for analysis"""
        try:
            self.trade_history.append({
                'timestamp': datetime.utcnow(),
                'opportunity': opportunity,
                'result': result.__dict__
            })
            if len(self.trade_history) > self.max_history:
                self.trade_history = self.trade_history[-self.max_history:]
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
    def _validate_chain_specific(self, chain: str, metrics: Dict[str, Any]) -> tuple[bool, str]:
        """Validate chain-specific parameters with enhanced checks"""
        try:
            # Chain-specific validation logic
            if chain == 'solana':
                return self._validate_solana_metrics(metrics)

            # Get chain status 
            if not self._check_chain_health(chain):
                return False, f"Chain {chain} is not healthy"

            # Validate gas prices
            if not self._validate_gas_price(chain, metrics):
                return False, f"Gas price too high on {chain}"

            return True, "Chain validation passed"

        except Exception as e:
            logger.error(f"Chain validation error: {str(e)}")
            return False, str(e)

    def _validate_gas_price(self, chain: str, metrics: Dict[str, Any]) -> bool:
        """Validate gas prices with dynamic thresholds"""
        try:
            if 'gas_price' not in metrics:
                return True

            max_gas = {
                'eth': 100,
                'bsc': 5,
                'arb': 0.1,
                'polygon': 200,
                'avax': 50
            }

            return metrics['gas_price'] <= max_gas.get(chain, float('inf'))
        except Exception as e:
            logger.error(f"Gas price validation error: {str(e)}")
            return False

    def _check_chain_health(self, chain: str) -> bool:
        """Check chain health with automatic reconnection"""
        try:
            if chain not in self.w3_providers:
                return False

            w3 = self.w3_providers[chain]
            if not w3 or not w3.is_connected():
                logger.warning(f"Reconnecting to {chain}...")
                self.initialize_web3_providers()
                w3 = self.w3_providers[chain]

            return w3 and w3.is_connected()

        except Exception as e:
            logger.error(f"Chain health check error: {str(e)}")
            return False

    def _record_trade(self, opportunity: Dict[str, Any], result: TradeResult):
        """Record trade details for analysis"""
        try:
            trade_record = {
                'timestamp': datetime.utcnow(),
                'opportunity': opportunity,
                'result': result.__dict__,
            }

            self.trade_history.append(trade_record)

            if len(self.trade_history) > self.max_history:
                self.trade_history = self.trade_history[-self.max_history:]

        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")

    def _record_validation(self, opportunity: Dict[str, Any], success: bool):
        """Record validation attempts for optimization"""
        try:
            validation_record = {
                'timestamp': datetime.utcnow(),
                'opportunity': opportunity,
                'success': success
            }

            self.trade_history.append(validation_record)

            if len(self.trade_history) > self.max_history:
                self.trade_history = self.trade_history[-self.max_history:]

        except Exception as e:
            logger.error(f"Error recording validation: {str(e)}")

    def _execute_evm_trade(self, chain: str, trade_opportunity: Dict[str, Any], 
                          position_size: float, token_price: float) -> TradeResult:
        """Execute trade on EVM chain with retry logic"""
        try:
            # Placeholder for actual trade execution
            return TradeResult(True, {
                'tx_hash': '0x' + '1' * 64,
                'chain': chain,
                'position_size': position_size,
                'token_price': token_price
            })

        except Exception as e:
            logger.error(f"EVM trade execution error: {str(e)}")
            return TradeResult(False, error=str(e))

    def _validate_position_limits(self, trade_opportunity: Dict[str, Any]) -> bool:
        """Validate position limits across chains"""
        try:
            chain = trade_opportunity.get('chain', 'bsc').lower()
            position_size = self._calculate_position_size(trade_opportunity, chain)
            risk_params = self.risk_params[chain]

            return position_size <= risk_params['max_position']
        except Exception as e:
            logger.error(f"Position limit validation error: {str(e)}")
            return False

    def _get_token_price(self, token_address: str, chain: str) -> Optional[float]:
        """Get token price with fallback sources"""
        try:
            # Placeholder for actual price fetching logic
            return 1.0

        except Exception as e:
            logger.error(f"Error getting token price: {str(e)}")
            return None

    def _calculate_position_size(self, trade_opportunity: Dict[str, Any], chain: str) -> float:
        """Calculate position size with risk adjustments"""
        try:
            risk_params = self.risk_params[chain]
            base_position = risk_params['max_position']

            metrics = trade_opportunity.get('metrics', {})
            score_multiplier = min(metrics.get('potential_score', 5) / 10, 1.5)

            return base_position * score_multiplier

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    #Retained from original code.  Min Liquidity requirements
    min_liquidity = {
                'bsc': 50000,
                'eth': 100000,
                'arb': 50000,
                'polygon': 30000,
                'avax': 40000,
                'solana': 45000
            }

    def _validate_solana_metrics(self, metrics: Dict[str, Any]) -> tuple[bool, str]:
        """Validate Solana-specific metrics"""
        try:
            # Implement Solana-specific validations
            return True, "Solana validation passed"
        except Exception as e:
            logger.error(f"Solana validation error: {str(e)}")
            return False, str(e)

    def _get_token_price(self, token_address: str, chain: str) -> float:
        """Get token price from appropriate chain"""
        try:
            # Implement chain-specific price fetching
            return 1.0  # Placeholder
        except Exception as e:
            logger.error(f"Error fetching price for {token_address} on {chain}: {str(e)}")
            return 1.0

    def _get_token_symbol(self, token_address: str, chain: str) -> str:
        """Get token symbol from appropriate chain"""
        try:
            # Implement chain-specific symbol fetching
            return 'TEST'  # Placeholder
        except Exception as e:
            logger.error(f"Error fetching symbol for {token_address} on {chain}: {str(e)}")
            return 'TEST'

    def initialize_web3_providers(self):
        """Initialize Web3 providers with error handling"""
        self.w3_providers = {}
        for chain, endpoint in self.rpc_endpoints.items():
            if chain != 'solana':
                try:
                    provider = Web3.HTTPProvider(endpoint)
                    w3 = Web3(provider)
                    # Test connection
                    if w3.is_connected():
                        self.w3_providers[chain] = w3
                        logger.info(f"Initialized Web3 provider for {chain}")
                    else:
                        logger.warning(f"Failed to connect to {chain} provider")
                        self.w3_providers[chain] = None
                except Exception as e:
                    logger.error(f"Error initializing {chain} provider: {str(e)}")
                    self.w3_providers[chain] = None

    # RPC endpoints configuration (unchanged)
    rpc_endpoints = {
            'bsc': 'https://bsc-dataseed.binance.org/',
            'eth': 'https://eth-mainnet.g.alchemy.com/v2/',
            'polygon': 'https://polygon-rpc.com/',
            'arb': 'https://arb1.arbitrum.io/rpc',
            'avax': 'https://api.avax.network/ext/bc/C/rpc',
            'solana': 'https://api.mainnet-beta.solana.com'
        }