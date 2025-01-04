import logging
from datetime import datetime
from models import db, Trade, TokenMetrics, CustomStrategy
from decimal import Decimal
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self):
        # Default risk parameters
        self.default_params = {
            'max_position_size': 0.1,  # Max 10% of portfolio per position
            'stop_loss_percent': 0.05,  # 5% stop loss
            'take_profit_percent': 0.15,  # 15% take profit
            'max_open_positions': 5,
            'max_chain_allocation': 0.3,  # Max 30% per chain
            'min_liquidity_required': 50000,  # Minimum liquidity in USD
            'max_slippage': 0.02,  # Maximum allowed slippage 2%
        }

        # Chain-specific risk parameters
        self.chain_params = {
            'bsc': {
                'max_position_size': 0.15,
                'min_liquidity_required': 75000,
                'max_slippage': 0.025,
                'stop_loss_percent': 0.05,
                'take_profit_percent': 0.15
            },
            'eth': {
                'max_position_size': 0.2,
                'min_liquidity_required': 100000,
                'max_slippage': 0.015,
                'stop_loss_percent': 0.05,
                'take_profit_percent': 0.15
            },
            'polygon': {
                'max_position_size': 0.12,
                'min_liquidity_required': 50000,
                'max_slippage': 0.02,
                'stop_loss_percent': 0.05,
                'take_profit_percent': 0.15
            },
            'arb': {
                'max_position_size': 0.12,
                'min_liquidity_required': 60000,
                'max_slippage': 0.02,
                'stop_loss_percent': 0.05,
                'take_profit_percent': 0.15
            },
            'avax': {
                'max_position_size': 0.1,
                'min_liquidity_required': 40000,
                'max_slippage': 0.02,
                'stop_loss_percent': 0.05,
                'take_profit_percent': 0.15
            },
            'solana': {
                'max_position_size': 0.1,
                'min_liquidity_required': 45000,
                'max_slippage': 0.02,
                'stop_loss_percent': 0.05,
                'take_profit_percent': 0.15
            }
        }

        # Dynamic risk adjustment parameters
        self.risk_multipliers = {
            'high_volatility': 0.7,  # Reduce position size in high volatility
            'strong_trend': 1.2,  # Increase position size in strong trends
            'low_liquidity': 0.6,  # Reduce position size in low liquidity
            'high_sentiment': 1.1,  # Increase position size with positive sentiment
        }

        # Add strategy-specific parameters
        self.strategy_params = {
            'min_win_rate': 0.5,  # Minimum required win rate for strategy
            'max_daily_risk': 0.05,  # Maximum 5% portfolio risk per day
            'max_strategy_allocation': 0.2,  # Maximum 20% allocation per strategy
        }

    def validate_trade(self, token_address, chain, amount, price, trade_type, strategy_id=None):
        """Validate if a trade meets all risk management criteria"""
        try:
            # Get token metrics
            metrics = TokenMetrics.query.filter_by(
                token_address=token_address,
                chain=chain
            ).first()

            if not metrics:
                logger.warning(f"No metrics found for token {token_address} on {chain}")
                return False, "No metrics available"

            # Get chain-specific parameters
            chain_params = self.chain_params.get(chain, self.default_params)

            # If strategy is specified, validate against strategy rules
            if strategy_id:
                strategy_valid, strategy_msg = self._validate_strategy_rules(
                    strategy_id, metrics, amount, price
                )
                if not strategy_valid:
                    return False, strategy_msg

            # 1. Check liquidity
            if metrics.liquidity < chain_params['min_liquidity_required']:
                return False, f"Insufficient liquidity: {metrics.liquidity} < {chain_params['min_liquidity_required']}"

            # 2. Check position size limits
            position_value = float(amount) * float(price)
            portfolio_value = self._get_portfolio_value()
            position_size_ratio = position_value / portfolio_value if portfolio_value > 0 else float('inf')

            adjusted_max_position_size = self._adjust_position_size(
                chain_params['max_position_size'],
                metrics
            )

            if position_size_ratio > adjusted_max_position_size:
                return False, f"Position size too large: {position_size_ratio:.2%} > {adjusted_max_position_size:.2%}"

            # 3. Check chain allocation
            chain_allocation = self._get_chain_allocation(chain)
            if chain_allocation + position_size_ratio > self.default_params['max_chain_allocation']:
                return False, f"Chain allocation exceeded: {chain_allocation + position_size_ratio:.2%} > {self.default_params['max_chain_allocation']:.2%}"

            # 4. Check open positions limit
            open_positions = self._get_open_positions_count()
            if open_positions >= self.default_params['max_open_positions']:
                return False, f"Max open positions reached: {open_positions}"

            # 5. Validate technical indicators
            if not self._validate_technical_indicators(metrics):
                return False, "Technical indicators not favorable"

            # 6. Check sentiment and volatility
            if not self._validate_market_conditions(metrics):
                return False, "Market conditions not favorable"

            # Additional strategy-based validation
            if strategy_id:
                position_value = float(amount) * float(price)
                if not self._validate_strategy_allocation(strategy_id, position_value):
                    return False, "Strategy allocation limit exceeded"

                if not self._validate_strategy_performance(strategy_id):
                    return False, "Strategy performance below threshold"

            return True, "Trade validated successfully"

        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def calculate_position_size(self, token_address, chain, available_capital, strategy_id=None):
        """Calculate optimal position size based on risk parameters and strategy"""
        try:
            metrics = TokenMetrics.query.filter_by(
                token_address=token_address,
                chain=chain
            ).first()

            if not metrics:
                return 0

            chain_params = self.chain_params.get(chain, self.default_params)

            # If strategy is specified, use strategy-specific position sizing
            if strategy_id:
                strategy = CustomStrategy.query.get(strategy_id)
                if strategy:
                    market_conditions = self._get_market_conditions(metrics)
                    return strategy.get_position_size(available_capital, market_conditions)

            # Base position size
            base_size = available_capital * chain_params['max_position_size']

            # Adjust based on risk factors
            risk_multiplier = self._calculate_risk_multiplier(metrics)

            # Calculate final position size
            position_size = base_size * risk_multiplier

            # Apply maximum limits
            max_allowed = available_capital * chain_params['max_position_size']
            return min(position_size, max_allowed)

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def update_stop_loss(self, trade_id, current_price):
        """Update stop loss based on price movement"""
        try:
            trade = Trade.query.get(trade_id)
            if not trade:
                return False

            chain_params = self.chain_params.get(trade.chain, self.default_params)

            # Calculate trailing stop loss
            price_change = (current_price - trade.price) / trade.price
            if price_change > 0.1:  # If price increased by 10% or more
                new_stop_loss = current_price * (1 - chain_params['stop_loss_percent'] * 0.5)  # Tighter stop loss
            else:
                new_stop_loss = current_price * (1 - chain_params['stop_loss_percent'])

            # Update trade
            trade.stop_loss = new_stop_loss
            db.session.commit()

            return True

        except Exception as e:
            logger.error(f"Error updating stop loss: {str(e)}")
            db.session.rollback()
            return False

    def _adjust_position_size(self, base_size, metrics):
        """Adjust position size based on market conditions"""
        multiplier = 1.0

        # Adjust for volatility
        if hasattr(metrics, 'bollinger') and metrics.bb_upper and metrics.bb_lower:
            volatility = (metrics.bb_upper - metrics.bb_lower) / metrics.vwap
            if volatility > 0.1:  # High volatility
                multiplier *= self.risk_multipliers['high_volatility']

        # Adjust for trend strength
        if hasattr(metrics, 'adx_value') and metrics.adx_value:
            if metrics.adx_value > 25:  # Strong trend
                multiplier *= self.risk_multipliers['strong_trend']

        # Adjust for liquidity
        if metrics.liquidity < self.default_params['min_liquidity_required'] * 2:
            multiplier *= self.risk_multipliers['low_liquidity']

        # Adjust for sentiment
        if hasattr(metrics, 'social_score') and metrics.social_score:
            if metrics.social_score > 7:
                multiplier *= self.risk_multipliers['high_sentiment']

        return base_size * multiplier

    def _calculate_risk_multiplier(self, metrics):
        """Calculate risk multiplier based on various factors"""
        multiplier = 1.0

        # Technical indicators weight
        if metrics.rsi:
            if 30 <= metrics.rsi <= 70:
                multiplier *= 1.1
            else:
                multiplier *= 0.9

        # Trend strength weight
        if metrics.trend_strength:
            if metrics.trend_strength > 0.7:
                multiplier *= 1.2
            elif metrics.trend_strength < 0.3:
                multiplier *= 0.8

        # Social sentiment weight
        if metrics.social_score:
            if metrics.social_score > 7:
                multiplier *= 1.1
            elif metrics.social_score < 4:
                multiplier *= 0.9

        return multiplier

    def _get_portfolio_value(self):
        """Get total portfolio value"""
        try:
            # Get all active trades
            trades = Trade.query.filter_by(trade_type='BUY').all()

            # Calculate total value including Solana trades
            total_value = sum(trade.price * trade.amount for trade in trades)

            return total_value
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {str(e)}")
            return 0

    def _get_chain_allocation(self, chain):
        """Calculate current allocation for a specific chain"""
        try:
            # Calculate chain value
            chain_trades = Trade.query.filter_by(
                chain=chain,
                trade_type='BUY'
            ).all()

            chain_value = sum(trade.price * trade.amount for trade in chain_trades)
            portfolio_value = self._get_portfolio_value()

            return chain_value / portfolio_value if portfolio_value > 0 else 0

        except Exception as e:
            logger.error(f"Error calculating chain allocation: {str(e)}")
            return 0

    def _get_open_positions_count(self):
        """Get number of currently open positions"""
        try:
            return Trade.query.filter_by(trade_type='BUY').count()
        except Exception as e:
            logger.error(f"Error counting open positions: {str(e)}")
            return 0

    def _validate_technical_indicators(self, metrics):
        """Validate if technical indicators are favorable"""
        try:
            # RSI Check
            if metrics.rsi:
                if metrics.rsi < 20 or metrics.rsi > 80:
                    return False

            # MACD Check
            if metrics.macd_interpretation:
                if metrics.macd_interpretation == 'strong_sell':
                    return False

            # Trend Strength Check
            if metrics.trend_strength:
                if metrics.trend_strength < 0.3:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating technical indicators: {str(e)}")
            return False

    def _validate_market_conditions(self, metrics):
        """Validate overall market conditions"""
        try:
            # Check sentiment
            if metrics.social_score and metrics.social_score < 3:
                return False

            # Check volume
            if metrics.volume_24h and metrics.volume_24h < metrics.liquidity * 0.1:
                return False

            # Check potential score
            if metrics.potential_score and metrics.potential_score < 5:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating market conditions: {str(e)}")
            return False

    def _validate_strategy_rules(self, strategy_id, metrics, amount, price):
        """Validate trade against strategy-specific rules"""
        try:
            strategy = CustomStrategy.query.get(strategy_id)
            if not strategy:
                return False, "Strategy not found"

            if not strategy.is_active:
                return False, "Strategy is not active"

            # Validate entry conditions
            entry_conditions = strategy.entry_conditions

            # Check technical indicators
            for indicator, threshold in entry_conditions.get('thresholds', {}).items():
                if hasattr(metrics, indicator):
                    value = getattr(metrics, indicator)
                    if value is None:
                        continue

                    if threshold.get('min') and value < threshold['min']:
                        return False, f"{indicator} below minimum threshold"

                    if threshold.get('max') and value > threshold['max']:
                        return False, f"{indicator} above maximum threshold"

            # Validate risk parameters
            risk_params = strategy.risk_parameters
            position_value = float(amount) * float(price)

            if position_value > risk_params.get('max_position_value', float('inf')):
                return False, "Position value exceeds strategy maximum"

            return True, "Strategy rules validated"

        except Exception as e:
            logger.error(f"Error validating strategy rules: {str(e)}")
            return False, f"Strategy validation error: {str(e)}"

    def _validate_strategy_allocation(self, strategy_id, position_value):
        """Validate if the position meets strategy allocation limits"""
        try:
            strategy = CustomStrategy.query.get(strategy_id)
            if not strategy:
                return False

            # Calculate current strategy allocation
            strategy_trades = Trade.query.filter_by(
                strategy_id=strategy_id,
                trade_type='BUY'
            ).all()

            strategy_allocation = sum(
                trade.price * trade.amount for trade in strategy_trades
            )

            portfolio_value = self._get_portfolio_value()

            # Check if new position would exceed strategy allocation limit
            new_allocation = (strategy_allocation + position_value) / portfolio_value
            return new_allocation <= self.strategy_params['max_strategy_allocation']

        except Exception as e:
            logger.error(f"Error validating strategy allocation: {str(e)}")
            return False

    def _validate_strategy_performance(self, strategy_id):
        """Validate if strategy performance meets minimum requirements"""
        try:
            strategy = CustomStrategy.query.get(strategy_id)
            if not strategy:
                return False

            # Check win rate
            if strategy.total_trades > 10:  # Only check after minimum number of trades
                if strategy.win_rate < self.strategy_params['min_win_rate'] * 100:
                    return False

            # Check daily risk
            daily_trades = Trade.query.filter(
                Trade.strategy_id == strategy_id,
                Trade.timestamp >= datetime.utcnow().date()
            ).all()

            daily_risk = sum(
                abs(trade.price * trade.amount) for trade in daily_trades
            ) / self._get_portfolio_value()

            return daily_risk <= self.strategy_params['max_daily_risk']

        except Exception as e:
            logger.error(f"Error validating strategy performance: {str(e)}")
            return False

    def _get_market_conditions(self, metrics):
        """Get current market conditions for position sizing"""
        return {
            'trend_strength': getattr(metrics, 'trend_strength', 0),
            'volatility': self._calculate_volatility(metrics),
            'sentiment': getattr(metrics, 'social_score', 0) / 10,
            'volume_profile': getattr(metrics, 'volume_24h', 0) / getattr(metrics, 'liquidity', 1)
        }

    def _calculate_volatility(self, metrics):
        """Calculate current volatility level"""
        try:
            if metrics.bb_upper and metrics.bb_lower and metrics.vwap:
                return (metrics.bb_upper - metrics.bb_lower) / metrics.vwap
            return 0
        except Exception:
            return 0

    def get_market_metrics(self, token_address: str, chain: str) -> Dict[str, Any]:
        """Get current market metrics for a token"""
        try:
            metrics = TokenMetrics.query.filter_by(
                token_address=token_address,
                chain=chain
            ).first()

            if not metrics:
                return {
                    'token_address': token_address,
                    'chain': chain,
                    'liquidity': 0,
                    'volume_24h': 0,
                    'holder_count': 0,
                    'max_position_size': self.chain_params.get(chain, self.default_params)['max_position_size']
                }

            return {
                'token_address': token_address,
                'chain': chain,
                'liquidity': metrics.liquidity,
                'volume_24h': metrics.volume_24h,
                'holder_count': metrics.holder_count,
                'max_position_size': self.chain_params.get(chain, self.default_params)['max_position_size'],
                'risk_score': metrics.risk_score if hasattr(metrics, 'risk_score') else 5.0,
                'volatility': metrics.volatility if hasattr(metrics, 'volatility') else 0.0
            }

        except Exception as e:
            logger.error(f"Error getting market metrics: {str(e)}")
            return {
                'token_address': token_address,
                'chain': chain,
                'error': str(e)
            }

    def apply_risk_constraints(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Apply risk management constraints to a trading opportunity"""
        try:
            if not opportunity or 'error' in opportunity:
                return opportunity

            chain = opportunity.get('chain', 'bsc')
            chain_params = self.chain_params.get(chain, self.default_params)

            # Apply position size constraints
            max_position = min(
                chain_params['max_position_size'],
                opportunity.get('position_size', 0)
            )
            opportunity['position_size'] = max_position

            # Apply risk level adjustments
            risk_level = opportunity.get('risk_level', 0.5)
            if risk_level > 0.8:  # High risk
                opportunity['position_size'] *= 0.7  # Reduce position size
            elif risk_level < 0.3:  # Low risk
                opportunity['position_size'] *= 1.2  # Increase position size

            # Add risk parameters
            opportunity['stop_loss'] = chain_params['stop_loss_percent']
            opportunity['take_profit'] = chain_params['take_profit_percent']
            opportunity['max_slippage'] = chain_params['max_slippage']

            return opportunity

        except Exception as e:
            logger.error(f"Error applying risk constraints: {str(e)}")
            return {'error': str(e)}

    def get_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions"""
        try:
            # Get latest metrics across all chains
            latest_metrics = TokenMetrics.query.order_by(
                TokenMetrics.timestamp.desc()
            ).limit(100).all()

            if not latest_metrics:
                return {
                    'market_sentiment': 'neutral',
                    'average_volatility': 0.0,
                    'risk_level': 'medium',
                    'timestamp': datetime.utcnow()
                }

            # Calculate average metrics
            avg_volatility = sum(
                getattr(m, 'volatility', 0) for m in latest_metrics
            ) / len(latest_metrics)

            # Determine market sentiment
            sentiment_score = sum(
                getattr(m, 'sentiment_score', 5) for m in latest_metrics
            ) / len(latest_metrics)

            sentiment = 'bearish' if sentiment_score < 4 else 'bullish' if sentiment_score > 6 else 'neutral'

            return {
                'market_sentiment': sentiment,
                'average_volatility': avg_volatility,
                'risk_level': 'high' if avg_volatility > 0.1 else 'low' if avg_volatility < 0.05 else 'medium',
                'timestamp': datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"Error getting market conditions: {str(e)}")
            return {
                'error': str(e),
                'market_sentiment': 'neutral',
                'timestamp': datetime.utcnow()
            }

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        try:
            trades = Trade.query.filter_by(trade_type='BUY').all()

            total_positions = len(trades)
            total_value = sum(trade.price * trade.amount for trade in trades)

            # Calculate risk metrics
            metrics = {
                'total_positions': total_positions,
                'total_exposure': total_value,
                'max_drawdown': self._calculate_max_drawdown(trades),
                'risk_exposure_ratio': total_value / self._get_portfolio_value() if self._get_portfolio_value() > 0 else 0,
                'chain_diversification': self._calculate_chain_diversification(),
                'timestamp': datetime.utcnow()
            }

            return metrics

        except Exception as e:
            logger.error(f"Error getting risk metrics: {str(e)}")
            return {'error': str(e)}

    def update_parameters(self, params: Dict[str, Any]):
        """Update risk management parameters"""
        try:
            # Update risk tolerance
            if 'risk_tolerance' in params:
                for chain in self.chain_params:
                    self.chain_params[chain]['max_position_size'] *= params['risk_tolerance']
                self.default_params['max_position_size'] *= params['risk_tolerance']

            # Update take profit levels
            if 'take_profit_adjustment' in params:
                for chain in self.chain_params:
                    self.chain_params[chain]['take_profit_percent'] *= params['take_profit_adjustment']
                self.default_params['take_profit_percent'] *= params['take_profit_adjustment']

            # Update stop loss levels
            if 'stop_loss_adjustment' in params:
                for chain in self.chain_params:
                    self.chain_params[chain]['stop_loss_percent'] *= params['stop_loss_adjustment']
                self.default_params['stop_loss_percent'] *= params['stop_loss_adjustment']

            logger.info("Risk parameters updated successfully")

        except Exception as e:
            logger.error(f"Error updating parameters: {str(e)}")
            raise

    def _calculate_max_drawdown(self, trades: List[Trade]) -> float:
        """Calculate maximum drawdown from trade history"""
        if not trades:
            return 0.0

        profits = [trade.profit for trade in trades if trade.profit is not None]
        if not profits:
            return 0.0

        peak = float('-inf')
        max_drawdown = 0

        for profit in profits:
            if profit > peak:
                peak = profit
            drawdown = (peak - profit) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_chain_diversification(self) -> float:
        """Calculate chain diversification score"""
        try:
            chain_allocations = {
                chain: self._get_chain_allocation(chain)
                for chain in self.chain_params.keys()
            }

            # Calculate Herfindahl-Hirschman Index (HHI)
            hhi = sum(alloc * alloc for alloc in chain_allocations.values())

            # Convert to diversification score (1 - normalized HHI)
            # HHI ranges from 1/n to 1, where n is number of chains
            n = len(self.chain_params)
            normalized_hhi = (hhi - 1/n) / (1 - 1/n) if n > 1 else 1

            return 1 - normalized_hhi

        except Exception as e:
            logger.error(f"Error calculating chain diversification: {str(e)}")
            return 0.0

    def assess_chain_risk(self, chain: str) -> Dict[str, Any]:
        """Assess risk metrics for a specific chain"""
        try:
            # Get chain-specific parameters
            chain_params = self.chain_params.get(chain, self.default_params)

            # Get recent trades on this chain
            trades = Trade.query.filter_by(chain=chain)\
                              .order_by(Trade.timestamp.desc())\
                              .limit(50).all()

            # Calculate metrics
            total_volume = sum(trade.amount * trade.price for trade in trades) if trades else 0
            profitable_trades = sum(1 for trade in trades if trade.profit and trade.profit > 0)
            win_rate = profitable_trades / len(trades) if trades else 0

            # Calculate risk score based on multiple factors
            risk_factors = {
                'liquidity': 1.0 if total_volume > chain_params['min_liquidity_required'] else 0.5,
                'win_rate': win_rate,
                'chain_allocation': self._get_chain_allocation(chain)
            }

            # Calculate weighted risk score (lower is better)
            weights = {'liquidity': 0.4, 'win_rate': 0.4, 'chain_allocation': 0.2}
            risk_score = sum(risk_factors[k] * weights[k] for k in weights)

            return {
                'chain': chain,
                'risk_score': risk_score,
                'metrics': {
                    'total_volume': total_volume,
                    'win_rate': win_rate,
                    'allocation': risk_factors['chain_allocation']
                },
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error assessing chain risk for {chain}: {str(e)}")
            return {
                'chain': chain,
                'risk_score': 1.0,  # Maximum risk on error
                'error': str(e)
            }