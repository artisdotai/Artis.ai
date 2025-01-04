"""
Trading service for Eliza Framework implementation
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.database import db
from src.models.trading import Trade, ChainMetrics
from src.services.pattern_recognition import PatternRecognitionService

logger = logging.getLogger(__name__)

class TradingService:
    """Trading service implementation"""

    def __init__(self):
        """Initialize trading service with chain configurations"""
        self.supported_chains = {
            'ETH': {'max_slippage': 0.02, 'gas_limit': 500000},
            'BSC': {'max_slippage': 0.03, 'gas_limit': 1000000},
            'POLYGON': {'max_slippage': 0.02, 'gas_limit': 700000},
            'ARB': {'max_slippage': 0.02, 'gas_limit': 400000},
            'SOLANA': {'max_slippage': 0.01, 'gas_limit': None},
        }
        self.pattern_recognition = PatternRecognitionService()

    def execute_trade(self, chain: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a trade on specified chain"""
        try:
            if chain not in self.supported_chains:
                raise ValueError(f"Unsupported chain: {chain}")

            # Analyze patterns before trade execution
            pattern_analysis = None
            if 'price_data' in params:
                pattern_analysis = self.pattern_recognition.analyze_patterns(params['price_data'])
                trade_signal = self._evaluate_pattern_signals(pattern_analysis)
                if not trade_signal['execute']:
                    logger.info(f"Trade rejected based on pattern analysis: {trade_signal['reason']}")
                    return None

                # Adjust position size based on pattern confidence
                params['position_size'] = self._adjust_position_size(
                    params['position_size'],
                    pattern_analysis['confidence'],
                    trade_signal['risk_score']
                )

            # Create trade record
            trade = Trade(
                chain=chain,
                token_address=params['token_address'],
                token_symbol=params.get('token_symbol'),
                entry_price=params['entry_price'],
                position_size=params['position_size'],
                risk_score=params.get('risk_score', 0.5),
                strategy=params.get('strategy', 'pattern_based'),
                pattern_data=pattern_analysis if pattern_analysis else None
            )

            db.session.add(trade)
            db.session.commit()

            return {
                'trade_id': trade.id,
                'status': 'executed',
                'timestamp': datetime.utcnow().isoformat(),
                'chain': chain,
                'details': {
                    'token_address': trade.token_address,
                    'entry_price': trade.entry_price,
                    'position_size': trade.position_size,
                    'pattern_analysis': pattern_analysis
                }
            }

        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}")
            return None

    def _evaluate_pattern_signals(self, pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trading signals from pattern analysis"""
        if not pattern_analysis:
            return {'execute': False, 'reason': 'No pattern analysis available', 'risk_score': 0.5}

        confidence_threshold = 0.65
        risk_score = 0.5

        # Evaluate pattern confidence
        if pattern_analysis.get('confidence', 0) < confidence_threshold:
            return {
                'execute': False,
                'reason': 'Pattern confidence below threshold',
                'risk_score': risk_score
            }

        # Check for strong signals
        signals = []
        risk_multiplier = 1.0

        for pattern in pattern_analysis.get('patterns', []):
            pattern_type = pattern.get('type')
            metrics = pattern.get('metrics', {})

            if pattern_type == 'head_shoulders':
                if metrics.get('volume_confirmation'):
                    signals.append('strong_reversal')
                    risk_multiplier *= 0.8
                if metrics.get('symmetry', 0) > 0.8:
                    signals.append('high_quality_pattern')
                    risk_multiplier *= 0.9

            elif pattern_type == 'triangle':
                if metrics.get('slope_convergence', 0) > 0.8:
                    signals.append('strong_continuation')
                    risk_multiplier *= 0.9
                if metrics.get('volume_decline'):
                    signals.append('breakout_potential')
                    risk_multiplier *= 0.95

        # Calculate final risk score
        risk_score = max(0.1, min(1.0, risk_score * risk_multiplier))

        return {
            'execute': len(signals) > 0,
            'reason': f"Pattern signals: {', '.join(signals)}",
            'risk_score': risk_score,
            'signals': signals
        }

    def _adjust_position_size(self, base_size: float, pattern_confidence: float, risk_score: float) -> float:
        """Adjust position size based on pattern confidence and risk score"""
        confidence_multiplier = pattern_confidence if pattern_confidence > 0.5 else 0.5
        risk_multiplier = 1 - (risk_score * 0.5)  # Higher risk reduces position size
        return base_size * confidence_multiplier * risk_multiplier

    def get_chain_metrics(self, chain: str) -> Dict[str, Any]:
        """Get metrics for specific chain"""
        try:
            metrics = ChainMetrics.query.filter_by(chain=chain).first()
            if not metrics:
                return {
                    'price': 0,
                    'market_cap': 0,
                    'liquidity': 0,
                    'volume_24h': 0,
                    'holder_count': 0,
                    'status': 'inactive'
                }

            return {
                'price': 0,  # Price would be fetched from external API
                'market_cap': 0,  # Market cap would be calculated
                'liquidity': metrics.total_liquidity or 0,
                'volume_24h': metrics.daily_volume or 0,
                'holder_count': 0,  # Holder count would be fetched from chain
                'status': metrics.status or 'active',
                'gas_price': metrics.gas_price,
                'active_pairs': metrics.active_pairs,
                'pattern_metrics': self.pattern_recognition.get_recent_patterns(chain)
            }

        except Exception as e:
            logger.error(f"Error getting chain metrics: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def update_chain_metrics(self, chain: str, metrics: Dict[str, Any]) -> bool:
        """Update metrics for specific chain"""
        try:
            chain_metrics = ChainMetrics.query.filter_by(chain=chain).first()
            if not chain_metrics:
                chain_metrics = ChainMetrics(chain=chain)

            # Update metrics
            chain_metrics.total_liquidity = metrics.get('total_liquidity')
            chain_metrics.daily_volume = metrics.get('daily_volume')
            chain_metrics.active_pairs = metrics.get('active_pairs')
            chain_metrics.gas_price = metrics.get('gas_price')
            chain_metrics.status = metrics.get('status', 'active')

            db.session.add(chain_metrics)
            db.session.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating chain metrics: {str(e)}")
            return False

    def get_active_trades(self, chain: Optional[str] = None) -> Dict[str, Any]:
        """Get active trades with optional chain filter"""
        try:
            query = Trade.query.filter_by(status='open')
            if chain:
                query = query.filter_by(chain=chain)

            trades = query.all()
            return {
                'count': len(trades),
                'trades': [{
                    'id': t.id,
                    'chain': t.chain,
                    'token_address': t.token_address,
                    'token_symbol': t.token_symbol,
                    'entry_price': t.entry_price,
                    'current_price': t.exit_price,
                    'position_size': t.position_size,
                    'profit_loss': t.profit_loss,
                    'pattern_data': t.pattern_data,
                    'created_at': t.created_at.isoformat()
                } for t in trades]
            }

        except Exception as e:
            logger.error(f"Error fetching active trades: {str(e)}")
            return {'count': 0, 'trades': []}