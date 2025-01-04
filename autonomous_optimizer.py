"""Enhanced autonomous optimization system for AI10X trading platform"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy import func
from app import db
from models import SystemMetrics, OptimizationEvent, Trade
import numpy as np

logger = logging.getLogger(__name__)

class AutonomousOptimizer:
    """
    Autonomous system optimizer with self-learning capabilities.
    Analyzes system performance and automatically adjusts parameters
    for optimal operation.
    """

    def __init__(self):
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': 80.0,  # Max CPU usage percentage
            'memory_usage': 80.0,  # Max memory usage percentage
            'api_latency': 2000,  # Max API latency in ms
            'error_rate': 0.05,  # Max error rate (5%)
            'success_rate': 0.95,  # Min success rate (95%)
            'optimization_score': 0.7  # Min optimization score
        }

        # Optimization parameters
        self.check_interval = 300  # Check every 5 minutes
        self.analysis_window = 3600  # Analyze last hour of data
        self.last_optimization = datetime.utcnow()
        self.consecutive_failures = 0
        self.max_failures = 3

        # Learning rates for parameter adjustments
        self.learning_rates = {
            'thresholds': 0.05,
            'position_size': 0.1,
            'risk_params': 0.05
        }

        logger.info("Autonomous optimizer initialized with enhanced capabilities")

    def analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze system performance metrics"""
        try:
            # Get recent metrics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(seconds=self.analysis_window)
            
            metrics = SystemMetrics.query.filter(
                SystemMetrics.timestamp.between(start_time, end_time)
            ).all()

            if not metrics:
                return {'status': 'insufficient_data'}

            # Calculate performance indicators
            analysis = {
                'cpu_usage': np.mean([m.cpu_usage for m in metrics]),
                'memory_usage': np.mean([m.memory_usage for m in metrics]),
                'api_latency': np.mean([m.api_latency for m in metrics]),
                'success_rate': np.mean([m.success_rate for m in metrics]),
                'error_count': sum(m.error_count for m in metrics),
                'warning_count': sum(m.warning_count for m in metrics),
                'optimization_score': np.mean([m.optimization_score for m in metrics])
            }

            # Add trend analysis
            analysis['trends'] = self._calculate_trends(metrics)
            
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing system performance: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _calculate_trends(self, metrics: list) -> Dict[str, float]:
        """Calculate trends in system metrics"""
        try:
            if len(metrics) < 2:
                return {}

            trends = {}
            for attr in ['cpu_usage', 'memory_usage', 'api_latency', 'success_rate']:
                values = [getattr(m, attr) for m in metrics]
                trends[attr] = (values[-1] - values[0]) / len(values)  # Simple linear trend

            return trends

        except Exception as e:
            logger.error(f"Error calculating trends: {str(e)}")
            return {}

    def optimize_system_parameters(self, analysis: Dict[str, Any]) -> bool:
        """
        Optimize system parameters based on performance analysis
        Returns True if optimization was successful
        """
        try:
            if analysis.get('status') == 'insufficient_data':
                logger.info("Insufficient data for optimization")
                return False

            changes_made = False
            
            # Check if optimization is needed
            if self._should_optimize(analysis):
                logger.info("Starting system optimization")

                # Adjust thresholds based on performance
                self._adjust_thresholds(analysis)

                # Optimize trading parameters
                self._optimize_trading_parameters(analysis)

                # Record optimization event
                self._record_optimization_event(analysis)

                changes_made = True
                self.consecutive_failures = 0
                logger.info("System optimization completed successfully")

            return changes_made

        except Exception as e:
            logger.error(f"Error optimizing system parameters: {str(e)}")
            self.consecutive_failures += 1
            return False

    def _should_optimize(self, analysis: Dict[str, Any]) -> bool:
        """Determine if optimization is needed"""
        try:
            # Check time since last optimization
            if (datetime.utcnow() - self.last_optimization).total_seconds() < self.check_interval:
                return False

            # Check if any thresholds are exceeded
            for metric, threshold in self.thresholds.items():
                if metric in analysis:
                    if metric in ['success_rate', 'optimization_score']:
                        if analysis[metric] < threshold:
                            return True
                    else:
                        if analysis[metric] > threshold:
                            return True

            # Check error rate
            total_ops = len(analysis.get('trends', {}))
            if total_ops > 0:
                error_rate = analysis['error_count'] / total_ops
                if error_rate > self.thresholds['error_rate']:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking optimization need: {str(e)}")
            return False

    def _adjust_thresholds(self, analysis: Dict[str, Any]):
        """Adjust system thresholds based on performance"""
        try:
            trends = analysis.get('trends', {})
            for metric, trend in trends.items():
                if metric in self.thresholds:
                    # Adjust threshold based on trend direction and magnitude
                    adjustment = trend * self.learning_rates['thresholds']
                    current = self.thresholds[metric]
                    
                    if metric in ['success_rate', 'optimization_score']:
                        # For positive metrics, increase threshold if performing well
                        if analysis[metric] > current:
                            self.thresholds[metric] = min(0.99, current + abs(adjustment))
                    else:
                        # For negative metrics, decrease threshold if struggling
                        if analysis[metric] > current:
                            self.thresholds[metric] = max(0.01, current - abs(adjustment))

            logger.info("Thresholds adjusted based on performance trends")

        except Exception as e:
            logger.error(f"Error adjusting thresholds: {str(e)}")

    def _optimize_trading_parameters(self, analysis: Dict[str, Any]):
        """Optimize trading parameters based on performance analysis"""
        try:
            # Get recent trades for analysis
            recent_trades = Trade.query.order_by(
                Trade.timestamp.desc()
            ).limit(100).all()

            if not recent_trades:
                return

            # Calculate trade performance metrics
            success_rate = len([t for t in recent_trades if t.profit > 0]) / len(recent_trades)
            avg_profit = sum(t.profit for t in recent_trades) / len(recent_trades)
            
            # Adjust position sizes based on performance
            if success_rate > 0.7 and avg_profit > 0:
                # Increase position sizes slightly
                self._adjust_position_sizes(1 + self.learning_rates['position_size'])
            elif success_rate < 0.5 or avg_profit < 0:
                # Decrease position sizes
                self._adjust_position_sizes(1 - self.learning_rates['position_size'])

            # Adjust risk parameters
            self._adjust_risk_parameters(analysis, success_rate)

            logger.info("Trading parameters optimized based on performance")

        except Exception as e:
            logger.error(f"Error optimizing trading parameters: {str(e)}")

    def _adjust_position_sizes(self, multiplier: float):
        """Adjust position sizes with safety checks"""
        try:
            from trade_executor import TradeExecutor
            
            executor = TradeExecutor()
            for chain in executor.risk_params:
                current = executor.risk_params[chain]['max_position']
                new_size = current * multiplier
                # Ensure position size stays within safe limits
                executor.risk_params[chain]['max_position'] = max(0.01, min(0.2, new_size))

        except Exception as e:
            logger.error(f"Error adjusting position sizes: {str(e)}")

    def _adjust_risk_parameters(self, analysis: Dict[str, Any], success_rate: float):
        """Adjust risk management parameters"""
        try:
            from trade_executor import TradeExecutor
            
            executor = TradeExecutor()
            for chain in executor.risk_params:
                params = executor.risk_params[chain]
                
                # Adjust stop loss and take profit based on performance
                if success_rate > 0.7:
                    # More aggressive settings
                    params['stop_loss'] *= (1 - self.learning_rates['risk_params'])
                    params['take_profit'] *= (1 + self.learning_rates['risk_params'])
                else:
                    # More conservative settings
                    params['stop_loss'] *= (1 + self.learning_rates['risk_params'])
                    params['take_profit'] *= (1 - self.learning_rates['risk_params'])

                # Ensure parameters stay within safe ranges
                params['stop_loss'] = max(0.01, min(0.1, params['stop_loss']))
                params['take_profit'] = max(0.05, min(0.5, params['take_profit']))

        except Exception as e:
            logger.error(f"Error adjusting risk parameters: {str(e)}")

    def _record_optimization_event(self, analysis: Dict[str, Any]):
        """Record optimization event in database"""
        try:
            event = OptimizationEvent(
                timestamp=datetime.utcnow(),
                metrics_before=analysis,
                thresholds_after=self.thresholds,
                success=True
            )
            db.session.add(event)
            db.session.commit()
            self.last_optimization = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error recording optimization event: {str(e)}")
            db.session.rollback()
