"""Core monitoring functionality with manual optimization controls"""
import logging
import psutil
import time
from datetime import datetime
from typing import Dict, Any, Optional
from decimal import Decimal

from monitoring_base import MonitoringBase

# Configure logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutonomousMonitor(MonitoringBase):
    """Autonomous Monitor with manual optimization controls"""

    def __init__(self, app=None):
        """Initialize monitor with basic parameters"""
        super().__init__()  # Initialize base monitoring
        self.app = app

        # Manual control parameters
        self.params = {
            'monitoring_interval': 60,  # seconds
            'metrics_threshold': Decimal('0.7'),
            'alert_threshold': Decimal('0.9'),
            'sampling_rate': Decimal('1.0'),
            # Solana-specific parameters
            'safety_score_threshold': Decimal('80.0'),
            'min_liquidity_threshold': Decimal('1000.0'),
            'max_mint_risk': Decimal('0.7'),
            'raydium_pool_required': True
        }

        logger.info("Monitoring components initialized successfully")

    def update_parameters(self, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """Update monitoring parameters manually"""
        try:
            for key, value in new_params.items():
                if key in self.params:
                    # Convert numerical values to Decimal
                    if isinstance(value, (int, float)):
                        value = Decimal(str(value))
                    self.params[key] = value

            logger.info(f"Parameters updated: {self.params}")
            return self.params
        except Exception as e:
            logger.error(f"Error updating parameters: {str(e)}")
            self.record_error(e)
            return {}

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics with sampling rate control"""
        try:
            # Get base metrics first
            metrics = self.collect_base_metrics()
            if not metrics:
                return {}

            # Add system-specific metrics
            metrics.update({
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'solana_metrics': {
                    'safety_score_threshold': float(self.params['safety_score_threshold']),
                    'min_liquidity_threshold': float(self.params['min_liquidity_threshold']),
                    'max_mint_risk': float(self.params['max_mint_risk']),
                    'raydium_pool_required': self.params['raydium_pool_required']
                }
            })

            # Add to metrics history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]

            return metrics
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            self.record_error(e)
            return {}

    def calculate_health_score(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system health score using configurable thresholds"""
        try:
            if not metrics:
                return {'score': Decimal('0.5'), 'status': 'unknown'}

            # Calculate base score
            base_score = self.calculate_base_score(metrics)

            # Add weighted component scores
            scores = {
                'cpu': Decimal('1') - metrics['cpu_percent'] / Decimal('100'),
                'memory': Decimal('1') - metrics['memory_percent'] / Decimal('100'),
                'eth_gas': Decimal('1') - metrics.get('eth_gas_price', Decimal('0')) / Decimal('200'),
                'llm_success': metrics.get('llm_success_rate', Decimal('1')),
                'solana_safety': metrics.get('safety_score', Decimal('80')) / Decimal('100'),
                'base': base_score
            }

            # Calculate final score
            final_score = sum(scores.values()) / Decimal(str(len(scores)))

            # Determine status based on thresholds
            status = 'healthy'
            if final_score < self.params['metrics_threshold']:
                status = 'degraded'
            if final_score < self.params['alert_threshold']:
                status = 'critical'

            return {
                'score': final_score,
                'status': status,
                'component_scores': scores
            }
        except Exception as e:
            logger.error(f"Error calculating health score: {str(e)}")
            self.record_error(e)
            return {'score': Decimal('0.5'), 'status': 'error'}

    def validate_memecoin_safety(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Solana memecoin safety against configured parameters"""
        try:
            safety_result = {
                'passed': True,
                'warnings': [],
                'risk_factors': [],
                'timestamp': datetime.utcnow().isoformat()
            }

            # Check safety score
            safety_score = Decimal(str(token_data.get('safety_score', 0)))
            if safety_score < self.params['safety_score_threshold']:
                safety_result['passed'] = False
                safety_result['risk_factors'].append(
                    f'Low safety score: {safety_score} < {self.params["safety_score_threshold"]}'
                )

            # Check liquidity
            liquidity = Decimal(str(token_data.get('liquidity', 0)))
            if liquidity < self.params['min_liquidity_threshold']:
                safety_result['warnings'].append(
                    f'Low liquidity: {liquidity} < {self.params["min_liquidity_threshold"]}'
                )

            # Check mint status
            if token_data.get('mint_enabled', True):
                safety_result['risk_factors'].append('Minting is enabled')

            # Check Raydium pool
            if self.params['raydium_pool_required'] and not token_data.get('raydium_pool_exists', False):
                safety_result['warnings'].append('No Raydium pool found')

            # Add overall assessment
            safety_result['assessment'] = {
                'safety_score': float(safety_score),
                'liquidity': float(liquidity),
                'risk_level': 'high' if len(safety_result['risk_factors']) > 2 else 'medium' if safety_result['risk_factors'] else 'low'
            }

            return safety_result

        except Exception as e:
            logger.error(f"Error validating memecoin safety: {str(e)}")
            self.record_error(e)
            return {
                'passed': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def run_monitoring_loop(self):
        """Run monitoring loop with configurable interval"""
        if not self.initialized:
            logger.error("Monitor not initialized properly")
            return

        logger.info("Starting monitoring loop...")

        try:
            while True:
                # Collect metrics
                metrics = self.collect_metrics()
                if metrics:
                    health_status = self.calculate_health_score(metrics)
                    metrics['health_score'] = health_status['score']
                    metrics['system_status'] = health_status['status']
                    logger.debug(f"Collected metrics: {metrics}")

                # Sleep based on configured interval
                time.sleep(self.params['monitoring_interval'])

        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            self.record_error(e)
            time.sleep(10)  # Basic error backoff