"""Base monitoring functionality shared across autonomous components with Solana memecoin support"""
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List

# Configure logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringBase:
    """Base class for monitoring functionality with Solana memecoin support"""

    def __init__(self):
        self.initialized = False
        self.metrics_interval = 60  # seconds
        self.metrics_history = []
        self.error_history = []
        self.solana_metrics = {
            'min_safety_score': 80,
            'min_liquidity': 1000,  # Minimum liquidity in USD
            'max_mint_risk': 0.7    # Maximum acceptable mint risk
        }
        self.initialized = True
        logger.info("Base monitoring initialized successfully")

    def collect_base_metrics(self) -> Dict[str, Any]:
        """Collect basic system metrics including Solana network stats"""
        try:
            metrics = {
                'timestamp': datetime.utcnow(),
                'monitoring_status': 'active' if self.initialized else 'inactive',
                'metrics_count': len(self.metrics_history),
                'error_count': len(self.error_history),
                'solana_metrics': self.solana_metrics
            }
            return metrics
        except Exception as e:
            logger.error(f"Error collecting base metrics: {str(e)}")
            return {}

    def validate_token_safety(self, token_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Solana memecoin safety based on configured thresholds"""
        try:
            safety_checks = {
                'passed': True,
                'warnings': [],
                'risk_factors': []
            }

            # Check safety score
            if token_metrics.get('safety_score', 0) < self.solana_metrics['min_safety_score']:
                safety_checks['passed'] = False
                safety_checks['risk_factors'].append('Low safety score')

            # Check liquidity
            if token_metrics.get('liquidity', 0) < self.solana_metrics['min_liquidity']:
                safety_checks['warnings'].append('Low liquidity')

            # Check mint status
            if token_metrics.get('mint_enabled', True):
                safety_checks['risk_factors'].append('Minting is enabled')

            # Check liquidity lock
            if not token_metrics.get('liquidity_locked', False):
                safety_checks['risk_factors'].append('Liquidity not locked')

            return safety_checks
        except Exception as e:
            logger.error(f"Error validating token safety: {str(e)}")
            return {'error': str(e), 'passed': False}

    def record_error(self, error: Exception):
        """Record an error with timestamp"""
        try:
            self.error_history.append({
                'timestamp': datetime.utcnow(),
                'error_type': type(error).__name__,
                'error_message': str(error)
            })

            # Keep only recent history
            if len(self.error_history) > 100:
                self.error_history = self.error_history[-100:]
        except Exception as e:
            logger.error(f"Error recording error: {str(e)}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status including Solana monitoring"""
        try:
            return {
                'initialized': self.initialized,
                'error_count': len(self.error_history),
                'recent_errors': self.error_history[-5:] if self.error_history else [],
                'metrics_recorded': len(self.metrics_history),
                'solana_monitoring': {
                    'safety_thresholds': self.solana_metrics,
                    'status': 'active' if self.initialized else 'inactive'
                }
            }
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {'error': str(e)}

    def update_monitoring_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        """Update Solana monitoring parameters manually"""
        try:
            if 'min_safety_score' in new_parameters:
                self.solana_metrics['min_safety_score'] = float(new_parameters['min_safety_score'])
            if 'min_liquidity' in new_parameters:
                self.solana_metrics['min_liquidity'] = float(new_parameters['min_liquidity'])
            if 'max_mint_risk' in new_parameters:
                self.solana_metrics['max_mint_risk'] = float(new_parameters['max_mint_risk'])

            logger.info(f"Updated Solana monitoring parameters: {self.solana_metrics}")
            return True
        except Exception as e:
            logger.error(f"Error updating monitoring parameters: {str(e)}")
            return False