"""
AI10X Autonomous System Module
=============================

Core functionality for the autonomous trading system with enhanced reliability
and basic monitoring capabilities.
"""

import logging
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime
from monitoring_base import MonitoringBase
from autonomous_monitor import AutonomousMonitor
from technical_analysis import TechnicalAnalyzer
from llm_connector import LLMConnector
from profit_manager import ProfitManager
from risk_manager import RiskManager
from kol_analyzer import KOLAnalyzer
from solana_monitor import SolanaMonitor
from pumpfun_analyzer import PumpFunAnalyzer
from trade_executor import TradeExecutor

logger = logging.getLogger(__name__)

class AutonomousSystem(MonitoringBase):
    def __init__(self, app=None):
        """Initialize autonomous system components"""
        super().__init__()  # Initialize base monitoring
        self.app = app

        # Initialize core components with proper error handling
        try:
            self.monitor = AutonomousMonitor(app)
            self.llm_connector = LLMConnector()
            self.technical_analyzer = TechnicalAnalyzer()
            self.profit_manager = ProfitManager()
            self.risk_manager = RiskManager()
            self.kol_analyzer = KOLAnalyzer()
            self.solana_monitor = SolanaMonitor()
            self.pumpfun_analyzer = PumpFunAnalyzer()
            self.trade_executor = TradeExecutor()

            # Set basic parameters
            self._initialize_parameters()

            logger.info("Initialized autonomous system with all components")
            self.initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize autonomous system: {str(e)}")
            self.record_error(e)
            raise

    def _initialize_parameters(self):
        """Initialize system parameters with safe defaults"""
        # Basic trading parameters
        self.confidence_threshold = Decimal('0.6')
        self.max_position_size = Decimal('0.2')

        # Basic tracking
        self.performance_history = []
        self.error_history = []

    def run_monitoring_loop(self):
        """Main monitoring loop for the autonomous system"""
        if not self.initialized or not self.monitor:
            logger.error("System not properly initialized")
            return

        try:
            logger.info("Starting autonomous system monitoring loop")
            self.monitor.run_monitoring_loop()
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            self.record_error(e)
            raise

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # Get base health metrics
            base_health = self.get_health_status()

            # Add monitor health if available
            monitor_health = self.monitor.get_health_status() if self.monitor else {}

            return {
                'timestamp': datetime.utcnow(),
                'base_health': base_health,
                'monitor_health': monitor_health,
                'components_status': {
                    'technical_analyzer': bool(self.technical_analyzer),
                    'risk_manager': bool(self.risk_manager),
                    'trade_executor': bool(self.trade_executor),
                    'llm_connector': bool(self.llm_connector)
                }
            }

        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            self.record_error(e)
            return {'error': str(e)}

    def analyze_market_conditions(self) -> Dict[str, Any]:
        """Basic market analysis across supported chains"""
        try:
            analysis_results = {}

            # Analyze different chains
            for chain in ['bsc', 'eth', 'polygon', 'arb', 'avax', 'solana']:
                chain_analysis = {
                    'market_metrics': self.risk_manager.get_market_metrics(chain=chain),
                    'sentiment': self.kol_analyzer.get_chain_sentiment(chain),
                    'risk_assessment': self.risk_manager.assess_chain_risk(chain)
                }

                analysis_results[chain] = chain_analysis

            return {
                'timestamp': datetime.utcnow().isoformat(),
                'chain_analysis': analysis_results,
                'system_status': 'healthy' if self.initialized else 'initializing'
            }

        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {'error': str(e)}

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