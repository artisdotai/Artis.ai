"""
Route definitions for the Eliza Framework implementation
"""
import logging
from flask import jsonify, render_template, request
from .services.monitoring import HealthMonitor
from .services.risk_management import RiskManager
from .services.trading import TradingService
from .services.sentiment import SentimentAnalyzer
from .services.llm_analysis import LLMAnalyzer
from .services.tool_connector import ToolConnector

logger = logging.getLogger(__name__)

def init_routes(app):
    """Initialize all application routes"""

    # Initialize services
    health_monitor = HealthMonitor()
    risk_manager = RiskManager()
    trading_service = TradingService()
    sentiment_analyzer = SentimentAnalyzer()
    llm_analyzer = LLMAnalyzer()
    tool_connector = ToolConnector()

    @app.route('/')
    def index():
        """Main dashboard route"""
        return render_template('index.html')

    @app.route('/api/health')
    def health_check():
        """System health check endpoint"""
        try:
            return jsonify(health_monitor.get_system_status())
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/risk/parameters')
    def risk_parameters():
        """Get current risk management parameters"""
        try:
            risk_params = risk_manager.get_current_parameters()
            return jsonify(risk_params)
        except Exception as e:
            logger.error(f"Failed to get risk parameters: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/trades')
    def get_trades():
        """Get active trades endpoint"""
        try:
            trades = trading_service.get_active_trades()
            return jsonify(trades)
        except Exception as e:
            logger.error(f"Error getting trades: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/sentiment/market-mood')
    def get_market_mood():
        """Get market sentiment endpoint"""
        try:
            mood = sentiment_analyzer.get_market_mood()
            return jsonify(mood)
        except Exception as e:
            logger.error(f"Error getting market mood: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/analysis/market/<chain>')
    def get_market_analysis(chain):
        """Get market analysis for specific chain"""
        try:
            metrics = trading_service.get_chain_metrics(chain)
            analysis = llm_analyzer.analyze_market_trends(chain, metrics)
            return jsonify(analysis)
        except Exception as e:
            logger.error(f"Error getting market analysis: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/analysis/llm/status')
    def get_llm_status():
        """Get LLM service status endpoint"""
        try:
            status = llm_analyzer.get_llm_status()
            return jsonify(status)
        except Exception as e:
            logger.error(f"Error getting LLM status: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/tools')
    def get_tools():
        """Get available tools endpoint"""
        try:
            return jsonify(tool_connector.get_available_tools())
        except Exception as e:
            logger.error(f"Error getting tools: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/tools/<tool_id>/execute', methods=['POST'])
    def execute_tool(tool_id):
        """Execute specific tool endpoint"""
        try:
            parameters = request.get_json()
            result = tool_connector.execute_tool(tool_id, parameters)
            if result:
                return jsonify(result)
            return jsonify({'status': 'error', 'message': 'Tool execution failed'}), 400
        except Exception as e:
            logger.error(f"Tool execution error: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    logger.info("Routes initialized successfully")