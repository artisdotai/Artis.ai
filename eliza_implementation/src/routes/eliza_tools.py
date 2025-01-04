"""
ELIZA tool-specific routes for the Eliza Framework implementation
"""
import logging
from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
from ..services.tool_connector import ToolConnector
from ..services.llm_analysis import LLMAnalyzer
from ..services.twitter_analyzer import TwitterAnalyzer
from ..services.pattern_recognition import PatternRecognizer

logger = logging.getLogger(__name__)

def create_eliza_tools_blueprint() -> Blueprint:
    """Create and configure ELIZA tools blueprint"""
    eliza_tools_bp = Blueprint('eliza_tools', __name__)

    # Initialize services
    tool_connector = ToolConnector()
    llm_analyzer = LLMAnalyzer()
    twitter_analyzer = TwitterAnalyzer()
    pattern_recognizer = PatternRecognizer()

    @eliza_tools_bp.route('/tools/pattern_recognition/execute', methods=['POST', 'OPTIONS'])
    @cross_origin()
    def execute_pattern_recognition():
        """Execute pattern recognition analysis"""
        if request.method == 'OPTIONS':
            return jsonify({'status': 'success'}), 200

        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'No data provided'
                }), 400

            # Extract required parameters
            token_data = data.get('token_data')
            pattern_types = data.get('pattern_types', [])

            if not token_data:
                return jsonify({
                    'status': 'error',
                    'message': 'Missing required parameter: token_data'
                }), 400

            # Execute pattern recognition
            result = pattern_recognizer.identify_patterns(token_data, pattern_types)
            logger.info("Pattern recognition executed successfully")
            return jsonify(result)

        except Exception as e:
            logger.error(f"Pattern recognition error: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @eliza_tools_bp.route('/tools', methods=['GET'])
    @cross_origin()
    def get_tools():
        """Get available ELIZA tools endpoint"""
        try:
            tools = tool_connector.get_available_tools()
            logger.info(f"Retrieved {len(tools.get('tools', {}))} available tools")
            return jsonify(tools)
        except Exception as e:
            logger.error(f"Error getting tools: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @eliza_tools_bp.route('/tools/<tool_id>/execute', methods=['POST'])
    @cross_origin()
    def execute_tool(tool_id):
        """Execute specific ELIZA tool endpoint"""
        try:
            parameters = request.get_json()
            if not parameters:
                return jsonify({
                    'status': 'error',
                    'message': 'No parameters provided'
                }), 400

            result = tool_connector.execute_tool(tool_id, parameters)
            if result:
                return jsonify(result)
            return jsonify({
                'status': 'error', 
                'message': 'Tool execution failed'
            }), 400

        except Exception as e:
            logger.error(f"Tool execution error: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @eliza_tools_bp.route('/tools/status', methods=['GET'])
    @cross_origin()
    def get_tool_status():
        """Get ELIZA tools status endpoint"""
        try:
            tools = tool_connector.get_available_tools()
            return jsonify({
                'status': 'success',
                'operational_count': sum(1 for t in tools['tools'].values() 
                                    if t.get('status') == 'operational'),
                'total_count': len(tools['tools']),
                'tools': tools['tools']
            })
        except Exception as e:
            logger.error(f"Error getting tool status: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @eliza_tools_bp.route('/llm/status', methods=['GET'])
    @cross_origin()
    def get_llm_status():
        """Get LLM service status endpoint"""
        try:
            status = llm_analyzer.get_llm_status()
            return jsonify(status)
        except Exception as e:
            logger.error(f"Error getting LLM status: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    return eliza_tools_bp