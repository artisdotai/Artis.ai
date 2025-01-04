"""Routes module for the Flask application"""
from flask import render_template, jsonify, request
import logging
from datetime import datetime
import time
from sqlalchemy import text
from models import SystemMetrics, TelegramSentiment, TelegramMessages
from extensions import db
from solana_health import SolanaHealthCheck
from telegram_monitor import TelegramMonitor

logger = logging.getLogger(__name__)
telegram_monitor = TelegramMonitor()

def setup_routes(app):
    """Setup all application routes with enhanced error handling"""

    @app.route('/')
    def dashboard():
        """Render the main dashboard"""
        try:
            return jsonify({
                'status': 'healthy',
                'message': 'AI Trading Platform API is running'
            })
        except Exception as e:
            logger.error(f"Error rendering dashboard: {str(e)}")
            return jsonify({'error': 'Error loading dashboard'}), 500

    @app.route('/api/telegram/monitor', methods=['POST'])
    def telegram_message():
        """Handle incoming Telegram messages"""
        try:
            message_data = request.json
            if not message_data:
                return jsonify({'error': 'No message data provided'}), 400

            # Validate required fields
            required_fields = ['token_address', 'group_name', 'message_text']
            missing_fields = [field for field in required_fields if not message_data.get(field)]
            if missing_fields:
                return jsonify({
                    'error': 'Missing required fields',
                    'missing_fields': missing_fields
                }), 400

            # Process message with tracing
            request_id = datetime.utcnow().isoformat()
            logger.info(f"Processing Telegram message {request_id} for token {message_data.get('token_address')}")

            success = telegram_monitor.process_message(message_data)
            if success:
                logger.info(f"Successfully processed message {request_id}")
                return jsonify({
                    'status': 'success', 
                    'message': 'Message processed successfully',
                    'request_id': request_id
                })

            logger.error(f"Failed to process message {request_id}")
            return jsonify({
                'error': 'Failed to process message',
                'request_id': request_id
            }), 500

        except Exception as e:
            logger.error(f"Error processing Telegram message: {str(e)}")
            return jsonify({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat()
            }), 500

    @app.route('/api/telegram/sentiment/<token_address>')
    def token_sentiment(token_address):
        """Get sentiment analysis for a specific token"""
        try:
            chain = request.args.get('chain', 'solana')
            request_id = datetime.utcnow().isoformat()
            logger.info(f"Fetching sentiment data {request_id} for token {token_address}")

            sentiment_data = telegram_monitor.get_token_sentiment(token_address, chain)

            # Handle rate limit response
            if isinstance(sentiment_data, tuple) and len(sentiment_data) == 2:
                return jsonify(sentiment_data[0]), sentiment_data[1]

            if sentiment_data:
                logger.info(f"Successfully retrieved sentiment data {request_id}")
                return jsonify({
                    'data': sentiment_data,
                    'request_id': request_id,
                    'timestamp': datetime.utcnow().isoformat()
                })

            logger.warning(f"No sentiment data found {request_id}")
            return jsonify({
                'error': 'No sentiment data found',
                'request_id': request_id
            }), 404

        except Exception as e:
            logger.error(f"Error getting token sentiment: {str(e)}")
            return jsonify({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat()
            }), 500

    @app.route('/api/health')
    def health_check():
        """Health check endpoint with enhanced service verification"""
        try:
            # Check database connection with retry
            db_status = False
            retry_count = 0
            max_retries = 3

            while not db_status and retry_count < max_retries:
                try:
                    db.session.execute(text('SELECT 1'))
                    db.session.commit()
                    db_status = True
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Database health check failed after {max_retries} attempts: {str(e)}")
                    time.sleep(1)  # Short delay between retries

            # Check SpyDefi integration status through telegram monitor
            spydefi_status = not telegram_monitor._circuit_breaker['is_open']

            health_data = {
                'status': 'healthy' if (db_status and spydefi_status) else 'degraded',
                'timestamp': datetime.utcnow().isoformat(),
                'services': {
                    'database': {
                        'connected': db_status,
                        'type': 'postgresql'
                    },
                    'spydefi': {
                        'status': 'operational' if spydefi_status else 'degraded',
                        'circuit_breaker': 'closed' if spydefi_status else 'open',
                        'failures': telegram_monitor._circuit_breaker['failures']
                    }
                }
            }

            return jsonify(health_data)

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 500

    @app.route('/api/monitor/solana/health')
    def check_solana_health():
        """Get Solana network health status"""
        try:
            solana_health = SolanaHealthCheck()
            health_status = solana_health.check_health()
            return jsonify(health_status)
        except Exception as e:
            logger.error(f"Error checking Solana health: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 500

    @app.route('/api/monitor/metrics')
    def get_monitoring_metrics():
        """Get system monitoring metrics"""
        try:
            latest_metrics = SystemMetrics.query.order_by(
                SystemMetrics.timestamp.desc()
            ).first()

            if not latest_metrics:
                return jsonify({
                    'error': 'No metrics available'
                }), 404

            return jsonify(latest_metrics.to_dict())
        except Exception as e:
            logger.error(f"Error fetching monitoring metrics: {str(e)}")
            return jsonify({
                'error': str(e)
            }), 500

    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500

    logger.info("Routes initialized successfully")
    return True