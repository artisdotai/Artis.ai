"""
Eliza Framework implementation of the cryptocurrency intelligence platform
"""
import os
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure Flask application instance"""
    try:
        # Get the absolute path to the eliza_implementation directory
        base_dir = Path(__file__).parent.parent.absolute()

        app = Flask(__name__, 
            template_folder=str(base_dir / 'templates'),
            static_folder=str(base_dir / 'static')
        )

        # Enable CORS with proper configuration
        CORS(app, resources={
            r"/api/*": {
                "origins": "*",
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })

        # Load configuration
        app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev_key')
        app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

        # Initialize database
        from .database import init_db
        init_db(app)
        logger.info("Database initialized successfully")

        # Initialize and register all routes
        from .routes import init_routes
        init_routes(app)

        # Register root route
        @app.route('/')
        def index():
            return render_template('index.html')

        # Add error handlers
        @app.errorhandler(404)
        def not_found_error(error):
            return jsonify({'status': 'error', 'message': 'Resource not found'}), 404

        @app.errorhandler(500)
        def internal_error(error):
            return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

        logger.info("Routes initialized successfully")
        return app

    except Exception as e:
        logger.error(f"Application creation error: {str(e)}")
        raise

def main():
    """Main entry point"""
    try:
        app = create_app()
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logger.error(f"Application startup error: {str(e)}")
        raise

if __name__ == '__main__':
    main()