"""Flask application initialization with monitoring controls"""
from flask import Flask
import os
import logging
import sys
from datetime import datetime
from flask_migrate import Migrate

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    try:
        logger.info("Starting Flask application creation")
        app = Flask(__name__)

        # Load configuration
        config_name = os.environ.get('FLASK_ENV', 'development')
        logger.info(f"Loading configuration for environment: {config_name}")

        from config import config
        app.config.from_object(config[config_name])

        # Initialize database
        from extensions import init_database, db
        if not init_database(app):
            raise RuntimeError("Failed to initialize database")

        # Initialize Flask-Migrate
        logger.info("Initializing Flask-Migrate")
        migrate = Migrate(app, db)

        # Initialize routes
        try:
            logger.info("Initializing routes")
            from routes import setup_routes
            if setup_routes(app):
                logger.info("Routes setup completed successfully")
            else:
                raise RuntimeError("Failed to setup routes")
        except Exception as e:
            logger.error(f"Error setting up routes: {str(e)}")
            raise

        logger.info("Flask application created successfully")
        return app

    except Exception as e:
        logger.error(f"Critical error creating Flask application: {str(e)}", exc_info=True)
        raise

# Create the app instance
app = None
try:
    logger.info("Creating Flask application instance")
    app = create_app()
    logger.info("Flask application instance created successfully")
except Exception as e:
    logger.error(f"Failed to create Flask application instance: {str(e)}", exc_info=True)
    sys.exit(1)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)