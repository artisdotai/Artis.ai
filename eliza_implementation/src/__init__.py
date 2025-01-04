"""
This is the main package for the Eliza Framework implementation.
"""
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import declarative_base
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize database
Base = declarative_base()
db = SQLAlchemy(model_class=Base)

def create_app():
    """Create and configure Flask application instance"""
    app = Flask(__name__,
        template_folder='../templates',
        static_folder='../static'
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
    from .config import Config
    app.config['SECRET_KEY'] = Config.FLASK_SECRET_KEY
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }

    # Initialize database
    db.init_app(app)

    # Create database tables
    with app.app_context():
        db.create_all()
        logger.info("Database tables created successfully")

    # Register routes
    with app.app_context():
        from .routes import init_routes
        init_routes(app)

    return app