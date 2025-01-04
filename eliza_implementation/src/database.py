"""Database initialization for Eliza Framework"""
import os
import logging
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Initialize logging
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

# Initialize database with proper model class
db = SQLAlchemy(model_class=Base)
migrate = None

def init_db(app):
    """Initialize database with app context"""
    global migrate

    try:
        # Configure database
        app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_pre_ping': True,
            'pool_recycle': 300,
        }

        # Initialize SQLAlchemy with app
        db.init_app(app)

        # Initialize Flask-Migrate
        migrate = Migrate(app, db)

        # Create tables within app context
        with app.app_context():
            logger.info("Creating database tables...")
            db.create_all()
            logger.info("Database tables created successfully")

    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise