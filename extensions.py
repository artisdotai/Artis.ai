"""Database extension initialization"""
import logging
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import time

logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy with the custom base class
db = SQLAlchemy(model_class=Base)

def init_database(app):
    """Initialize database with proper error handling and connection retry"""
    max_retries = 3
    retry_delay = 2
    current_retry = 0

    while current_retry < max_retries:
        try:
            logger.info(f"Initializing database (attempt {current_retry + 1}/{max_retries})")

            # Initialize without overriding existing instance
            if not hasattr(app, 'extensions') or 'sqlalchemy' not in app.extensions:
                db.init_app(app)

            with app.app_context():
                # Test database connection with proper text() wrapper
                db.session.execute(text('SELECT 1'))
                db.session.commit()

                # Import models here to avoid circular imports
                import models  # noqa: F401
                logger.info("Creating database tables")
                db.create_all()
                logger.info("Database tables created successfully")

            return True

        except SQLAlchemyError as e:
            logger.error(f"Database initialization error: {str(e)}")
            current_retry += 1
            if current_retry < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Maximum retries reached. Database initialization failed.")
                return False

        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {str(e)}", exc_info=True)
            return False

    return False