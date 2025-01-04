"""Database module for handling database connections and management"""
import logging
import time
from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from extensions import db

logger = logging.getLogger(__name__)

@event.listens_for(Engine, "connect")
def connect(dbapi_connection, connection_record):
    """Set up connection parameters for better reliability"""
    try:
        connection_record.info['pid'] = dbapi_connection.get_backend_pid()
        cursor = dbapi_connection.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
    except Exception as e:
        logger.error(f"Error setting connection parameters: {str(e)}")
        raise

@event.listens_for(Engine, "checkout")
def checkout(dbapi_connection, connection_record, connection_proxy):
    """Ensure connections are live and handle reconnection"""
    pid = connection_record.info.get('pid')
    if pid:
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        except Exception as e:
            logger.error(f"Connection {pid} is invalid: {str(e)}")
            connection_proxy._pool.dispose()
            raise Exception(f"Invalid connection {pid} - forcing reconnection")

def execute_with_retry(func, max_retries=3, delay=1):
    """Execute database operations with exponential backoff retry"""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                logger.error(f"Final attempt failed: {str(e)}")
                raise last_error
            logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(delay * (2 ** attempt))

def verify_database_connection():
    """Verify database connection with retries"""
    def check_connection():
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True

    try:
        return execute_with_retry(check_connection)
    except Exception as e:
        logger.error(f"Database verification failed: {str(e)}")
        return False

def get_connection_health():
    """Check database connection health"""
    try:
        return verify_database_connection()
    except Exception as e:
        logger.error(f"Database connection health check failed: {str(e)}")
        return False