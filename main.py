"""Main application entry point with enhanced error handling"""
import os
import logging
import psutil
import signal
import time
from datetime import datetime
from app import app

# Configure logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_old_processes():
    """Cleanup any existing Flask processes with enhanced error handling"""
    try:
        current_pid = os.getpid()
        cleaned = False

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if it's a Python/Flask process
                if proc.info['name'] == 'python' and proc.pid != current_pid:
                    cmdline = proc.info.get('cmdline', [])
                    cmdline_str = ' '.join(cmdline) if cmdline else ''

                    if 'flask' in cmdline_str.lower() or 'main.py' in cmdline_str:
                        logger.info(f"Terminating old Flask process: {proc.pid}")
                        proc.send_signal(signal.SIGTERM)
                        cleaned = True

                        # Wait for process to terminate
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            logger.warning(f"Process {proc.pid} did not terminate, forcing kill")
                            proc.kill()

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
                logger.warning(f"Error handling process: {str(e)}")
                continue

        if cleaned:
            # Give OS time to free up ports
            time.sleep(2)

        return cleaned

    except Exception as e:
        logger.error(f"Error cleaning up processes: {str(e)}", exc_info=True)
        return False

def run_flask_app():
    """Run the Flask application with proper error handling"""
    try:
        # First cleanup any existing Flask processes
        cleanup_old_processes()

        # Start Flask application
        port = int(os.environ.get('PORT', 5000))
        retries = 3

        while retries > 0:
            try:
                logger.info(f"Starting Flask server on port {port}")
                app.run(host='0.0.0.0', port=port, debug=True)
                break
            except OSError as e:
                if "Address already in use" in str(e):
                    logger.error(f"Port {port} is already in use. Attempting cleanup...")
                    if cleanup_old_processes():
                        logger.info("Cleanup successful, retrying...")
                        retries -= 1
                        time.sleep(2)  # Wait for port to be freed
                        continue
                    else:
                        port = port + 1  # Try next port if cleanup failed
                        logger.info(f"Cleanup failed, trying port {port}")
                        retries -= 1
                else:
                    logger.error(f"OSError in Flask application: {str(e)}", exc_info=True)
                    raise
            except Exception as e:
                logger.error(f"Error starting Flask: {str(e)}", exc_info=True)
                raise

        if retries == 0:
            raise RuntimeError("Failed to start Flask after multiple retries")

    except Exception as e:
        logger.error(f"Critical error in Flask application: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        run_flask_app()
    except Exception as e:
        logger.error(f"Critical error starting application: {str(e)}", exc_info=True)
        raise