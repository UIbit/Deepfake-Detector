import os
import logging
from app import app

# Configure logging for production
if not app.debug:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set up server timeout for long processing tasks
timeout_seconds = int(os.environ.get('GUNICORN_TIMEOUT', 120))

if __name__ == "__main__":
    # This block is only used for local development
    app.run(host="0.0.0.0", port=5000, debug=True)
    
# Gunicorn will import this file and use the 'app' variable directly
