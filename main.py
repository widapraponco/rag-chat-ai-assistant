"""
Main entry point for Soravisi API
Run with: python main.py
"""
import os
import logging
from dotenv import load_dotenv
from app import create_app
app = create_app()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # Create Flask app

    
    # Get port from environment or use default
    port = int(os.getenv('FLASK_PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Soravisi API on http://localhost:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info("Visit http://localhost:{port}/ for API documentation")
    logger.info("Open http://localhost:{port}/chat for chat interface")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        use_reloader=debug
    )
