"""
Flask application factory
"""
import logging
from flask import Flask
from flask_cors import CORS
from config.settings import get_config

logger = logging.getLogger(__name__)


def create_app(config=None):
    """Create and configure Flask application"""
    app = Flask(__name__, template_folder='templates')
    
    # Load configuration
    if config is None:
        config = get_config()
    app.config.from_object(config)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    from app.routes import api_bp, health_bp
    app.register_blueprint(health_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    logger.info(f"Flask app created in {app.config['ENV']} environment")
    
    return app
