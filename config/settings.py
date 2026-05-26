import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    
    # Flask settings
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = True
    
    # MySQL settings
    MYSQL_HOST = os.getenv('MYSQL_HOST', '45.77.35.173')
    MYSQL_USER = os.getenv('MYSQL_USER', 'remote-soravisi')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'sv_analytics')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
    
    # Vanna settings
    VANNA_EMBEDDING_MODEL = 'BAAI/bge-m3'  # BGE-M3 model
    CHROMADB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chromadb')
    
    # DeepSeek LLM settings
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
    DEEPSEEK_MODEL = 'deepseek-chat'  # DeepSeek chat model
    DEEPSEEK_API_BASE = 'https://api.deepseek.com/v1'
    
    # DBT catalog path
    DBT_CATALOG_PATH = os.getenv(
        'DBT_CATALOG_PATH', 
        '../dbt_analytics/target/catalog.json'
    )
    
    # API settings
    API_TITLE = 'Soravisi SQL Chat API'
    API_VERSION = '1.0.0'
    MAX_QUERY_RESULTS = 1000

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    ENV = 'development'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    ENV = 'production'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    ENV = 'testing'
    CHROMADB_PATH = os.path.join(os.path.dirname(__file__), '..', 'tests', 'chromadb')

# Config factory
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get config based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
