"""
API routes for Soravisi SQL Chat
"""
import logging
from flask import Blueprint, request, jsonify, current_app, render_template
from app.vanna_init import initialize_vanna
from app.catalog_loader import load_dbt_catalog

logger = logging.getLogger(__name__)

# Blueprints
health_bp = Blueprint('health', __name__)
api_bp = Blueprint('api', __name__)

# Global Vanna instance
vanna_instance = None


def get_vanna_instance():
    """Get or initialize Vanna instance"""
    global vanna_instance
    
    if vanna_instance is None:
        logger.info("Initializing Vanna instance...")
        vanna_instance = initialize_vanna(
            mysql_host=current_app.config['MYSQL_HOST'],
            mysql_user=current_app.config['MYSQL_USER'],
            mysql_password=current_app.config['MYSQL_PASSWORD'],
            mysql_database=current_app.config['MYSQL_DATABASE'],
            mysql_port=current_app.config['MYSQL_PORT'],
            hf_token=current_app.config['HF_TOKEN'],
            chromadb_path=current_app.config['CHROMADB_PATH'],
            embedding_model=current_app.config['VANNA_EMBEDDING_MODEL'],
            deepseek_api_key=current_app.config.get('DEEPSEEK_API_KEY'),
            deepseek_api_base=current_app.config.get('DEEPSEEK_API_BASE')
        )
        
        # Load DBT catalog into Vanna
        try:
            logger.info("Loading dbt catalog...")
            catalog = load_dbt_catalog(current_app.config['DBT_CATALOG_PATH'])
            
            # Add table and column information
            tables = catalog.extract_table_descriptions()
            for table_name, schema_name, description, columns in tables:
                ddl = catalog.get_table_schema_ddl().get(f"{schema_name}.{table_name}", "")
                if ddl:
                    vanna_instance.add_table_context(description, ddl, columns)
            
            logger.info(f"Loaded {len(tables)} tables into Vanna")
        except Exception as e:
            logger.error(f"Failed to load dbt catalog: {e}")
    
    return vanna_instance


# Health Check Routes
@health_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': current_app.config['API_TITLE'],
        'version': current_app.config['API_VERSION']
    }), 200


@health_bp.route('/', methods=['GET'])
def index():
    """Index page with API info"""
    return jsonify({
        'title': current_app.config['API_TITLE'],
        'version': current_app.config['API_VERSION'],
        'endpoints': {
            'health': '/health',
            'chat_ui': '/chat',
            'generate_sql': '/api/generate_sql',
            'execute_sql': '/api/execute_sql',
            'chat': '/api/chat',
            'tables': '/api/tables',
            'columns': '/api/columns/<table_name>'
        }
    }), 200


@health_bp.route('/chat', methods=['GET'])
def chat_ui():
    """Serve the chat interface"""
    return render_template('chat.html')


# API Routes
@api_bp.route('/generate_sql', methods=['POST'])
def generate_sql():
    """
    Generate SQL from natural language question using DeepSeek LLM
    
    Request JSON:
        {
            "question": "How many employees are in the system?"
        }
    
    Response:
        {
            "success": true,
            "question": "...",
            "sql": "SELECT COUNT(*) FROM users;",
            "error": null
        }
    """
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400
        
        vanna = get_vanna_instance()
        sql = vanna.generate_sql(question)
        
        if sql:
            return jsonify({
                'success': True,
                'question': question,
                'sql': sql,
                'error': None
            }), 200
        else:
            return jsonify({
                'success': False,
                'question': question,
                'sql': None,
                'error': 'Failed to generate SQL'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in generate_sql: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/execute_sql', methods=['POST'])
def execute_sql():
    """
    Execute SQL query against MySQL database
    
    Request JSON:
        {
            "sql": "SELECT * FROM users LIMIT 10;"
        }
    
    Response:
        {
            "success": true,
            "sql": "...",
            "rows": [...],
            "row_count": 10,
            "error": null
        }
    """
    try:
        data = request.get_json()
        sql = data.get('sql', '').strip()
        
        if not sql:
            return jsonify({
                'success': False,
                'error': 'SQL query is required'
            }), 400
        
        vanna = get_vanna_instance()
        results = vanna.execute_sql(sql)
        
        if results is not None:
            return jsonify({
                'success': True,
                'sql': sql,
                'rows': results,
                'row_count': len(results),
                'error': None
            }), 200
        else:
            return jsonify({
                'success': False,
                'sql': sql,
                'rows': None,
                'error': 'Failed to execute SQL'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in execute_sql: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint - generates and optionally executes SQL
    
    Request JSON:
        {
            "question": "How many employees joined in 2024?",
            "execute": true
        }
    
    Response:
        {
            "success": true,
            "question": "...",
            "sql": "...",
            "results": [...],
            "error": null
        }
    """
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        execute = data.get('execute', False)
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400
        
        vanna = get_vanna_instance()
        
        # Generate SQL
        sql = vanna.generate_sql(question)
        if not sql:
            return jsonify({
                'success': False,
                'question': question,
                'sql': None,
                'results': None,
                'error': 'Failed to generate SQL'
            }), 500
        
        # Execute if requested
        results = None
        if execute:
            results = vanna.execute_sql(sql)
        
        return jsonify({
            'success': True,
            'question': question,
            'sql': sql,
            'results': results,
            'executed': execute and results is not None,
            'error': None
        }), 200
            
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/tables', methods=['GET'])
def get_tables():
    """Get list of available tables from dbt catalog"""
    try:
        catalog = load_dbt_catalog(current_app.config['DBT_CATALOG_PATH'])
        tables = catalog.extract_table_descriptions()
        
        table_list = [
            {
                'table': f"{schema}.{table}" if schema else table,
                'schema': schema,
                'name': table,
                'description': desc,
                'columns': len(cols)
            }
            for table, schema, desc, cols in tables
        ]
        
        return jsonify({
            'success': True,
            'tables': table_list,
            'count': len(table_list)
        }), 200
            
    except Exception as e:
        logger.error(f"Error in get_tables: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/columns/<table_name>', methods=['GET'])
def get_columns(table_name):
    """Get column information for a specific table"""
    try:
        catalog = load_dbt_catalog(current_app.config['DBT_CATALOG_PATH'])
        column_metadata = catalog.get_column_metadata()
        
        # Filter for matching table
        table_columns = [
            col for col in column_metadata
            if table_name.lower() in col['table'].lower()
        ]
        
        if not table_columns:
            return jsonify({
                'success': False,
                'error': f'Table {table_name} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'table': table_name,
            'columns': table_columns,
            'count': len(table_columns)
        }), 200
            
    except Exception as e:
        logger.error(f"Error in get_columns: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers
@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@api_bp.errorhandler(500)
def server_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500
