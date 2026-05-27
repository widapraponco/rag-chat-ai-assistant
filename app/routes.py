import logging
import asyncio
import json
import inspect
import threading
from flask import Blueprint, request, jsonify, current_app, render_template, Response, stream_with_context
from app.vanna_init import initialize_vanna
from app.catalog_loader import load_dbt_catalog
from vanna.core.user import RequestContext, User
from vanna.core.tool import Tool, ToolSchema, ToolContext

logger = logging.getLogger(__name__)

# Blueprints
health_bp = Blueprint('health', __name__)
api_bp = Blueprint('api', __name__)

# Global Vanna 2.0 agent instance
_agent = None
_agent_lock = threading.Lock()


def get_agent():
    """Get or initialize Vanna 2.0 Agent (singleton) — thread-safe"""
    global _agent

    if _agent is None:
        with _agent_lock:
            if _agent is None:  # Double-check pattern
                logger.info("Initializing Vanna 2.0 agent...")

                _agent = initialize_vanna(
                    mysql_host=current_app.config['MYSQL_HOST'],
                    mysql_user=current_app.config['MYSQL_USER'],
                    mysql_password=current_app.config['MYSQL_PASSWORD'],
                    mysql_database=current_app.config['MYSQL_DATABASE'],
                    mysql_port=current_app.config['MYSQL_PORT'],
                    hf_token=current_app.config['HF_TOKEN'],
                    chromadb_path=current_app.config['CHROMADB_PATH'],
                    embedding_model=current_app.config['VANNA_EMBEDDING_MODEL'],
                    deepseek_api_key=current_app.config.get('DEEPSEEK_API_KEY'),
                    deepseek_api_base=current_app.config.get('DEEPSEEK_API_BASE'),
                )

                # Load DBT catalog ke agent memory
                try:
                    logger.info("Loading dbt catalog into agent memory...")
                    catalog = load_dbt_catalog(current_app.config['DBT_CATALOG_PATH'])
                    tables = catalog.extract_table_descriptions()
                    ddl_map = catalog.get_table_schema_ddl()

                    schema_entries = []
                    for table_name, schema_name, description, columns in tables:
                        key = f"{schema_name}.{table_name}"
                        ddl = ddl_map.get(key, "")
                        if not ddl:
                            continue
                        schema_entries.append({
                            "table": table_name,
                            "schema": schema_name,
                            "ddl": ddl,
                            "description": description,
                            "full_key": key
                        })

                    if schema_entries:
                        _save_schema_to_memory(_agent, schema_entries)
                        logger.info(f"✅ Loaded {len(schema_entries)} tables into agent memory")

                except Exception as e:
                    logger.error(f"❌ Failed to load dbt catalog: {e}", exc_info=True)

    return _agent


def _save_schema_to_memory(agent, schema_entries):
    """Save schema entries to agent memory with proper event loop handling."""
    memory = agent.agent_memory
    admin_user = User(
        id="admin",
        email="admin@system.local",
        group_memberships=["admin", "user"]
    )

    # Check if save_text_memory is async
    is_async = asyncio.iscoroutinefunction(memory.save_text_memory)

    # Prepare all entries
    documents = []
    metadatas = []
    ids = []

    for i, entry in enumerate(schema_entries):
        documents.append(f"CREATE TABLE {entry['full_key']} AS:\n{entry['ddl']}")
        metadatas.append({
            "type": "ddl",
            "table": entry['table'],
            "schema": entry['schema'],
            "source": "dbt_catalog"
        })
        ids.append(f"ddl_{entry['schema']}_{entry['table']}_{i}")

    # Schema overview
    overview = f"Database: {current_app.config['MYSQL_DATABASE']}\n\n"
    overview += "\n".join([f"Table: {e['full_key']}\n{e['ddl'][:200]}..." for e in schema_entries])

    documents.append(overview)
    metadatas.append({"type": "schema_overview", "source": "dbt_catalog"})
    ids.append("schema_overview")

    # System prompt
    system_prompt = f"""You are an expert SQL assistant for the '{current_app.config['MYSQL_DATABASE']}' MySQL database.

Your ONLY job is to:
1. Understand the user's question about data
2. Generate a correct SQL query based on the database schema below
3. ALWAYS execute the SQL using the RunSqlTool — this is MANDATORY
4. Return the query results to the user

You MUST NOT answer from general knowledge or make up data.

Database Schema:
{overview}

Rules:
- ALWAYS use RunSqlTool to execute SQL queries
- Use LIMIT 100 for SELECT queries unless user specifies otherwise
- For COUNT/SUM/AVG queries, include meaningful column aliases"""

    documents.append(system_prompt)
    metadatas.append({"type": "system_prompt", "source": "dbt_catalog"})
    ids.append("system_prompt")

    # Try direct ChromaDB access first (most reliable)
    try:
        collection = None
        if hasattr(memory, '_collection'):
            collection = memory._collection
        elif hasattr(memory, 'collection'):
            collection = memory.collection
        elif hasattr(memory, '_client'):
            collection = memory._client.get_collection("vanna_tool_memory")

        if collection:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info("✅ Saved schema via direct ChromaDB access")
            return
    except Exception as e:
        logger.warning(f"Direct ChromaDB access failed: {e}")

    # Fallback: Try save_text_memory
    try:
        if is_async:
            # Run async function in a new loop
            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    for doc, meta, doc_id in zip(documents, metadatas, ids):
                        ctx = ToolContext(
                            user=admin_user,
                            conversation_id="dbt_training",
                            request_id=doc_id,
                            agent_memory=memory
                        )
                        loop.run_until_complete(
                            memory.save_text_memory(content=doc, context=ctx)
                        )
                finally:
                    loop.close()

            thread = threading.Thread(target=run_async)
            thread.start()
            thread.join(timeout=60)  # Wait max 60 seconds
            if thread.is_alive():
                logger.warning("save_text_memory thread timed out")
            else:
                logger.info("✅ Saved schema via save_text_memory (async)")
        else:
            # Sync call
            for doc, meta, doc_id in zip(documents, metadatas, ids):
                ctx = ToolContext(
                    user=admin_user,
                    conversation_id="dbt_training",
                    request_id=doc_id,
                    agent_memory=memory
                )
                memory.save_text_memory(content=doc, context=ctx)
            logger.info("✅ Saved schema via save_text_memory (sync)")
    except Exception as e:
        logger.error(f"save_text_memory failed: {e}")

        # Last resort: Save to file
        try:
            with open("schema_cache.json", "w") as f:
                json.dump({
                    "database": current_app.config['MYSQL_DATABASE'],
                    "tables": schema_entries
                }, f, indent=2)
            logger.info("✅ Saved schema to schema_cache.json")
        except Exception as e2:
            logger.error(f"File save also failed: {e2}")


def _build_request_context() -> RequestContext:
    """Build RequestContext with database metadata."""
    return RequestContext(
        headers=dict(request.headers),
        cookies=dict(request.cookies),
        metadata={
            "database": current_app.config.get('MYSQL_DATABASE', ''),
            "database_type": "mysql",
            "instruction": "Always query the database using RunSqlTool. Never answer from general knowledge."
        }
    )


# ═══════════════════════════════════════════════════════════════
# FIX: SSE Streaming dengan ThreadPoolExecutor (anti-timeout)
# ═══════════════════════════════════════════════════════════════

from concurrent.futures import ThreadPoolExecutor
import queue

_executor = ThreadPoolExecutor(max_workers=4)


def _stream_agent_response(agent, request_context, question, conversation_id):
    """
    Run agent.send_message() in a separate thread with its own event loop.
    This avoids timeout issues with Flask's request context.
    """
    result_queue = queue.Queue()

    def _run_in_thread():
        """Run async generator in a dedicated thread with fresh event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _collect():
            try:
                async for component in agent.send_message(
                    request_context=request_context,
                    message=question,
                    conversation_id=conversation_id,
                ):
                    result_queue.put(component.model_dump())
                result_queue.put(None)  # Sentinel: done
            except Exception as e:
                logger.error(f"Agent stream error: {e}", exc_info=True)
                result_queue.put({"type": "error", "content": str(e)})
                result_queue.put(None)

        try:
            loop.run_until_complete(_collect())
        finally:
            loop.close()

    # Start thread
    thread = threading.Thread(target=_run_in_thread)
    thread.start()

    # Yield results as they come
    while True:
        try:
            item = result_queue.get(timeout=120)  # 2 minute timeout per item
            if item is None:
                break
            yield item
        except queue.Empty:
            logger.error("Agent stream timeout — no response for 120 seconds")
            yield {"type": "error", "content": "Agent response timeout"}
            break

    thread.join(timeout=5)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@health_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': current_app.config['API_TITLE'],
        'version': current_app.config['API_VERSION']
    }), 200


@health_bp.route('/', methods=['GET'])
def index():
    return jsonify({
        'title': current_app.config['API_TITLE'],
        'version': current_app.config['API_VERSION'],
        'endpoints': {
            'health': '/health',
            'chat_ui': '/chat',
            'chat': '/api/chat',
            'chat_stream': '/api/chat/stream',
            'debug_memory': '/api/debug/memory'
        }
    }), 200


@health_bp.route('/chat', methods=['GET'])
def chat_ui():
    return render_template('chat.html')


@api_bp.route('/debug/memory', methods=['GET'])
def debug_memory():
    """Debug: check what's stored in agent memory."""
    try:
        agent = get_agent()
        memory = agent.agent_memory

        results = {"success": True, "strategies": {}}

        # Strategy 1: Direct ChromaDB
        try:
            collection = None
            if hasattr(memory, '_collection'):
                collection = memory._collection
            elif hasattr(memory, 'collection'):
                collection = memory.collection

            if collection:
                count = collection.count()
                query_results = collection.query(
                    query_texts=["employees table schema"],
                    n_results=5,
                    where={"type": "ddl"}
                )
                results["strategies"]["direct_chromadb"] = {
                    "total_documents": count,
                    "ddl_found": len(query_results.get('ids', [[]])[0]),
                    "ids": query_results.get('ids', [[]])[0][:5]
                }
            else:
                results["strategies"]["direct_chromadb"] = {"error": "No collection access"}
        except Exception as e:
            results["strategies"]["direct_chromadb"] = {"error": str(e)}

        # Strategy 2: File cache
        try:
            import os
            if os.path.exists("schema_cache.json"):
                with open("schema_cache.json", "r") as f:
                    cache = json.load(f)
                results["strategies"]["file_cache"] = {
                    "exists": True,
                    "table_count": len(cache.get("tables", []))
                }
            else:
                results["strategies"]["file_cache"] = {"exists": False}
        except Exception as e:
            results["strategies"]["file_cache"] = {"error": str(e)}

        return jsonify(results)

    except Exception as e:
        logger.error(f"Debug memory error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route('/chat', methods=['POST'])
def chat():
    """Non-streaming chat endpoint."""
    try:
        data = request.get_json(silent=True) or {}
        question = data.get('question', '').strip()
        conversation_id = data.get('conversation_id')

        if not question:
            return jsonify({'success': False, 'error': 'Question is required'}), 400

        request_context = _build_request_context()
        agent = get_agent()

        # Use thread-based streaming even for non-streaming (to avoid timeout)
        components = list(_stream_agent_response(agent, request_context, question, conversation_id))

        return jsonify({
            'success': True,
            'question': question,
            'components': components,
            'conversation_id': conversation_id,
            'error': None
        }), 200

    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint via SSE — with thread-based async handling."""
    data = request.get_json(silent=True) or {}
    question = data.get('question', '').strip()
    conversation_id = data.get('conversation_id')

    if not question:
        return jsonify({'success': False, 'error': 'Question is required'}), 400

    request_context = _build_request_context()
    agent = get_agent()

    def generate():
        for component in _stream_agent_response(agent, request_context, question, conversation_id):
            yield f"data: {json.dumps(component)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@api_bp.route('/tables', methods=['GET'])
def get_tables():
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
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/columns/<table_name>', methods=['GET'])
def get_columns(table_name):
    try:
        catalog = load_dbt_catalog(current_app.config['DBT_CATALOG_PATH'])
        column_metadata = catalog.get_column_metadata()

        table_columns = [
            col for col in column_metadata
            if table_name.lower() in col['table'].lower()
        ]

        if not table_columns:
            return jsonify({'success': False, 'error': f'Table {table_name} not found'}), 404

        return jsonify({
            'success': True,
            'table': table_name,
            'columns': table_columns,
            'count': len(table_columns)
        }), 200

    except Exception as e:
        logger.error(f"Error in get_columns: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/debug/search', methods=['POST'])
def debug_search():
    """Test semantic search in agent memory"""
    try:
        data = request.get_json() or {}
        query = data.get('query', 'employees')
        
        agent = get_agent()
        memory = agent.agent_memory
        collection = memory._collection if hasattr(memory, '_collection') else None
        
        if collection:
            results = collection.query(
                query_texts=[query],
                n_results=3,
            )
            return jsonify({
                "query": query,
                "found": len(results.get('ids', [[]])[0]),
                "documents": results.get('documents', [[]])[0][:3],
                "distances": results.get('distances', [[]])[0][:3]
            })
        return jsonify({"error": "No collection"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────

@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@api_bp.errorhandler(500)
def server_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500