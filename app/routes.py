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

import decimal

def json_default(obj):
    """Handle tipe data yang tidak bisa di-serialize JSON secara default."""
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode('utf-8', errors='ignore')
    if hasattr(obj, 'isoformat'):  # datetime, date, time
        return obj.isoformat()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')  

def get_agent():
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
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

                try:
                    catalog = load_dbt_catalog(current_app.config['DBT_CATALOG_PATH'])
                    tables = catalog.extract_table_descriptions()
                    ddl_map = catalog.get_table_schema_ddl()

                    schema_parts = []
                    for table_name, schema_name, description, columns in tables:
                        key = f"{schema_name}.{table_name}"
                        ddl = ddl_map.get(key, "")
                        if ddl:
                            schema_parts.append(f"-- Table: {key}\n{ddl}")

                    # Inject ke LLM system prompt — INI yang menentukan konteks
                    if schema_parts:
                        _agent.llm_service.schema_context = "\n\n".join(schema_parts)
                        logger.info(f"✅ Injected {len(schema_parts)} tables into LLM schema_context")
                    else:
                        logger.warning("⚠ No DDL found to inject into schema_context")

                except Exception as e:
                    logger.error(f"Failed to inject schema_context: {e}", exc_info=True)

    return _agent

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
    result_queue = queue.Queue()

    def _run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _collect():
            gen = agent.send_message(
                request_context=request_context,
                message=question,
                conversation_id=conversation_id,
            )
            try:
                async for component in gen:
                    dump = component.model_dump()
                    rc = dump.get('rich_component') or {}
                    sc = dump.get('simple_component') or {}
                    logger.info(f"🔍 type={rc.get('type')} | rc.data={json.dumps(rc.get('data'), default=str)[:150]} | sc={json.dumps(sc, default=str)[:100]}")
                    result_queue.put(dump)
                result_queue.put(None)
            except Exception as e:
                logger.error(f"Agent stream error: {e}", exc_info=True)
                result_queue.put({"type": "error", "content": str(e)})
                result_queue.put(None)
            finally:
                # Pastikan generator selalu ditutup — ini yang mencegah error
                await gen.aclose()

        try:
            loop.run_until_complete(_collect())
        except Exception as e:
            logger.error(f"Event loop error: {e}", exc_info=True)
            result_queue.put({"type": "error", "content": str(e)})
            result_queue.put(None)
        finally:
            # Tutup semua pending task sebelum loop ditutup
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()

    while True:
        try:
            item = result_queue.get(timeout=120)
            if item is None:
                break
            yield item
        except queue.Empty:
            logger.error("Agent stream timeout")
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
                    query_texts=["CREATE TABLE"],
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
            'components': json.loads(json.dumps(components, default=json_default)),
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
            yield f"data: {json.dumps(component, default=json_default)}\n\n"
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
    """
    Test semantic search + verifikasi schema_context di LLM.
    
    Request: { "query": "employees", "n_results": 3, "type_filter": "ddl" }
    """
    try:
        data = request.get_json() or {}
        query = data.get('query', 'employees')
        n_results = data.get('n_results', 3)
        type_filter = data.get('type_filter')  # 'ddl', 'system_prompt', dll — opsional

        agent = get_agent()
        memory = agent.agent_memory

        # ── 1. Cek ChromaDB ──────────────────────────────────────────────────
        chroma_result = {"status": "not_accessible"}
        collection = None
        if hasattr(memory, '_collection'):
            collection = memory._collection
        elif hasattr(memory, 'collection'):
            collection = memory.collection

        if collection:
            query_kwargs = {
                "query_texts": [query],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }
            if type_filter:
                query_kwargs["where"] = {"type": type_filter}

            results = collection.query(**query_kwargs)
            docs      = results.get('documents', [[]])[0]
            distances = results.get('distances', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]

            chroma_result = {
                "status": "ok",
                "total_documents_in_collection": collection.count(),
                "results_found": len(docs),
                "results": [
                    {
                        "rank": i + 1,
                        "distance": round(d, 4),
                        # distance < 0.5 = sangat relevan, > 1.0 = tidak relevan
                        "relevance": "high" if d < 0.5 else "medium" if d < 1.0 else "low",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        # Potong dokumen panjang agar response tidak terlalu besar
                        "document_preview": docs[i][:300] + "..." if len(docs[i]) > 300 else docs[i]
                    }
                    for i, d in enumerate(distances)
                ]
            }

        # ── 2. Cek schema_context di LLM ────────────────────────────────────
        # Ini yang paling penting — kalau kosong, agent pasti jawab umum
        llm_context_result = {"status": "not_accessible"}
        if hasattr(agent, 'llm_service'):
            llm = agent.llm_service
            schema_ctx = getattr(llm, 'schema_context', None)
            llm_context_result = {
                "status": "ok",
                "schema_context_present": bool(schema_ctx),
                "schema_context_length": len(schema_ctx) if schema_ctx else 0,
                # Preview 500 char pertama dari system prompt
                "schema_context_preview": schema_ctx[:500] + "..." if schema_ctx and len(schema_ctx) > 500 else schema_ctx,
                # Cek apakah query term ada di schema_context
                "query_found_in_schema": query.lower() in schema_ctx.lower() if schema_ctx else False
            }

        # ── 3. Diagnosis ─────────────────────────────────────────────────────
        diagnosis = []
        if chroma_result.get("total_documents_in_collection", 0) == 0:
            diagnosis.append("❌ ChromaDB kosong — jalankan seeder terlebih dahulu")
        elif not any(r["relevance"] == "high" for r in chroma_result.get("results", [])):
            diagnosis.append("⚠ Tidak ada hasil dengan relevance tinggi — coba query yang lebih spesifik")
        else:
            diagnosis.append("✅ ChromaDB berisi data relevan")

        if not llm_context_result.get("schema_context_present"):
            diagnosis.append("❌ schema_context LLM kosong — agent akan jawab dari pengetahuan umum")
        elif not llm_context_result.get("query_found_in_schema"):
            diagnosis.append(f"⚠ '{query}' tidak ditemukan di schema_context — cek apakah DDL sudah ter-load")
        else:
            diagnosis.append(f"✅ '{query}' ditemukan di schema_context LLM")

        return jsonify({
            "query": query,
            "chromadb": chroma_result,
            "llm_schema_context": llm_context_result,
            "diagnosis": diagnosis
        }), 200

    except Exception as e:
        logger.error(f"debug_search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
@api_bp.route('/debug/reseed', methods=['POST'])
def debug_reseed():
    """Hapus semua data ChromaDB dan seed ulang dengan embedding yang benar."""
    try:
        agent = get_agent()
        memory = agent.agent_memory

        # 1. Hapus semua dokumen lama
        collection = None
        if hasattr(memory, '_collection'):
            collection = memory._collection
        elif hasattr(memory, 'collection'):
            collection = memory.collection

        if collection and collection.count() > 0:
            all_ids = collection.get()['ids']
            collection.delete(ids=all_ids)
            logger.info(f"🗑 Deleted {len(all_ids)} documents from ChromaDB")

        # 2. Seed ulang via save_text_memory — pakai embedding yang sama dengan query
        catalog = load_dbt_catalog(current_app.config['DBT_CATALOG_PATH'])
        tables = catalog.extract_table_descriptions()
        ddl_map = catalog.get_table_schema_ddl()

        schema_text_parts = []
        seeded = 0

        admin_user = User(
            id="admin",
            email="admin@system.local",
            group_memberships=["admin", "user"]
        )

        def save_doc(text, doc_id):
            """Simpan via save_text_memory agar embedding konsisten."""
            ctx = ToolContext(
                user=admin_user,
                conversation_id="reseed",
                request_id=doc_id,
                agent_memory=memory
            )
            is_async = asyncio.iscoroutinefunction(memory.save_text_memory)
            if is_async:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(memory.save_text_memory(content=text, context=ctx))
                finally:
                    loop.close()
            else:
                memory.save_text_memory(content=text, context=ctx)

        for table_name, schema_name, description, columns in tables:
            key = f"{schema_name}.{table_name}"
            ddl = ddl_map.get(key, "")
            if not ddl:
                continue

            text = f"Table: {key}\nDescription: {description}\nDDL:\n{ddl}"
            save_doc(text, f"ddl_{schema_name}_{table_name}")
            schema_text_parts.append(f"-- Table: {key}\n{ddl}")
            seeded += 1

        # 3. Inject ke schema_context LLM sekaligus
        agent.llm_service.schema_context = "\n\n".join(schema_text_parts)
        logger.info(f"✅ Reseeded {seeded} tables, schema_context injected")

        return jsonify({
            "success": True,
            "tables_seeded": seeded,
            "chromadb_count": collection.count() if collection else 0,
            "schema_context_length": len(agent.llm_service.schema_context)
        })

    except Exception as e:
        logger.error(f"Reseed error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/debug/embedding_check', methods=['POST'])
def debug_embedding_check():
    """
    Diagnosa embedding mismatch antara insert dan query.
    Cek apakah ChromaAgentMemory menggunakan embedding function yang sama
    dengan yang dipakai saat dokumen disimpan.
    """
    try:
        agent = get_agent()
        memory = agent.agent_memory

        collection = None
        if hasattr(memory, '_collection'):
            collection = memory._collection
        elif hasattr(memory, 'collection'):
            collection = memory.collection

        if not collection:
            return jsonify({"error": "Cannot access collection"}), 500

        result = {}

        # 1. Cek embedding function yang terdaftar di collection
        result["collection_name"] = collection.name
        result["total_docs"] = collection.count()

        # ChromaDB menyimpan metadata embedding function di collection
        col_meta = collection.metadata or {}
        result["collection_metadata"] = col_meta

        # 2. Peek dokumen — ambil tanpa vector search
        peek = collection.peek(limit=3)
        result["peek"] = {
            "ids": peek.get("ids", []),
            "documents": [d[:150] for d in peek.get("documents", [])],
            "embeddings_present": peek.get("embeddings") is not None
        }

        # 3. Cek embedding function di ChromaAgentMemory
        ef_info = {}
        if hasattr(memory, '_embedding_function'):
            ef_info["source"] = "_embedding_function attr"
            ef_info["type"] = str(type(memory._embedding_function))
        elif hasattr(memory, 'embedding_function'):
            ef_info["source"] = "embedding_function attr"
            ef_info["type"] = str(type(memory.embedding_function))
        else:
            ef_info["source"] = "not found on memory object"
            ef_info["available_attrs"] = [
                a for a in dir(memory) if 'embed' in a.lower()
            ]
        result["memory_embedding_function"] = ef_info

        # 4. Test: query dengan embedding kosong (get by ID) vs query vector
        # Kalau get() berhasil tapi query() return 0, ini pasti mismatch
        if peek.get("ids"):
            test_id = peek["ids"][0]
            get_result = collection.get(ids=[test_id], include=["documents"])
            result["get_by_id_works"] = len(get_result.get("documents", [])) > 0

            # Query dengan teks yang persis sama dengan dokumen
            first_doc_words = peek["documents"][0][:50] if peek["documents"] else "table"
            query_result = collection.query(
                query_texts=[first_doc_words],
                n_results=1,
                include=["documents", "distances"]
            )
            found = len(query_result.get("ids", [[]])[0])
            distance = query_result.get("distances", [[]])[0]
            result["query_with_doc_content"] = {
                "query": first_doc_words,
                "found": found,
                "distance": distance[0] if distance else None
            }

            # Diagnosis
            if result["get_by_id_works"] and found == 0:
                result["diagnosis"] = "❌ EMBEDDING MISMATCH: dokumen ada tapi tidak bisa di-query. Insert dan query pakai embedding berbeda."
            elif found > 0 and distance and distance[0] < 0.5:
                result["diagnosis"] = "✅ Embedding konsisten — masalah bukan di sini"
            else:
                result["diagnosis"] = "⚠ Dokumen ditemukan tapi distance tinggi — embedding kurang akurat"

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"embedding_check error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@api_bp.route('/debug/system_prompt', methods=['GET'])
def debug_system_prompt():
    """Cek isi system prompt yang dikirim ke LLM setiap request."""
    try:
        agent = get_agent()
        llm = agent.llm_service

        # Cek apakah get_system_prompt ada dan apa isinya
        has_method = hasattr(llm, 'get_system_prompt')
        prompt_content = ""
        if has_method:
            try:
                prompt_content = llm.get_system_prompt()
            except Exception as e:
                prompt_content = f"Error calling get_system_prompt: {e}"

        # Cek apakah Vanna Agent benar-benar memanggil get_system_prompt
        # dengan inspect method build_prompt atau prepare_messages
        agent_methods = [m for m in dir(agent) if 'prompt' in m.lower() or 'message' in m.lower()]

        return jsonify({
            "llm_type": type(llm).__name__,
            "has_get_system_prompt": has_method,
            "schema_context_length": len(getattr(llm, 'schema_context', '')),
            "system_prompt_preview": prompt_content[:500] if prompt_content else "",
            "system_prompt_length": len(prompt_content),
            "agent_prompt_methods": agent_methods
        })
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