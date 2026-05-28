"""
Vanna 2.0 initialization with BGE-M3 embeddings, ChromaDB, DeepSeek LLM, and MySQL
"""
import json
import logging
import asyncio
import requests
from pathlib import Path
from typing import AsyncGenerator, List, Dict, Any

import httpx

# Vanna 2.0 core
from vanna import Agent
from vanna.core.llm import LlmService, LlmRequest, LlmResponse
from vanna.core.llm.models import LlmStreamChunk, LlmResponse, ToolCall
from vanna.core.registry import ToolRegistry, ToolSchema
from vanna.core.user import UserResolver, User, RequestContext
from vanna.core.system_prompt import SystemPromptBuilder

# Vanna 2.0 tools
from vanna.tools import RunSqlTool
from vanna.tools.agent_memory import (
    SaveQuestionToolArgsTool,
    SearchSavedCorrectToolUsesTool,
    SaveTextMemoryTool,
)

# Vanna 2.0 integrations
from vanna.integrations.mysql import MySQLRunner
from vanna.integrations.chromadb import ChromaAgentMemory

logger = logging.getLogger(__name__)

from vanna.core.tool import Tool, ToolSchema

# Custom retrieval tool
class GetDdlFromMemoryTool(Tool):
    """Retrieve table DDL dari ChromaDB memory berdasarkan table name atau description."""
    
    @property
    def name(self) -> str:
        return "GetDdlFromMemory"
    
    @property
    def description(self) -> str:
        return "Search dan retrieve table DDL/schema dari memory berdasarkan keyword"
    
    def get_args_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Table name atau deskripsi yang dicari (contoh: 'dim_attendance_performance columns')"
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, query: str, **kwargs) -> str:
        """Execute tool"""
        try:
            # Akses memory dari agent
            memory = kwargs.get("agent_memory")
            if not memory or not hasattr(memory, '_collection'):
                return "Error: Memory not accessible"
            
            collection = memory._collection
            results = collection.query(
                query_texts=[query],
                n_results=5,
                where={"type": "ddl"}
            )
            
            docs = results.get('documents', [[]])[0]
            if not docs:
                return f"Tidak ada DDL ditemukan untuk query: {query}"
            
            return "\n\n".join(docs[:3])  # Return top 3 results
        except Exception as e:
            return f"Error retrieving DDL: {e}"

# Di vanna_init.py — tambahkan class ini

class SchemaAwareSystemPromptBuilder(SystemPromptBuilder):

    def __init__(self, llm_service, db_name: str):
        self.llm_service = llm_service
        self.db_name = db_name

    async def build_system_prompt(
        self,
        user: User,
        tool_schemas: List[Any],
    ) -> str:
        logger.info(f"🔍 Tools available to agent: {[tool.name for tool in tool_schemas]}")
        logger.info(f"🔍 schema_context length: {len(self.llm_service.schema_context)}")

        schema = self.llm_service.schema_context or "(schema not loaded)"

        prompt = f"""You are an expert SQL assistant for '{self.db_name}' MySQL database.

    CRITICAL RULES:
    - ONLY query THIS database: {self.db_name}
    - NEVER answer from general knowledge or mention external companies
    - ALWAYS use RunSqlTool to execute SQL — never fabricate results
    - Table names MUST use schema prefix, example: {self.db_name}.table_name
    - Use LIMIT 100 for SELECT unless user specifies otherwise

    ## Available Tools
    """
        for tool in tool_schemas:
            # ToolSchema adalah object — pakai attribute access, bukan dict
            prompt += f"\n### {tool.name}\n"
            prompt += f"{getattr(tool, 'description', '')}\n"

        prompt += f"\n## Database Schema\n{schema}\n"
        prompt += f"\n## Current User\nGroups: {', '.join(user.group_memberships)}\n"

        return prompt
# ─────────────────────────────────────────────
# 1. DeepSeek LlmService
#
# DeepSeek pakai OpenAI-compatible API, tapi mengembalikan nested dict
# di field 'usage' (prompt_tokens_details, completion_tokens_details)
# yang menyebabkan ValidationError di Vanna 2.0 LlmResponse.
# Ref: https://github.com/vanna-ai/vanna/issues/1037
# Workaround: strip nested fields sebelum konstruksi LlmResponse.
# ─────────────────────────────────────────────

class DeepSeekLlmService(LlmService):
    """LlmService implementation untuk DeepSeek via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        db_name,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        schema_context = ''
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.db_name = db_name
        self.schema_context = schema_context

    def get_system_prompt(self, context=None) -> str:
        logger.info(f"✅ get_system_prompt called — schema_context length: {len(self.schema_context)}")
        base = f"""You are an expert SQL assistant for the '{self.db_name}' MySQL database.

    CRITICAL RULES:
    - ONLY use dbt model tables — DO NOT query raw tables directly
    - Table names use schema prefix: {self.db_name}.table_name  
    - ALWAYS use RunSqlTool to execute SQL
    - NEVER answer from general knowledge

    ## Business Terms
    - "karyawan" / "pegawai" / "employee" → kolom: nama_pegawai
    - "jumlah karyawan" → COUNT(DISTINCT nama_pegawai)
    - "performa" → kolom: kategori_performa
    - "hadir" → kolom: total_hadir
    - "terlambat" → kolom: total_menit_terlambat
    - "cuti" → kolom: hari_cuti_disetujui"""

    ## Database Schema
    # {self.schema_context if self.schema_context else '(schema not loaded)'}"""

        if self.schema_context:
            return base + f"\n\nDatabase Schema:\n{self.schema_context}"
        else:
            return base + "\n\n⚠ Schema not loaded yet — ask user to check /api/debug/reseed"

    def _get_messages_with_system(self, request: LlmRequest) -> list:
        messages = []

        for msg in request.messages:
            if msg.role == "tool":
                messages.append({
                    "role": "tool",
                    "content": msg.content or "",
                    "tool_call_id": msg.tool_call_id or f"call_{hash(msg.content or '')}"
                })
            elif msg.role == "assistant" and msg.tool_calls:
                tc_list = []
                for tc in msg.tool_calls:
                    tc_dict = tc.model_dump() if hasattr(tc, 'model_dump') else vars(tc)
                    tc_list.append({
                        "id": tc_dict.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc_dict.get("name", ""),
                            "arguments": json.dumps(tc_dict.get("arguments", {}))
                        }
                    })
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": tc_list
                })
            else:
                messages.append({
                    "role": msg.role,
                    "content": msg.content or ""
                })

        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            system = request.system_prompt or self.get_system_prompt()
            messages.insert(0, {"role": "system", "content": system})

        return messages
    
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, request: LlmRequest, stream: bool = False) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
            "stream": stream,
        }

    @staticmethod
    def _clean_usage(usage: dict) -> dict:
        """
        Hapus nested fields yang menyebabkan pydantic ValidationError.
        Vanna 2.0 LlmResponse mengharapkan integer, bukan dict di token details.
        """
        cleaned = usage.copy()
        cleaned.pop("prompt_tokens_details", None)
        cleaned.pop("completion_tokens_details", None)
        return cleaned

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        messages = self._get_messages_with_system(request)
        
        if request.system_prompt:
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages.insert(0, {"role": "system", "content": request.system_prompt})

        tools_payload = None
        if request.tools:
            tools_payload = []
            for tool in request.tools:
                td = tool.model_dump() if hasattr(tool, 'model_dump') else dict(tool)
                tools_payload.append({
                    "type": "function",
                    "function": {
                        "name": td.get("name"),
                        "description": td.get("description", ""),
                        "parameters": td.get("parameters") or td.get("input_schema") or {
                            "type": "object", "properties": {}
                        }
                    }
                })

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if tools_payload:
            payload["tools"] = tools_payload
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            
            # Log response error sebelum raise
            if response.status_code >= 400:
                logger.error(f"DeepSeek API error {response.status_code}: {response.text}")
                logger.error(f"Payload sent: {json.dumps(payload, default=str)[:1000]}")
            
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        message = choice["message"]
        usage = self._clean_usage(data.get("usage", {}))

        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=args
                ))

        from vanna.core.llm.models import LlmResponse
        return LlmResponse(
            content=message.get("content"),
            tool_calls=tool_calls or None,
            model=self.model,
            usage=usage,
            finish_reason=choice.get("finish_reason")
        )

    async def stream_request(self, request: LlmRequest) -> AsyncGenerator[LlmStreamChunk, None]:
        messages = self._get_messages_with_system(request)

        # Cek apakah system_prompt sudah ada di request dari Vanna
        if request.system_prompt:
            logger.info(f"🔍 system_prompt from request: {request.system_prompt[:100]}")
            # Pastikan tidak duplikat dengan yang sudah kita inject
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
        
        # Konversi tools ke format OpenAI
        tools_payload = None
        if request.tools:
            tools_payload = []
            for tool in request.tools:
                try:
                    # ToolSchema — cek berbagai format
                    if hasattr(tool, 'model_dump'):
                        td = tool.model_dump()
                    elif isinstance(tool, dict):
                        td = tool
                    else:
                        td = dict(tool)

                    logger.info(f"🔍 tool keys: {list(td.keys())}")

                    # Format OpenAI function calling
                    # Vanna ToolSchema kemungkinan punya: name, description, parameters
                    tools_payload.append({
                        "type": "function",
                        "function": {
                            "name": td.get("name"),
                            "description": td.get("description", ""),
                            "parameters": td.get("parameters") or td.get("input_schema") or {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    })
                except Exception as e:
                    logger.error(f"Failed to convert tool: {e} | tool={tool}")

            logger.info(f"🔍 tools_payload: {[t['function']['name'] for t in tools_payload]}")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if tools_payload:
            payload["tools"] = tools_payload
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            
            # Log response error sebelum raise
            if response.status_code >= 400:
                logger.error(f"DeepSeek API error {response.status_code}: {response.text}")
                logger.error(f"Payload sent: {json.dumps(payload, default=str)[:1000]}")
            
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")

        logger.info(f"🔍 finish_reason={finish_reason} | has_tool_calls={bool(message.get('tool_calls'))}")

        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                try:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_calls.append(ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=args
                    ))
                    logger.info(f"🔧 Tool call: {tc['function']['name']} | args={str(args)[:150]}")
                except Exception as e:
                    logger.error(f"Failed to parse tool call: {e}")

            yield LlmStreamChunk(
                content=message.get("content"),
                tool_calls=tool_calls,
                finish_reason=finish_reason
            )
        else:
            content = message.get("content") or ""
            words = content.split(" ")
            for i, word in enumerate(words):
                text = word if i == len(words) - 1 else word + " "
                yield LlmStreamChunk(
                    content=text,
                    finish_reason=finish_reason if i == len(words) - 1 else None
                )

    def validate_tools(self, tools: list[ToolSchema]) -> list[str]:
        # DeepSeek mendukung tool calling — semua tool valid
        return []


# ─────────────────────────────────────────────
# 2. UserResolver
#
# Baca identity user dari header HTTP.
# Sesuaikan dengan sistem auth Anda (JWT, cookie, OAuth, dll).
# ─────────────────────────────────────────────

class HeaderUserResolver(UserResolver):
    """
    Resolver sederhana berbasis header.
    - X-User-Id   : user identifier (default: 'anonymous')
    - X-Is-Admin  : 'true' untuk akses admin (default: user biasa)

    Untuk production, ganti dengan validasi JWT/OAuth yang sebenarnya.
    """

    async def resolve_user(self, request_context: RequestContext) -> User:
        user_id = request_context.get_header("X-User-Id") or "anonymous"
        is_admin = request_context.get_header("X-Is-Admin", "").lower() == "true"
        groups = ["admin", "user"] if is_admin else ["user"]
        return User(
            id=user_id,
            email=f"{user_id}@app.local",
            group_memberships=groups,
        )


# ─────────────────────────────────────────────
# 3. BGE-M3 embedding function untuk ChromaDB
# ─────────────────────────────────────────────

def make_bge_m3_embedding(hf_api_url: str, hf_token: str):
    """
    Factory yang mengembalikan embedding function kompatibel ChromaDB.
    ChromaDB mengharapkan format list[list[float]].
    """

    def bge_m3_embedding_function(texts: list[str]) -> list[list[float]]:
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        headers = {"Authorization": f"Bearer {hf_token}"}
        try:
            response = requests.post(
                hf_api_url,
                headers=headers,
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            result = response.json()

            # Validasi format output
            if (
                isinstance(result, list)
                and len(result) > 0
                and isinstance(result[0], list)
            ):
                return result

            raise ValueError(
                f"Format embedding tidak valid dari HuggingFace API: "
                f"expected list[list[float]], got {type(result).__name__}"
            )
        except requests.RequestException as e:
            logger.error(f"HuggingFace API request gagal: {e}")
            raise
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"BGE-M3 embedding error: {e}")
            raise

    return bge_m3_embedding_function


# ─────────────────────────────────────────────
# 4. initialize_vanna — factory function utama
#
# Mengembalikan Agent (bukan VannaInstance 0.x).
# Agent inilah yang dipanggil dari routes.py via agent.send_message().
# ─────────────────────────────────────────────

def initialize_vanna(
    mysql_host: str,
    mysql_user: str,
    mysql_password: str,
    mysql_database: str,
    mysql_port: int = 3306,
    hf_token: str = "",
    hf_api_url: str = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-m3",
    chromadb_path: str = "./data/chromadb",
    embedding_model: str = "BAAI/bge-m3",  # dipertahankan untuk kompatibilitas config
    deepseek_api_key: str = None,
    deepseek_api_base: str = "https://api.deepseek.com/v1",
) -> Agent:
    """
    Inisialisasi dan kembalikan Vanna 2.0 Agent yang siap dipakai.

    Komponen yang dibuat:
    - DeepSeekLlmService   : LLM untuk generate SQL
    - MySQLRunner          : koneksi dan eksekusi query ke MySQL
    - ChromaAgentMemory    : vector store dengan BGE-M3 embedding
    - ToolRegistry         : daftar tool yang bisa dipakai agent
    - HeaderUserResolver   : identifikasi user dari request header
    - Agent                : orkestrator utama Vanna 2.0

    Returns:
        Agent instance yang siap dipanggil via agent.send_message()
    """
    if not deepseek_api_key:
        raise ValueError("deepseek_api_key wajib diisi untuk Vanna 2.0")

    # Buat direktori ChromaDB jika belum ada
    Path(chromadb_path).mkdir(parents=True, exist_ok=True)

    # ── LLM ──────────────────────────────────
    logger.info("Initializing DeepSeek LlmService...")
    llm = DeepSeekLlmService(
        api_key=deepseek_api_key,
        db_name=mysql_database,
        model="deepseek-chat",
        base_url=deepseek_api_base,
        schema_context=""  # Opsional: bisa diisi dengan schema DB untuk bantu LLM generate SQL lebih akurat
    )
    logger.info("✓ DeepSeekLlmService ready")

    # ── Database ─────────────────────────────
    logger.info(f"Connecting to MySQL at {mysql_host}:{mysql_port}/{mysql_database}...")
    db_tool = RunSqlTool(
        sql_runner=MySQLRunner(
            host=mysql_host,
            port=mysql_port,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
        )
    )
    logger.info("✓ MySQLRunner ready")

    # ── Agent Memory (ChromaDB + BGE-M3) ─────
    logger.info(f"Initializing ChromaAgentMemory at {chromadb_path}...")
    embedding_fn = make_bge_m3_embedding(hf_api_url, hf_token)
    agent_memory = ChromaAgentMemory(
        collection_name="vanna_tool_memory",
        persist_directory=chromadb_path,
        embedding_function=embedding_fn,
    )
    logger.info("✓ ChromaAgentMemory ready")

    # ── Tool Registry ─────────────────────────
    # Akses tool dikontrol via access_groups:
    # - RunSqlTool              : semua user bisa eksekusi query
    # - SaveQuestionToolArgsTool: hanya admin yang bisa simpan memory baru
    # - SearchSavedCorrectTool  : semua user bisa cari dari memory
    # - SaveTextMemoryTool      : semua user bisa simpan text memory (DDL, docs)
    tools = ToolRegistry()
    retrieval_tool = GetDdlFromMemoryTool()
    tools.register_local_tool(retrieval_tool, access_groups=["admin", "user"])
    tools.register_local_tool(db_tool, access_groups=["admin", "user"])
    tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=["admin"])
    tools.register_local_tool(
        SearchSavedCorrectToolUsesTool(), access_groups=["admin", "user"]
    )
    tools.register_local_tool(
        SaveTextMemoryTool(), access_groups=["admin", "user"]
    )
    logger.info("✓ ToolRegistry configured")

    # ── Agent ─────────────────────────────────
    agent = Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=HeaderUserResolver(),
        agent_memory=agent_memory,
        system_prompt_builder=SchemaAwareSystemPromptBuilder(
            llm_service=llm,
            db_name=mysql_database
        ),
    )
    logger.info("✓ Vanna 2.0 Agent ready — DeepSeek + ChromaDB(BGE-M3) + MySQL")

    
    # logger.info("✅ system_prompt_builder overridden with schema context")

    return agent