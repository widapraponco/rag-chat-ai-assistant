"""
Vanna 2.0 initialization with BGE-M3 embeddings, ChromaDB, DeepSeek LLM, and MySQL
"""
import json
import logging
import asyncio
import requests
from pathlib import Path
from typing import AsyncGenerator

import httpx

# Vanna 2.0 core
from vanna import Agent
from vanna.core.llm import LlmService, LlmRequest, LlmResponse
from vanna.core.llm.models import LlmStreamChunk
from vanna.core.registry import ToolRegistry, ToolSchema
from vanna.core.user import UserResolver, User, RequestContext

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
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

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
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._build_payload(request, stream=False),
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = self._clean_usage(data.get("usage", {}))

        return LlmResponse(content=content, model=self.model, usage=usage)

    async def stream_request(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._build_payload(request, stream=True),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    if text := delta.get("content"):
                        yield LlmStreamChunk(content=text)

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
        base_url=deepseek_api_base,
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
    )
    logger.info("✓ Vanna 2.0 Agent ready — DeepSeek + ChromaDB(BGE-M3) + MySQL")

    return agent