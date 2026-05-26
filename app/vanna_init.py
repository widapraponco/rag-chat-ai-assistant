"""
Vanna 2.0 initialization with BGE-M3 embeddings and ChromaDB
"""
import os
import logging
from typing import Optional
import mysql.connector
from pathlib import Path

# Vanna imports
try:
    from vanna.legacy.chromadb import ChromaDB_VectorStore
except ImportError:
    from vanna.chromadb import ChromaDB_VectorStore
try:
    from vanna.legacy.vannadb import VannaDB_VectorStore as VannaDB
except ImportError:
    try:
        from vanna.vannadb import VannaDB
    except ImportError:
        VannaDB = None

# Embeddings
from sentence_transformers import SentenceTransformer

# DeepSeek LLM
from openai import OpenAI as DeepSeekClient  # DeepSeek uses OpenAI-compatible API

logger = logging.getLogger(__name__)


class VannaConfig:
    """Configuration for Vanna with BGE-M3 embeddings"""
    
    def __init__(
        self,
        mysql_host: str,
        mysql_user: str,
        mysql_password: str,
        mysql_database: str,
        mysql_port: int = 3306,
        chromadb_path: str = './data/chromadb',
        embedding_model: str = 'BAAI/bge-m3',
    ):
        """Initialize Vanna configuration"""
        self.mysql_host = mysql_host
        self.mysql_user = mysql_user
        self.mysql_password = mysql_password
        self.mysql_database = mysql_database
        self.mysql_port = mysql_port
        self.chromadb_path = chromadb_path
        self.embedding_model = embedding_model
        
        # Create chromadb path if doesn't exist
        Path(chromadb_path).mkdir(parents=True, exist_ok=True)
    
    def get_mysql_connection_string(self) -> str:
        """Build MySQL connection string"""
        return f"mysql+mysqlconnector://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
    
    def test_mysql_connection(self) -> bool:
        """Test MySQL connection"""
        try:
            conn = mysql.connector.connect(
                host=self.mysql_host,
                user=self.mysql_user,
                password=self.mysql_password,
                database=self.mysql_database,
                port=self.mysql_port
            )
            conn.close()
            logger.info("✓ MySQL connection successful")
            return True
        except Exception as e:
            logger.error(f"✗ MySQL connection failed: {e}")
            return False


class VannaInstance:
    """Vanna instance with BGE-M3, ChromaDB, and DeepSeek LLM integration"""
    
    def __init__(self, config: VannaConfig, deepseek_api_key: str = None, deepseek_api_base: str = None):
        """Initialize Vanna with embeddings, ChromaDB, and DeepSeek LLM"""
        self.config = config
        self.embedding_model = None
        self.vanna = None
        self.deepseek_client = None
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_api_base = deepseek_api_base or 'https://api.deepseek.com/v1'
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Vanna components"""
        try:
            logger.info("Initializing Vanna with BGE-M3 embeddings and DeepSeek LLM...")
            
            # Load BGE-M3 embedding model
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info("✓ BGE-M3 embedding model loaded")
            
            # Initialize DeepSeek client if API key provided
            if self.deepseek_api_key:
                logger.info("Initializing DeepSeek LLM client...")
                self.deepseek_client = DeepSeekClient(
                    api_key=self.deepseek_api_key,
                    base_url=self.deepseek_api_base
                )
                logger.info("✓ DeepSeek LLM client initialized")
            else:
                logger.warning("No DeepSeek API key provided - LLM functionality will be limited")
            
            # Initialize ChromaDB backend
            logger.info(f"Initializing ChromaDB at: {self.config.chromadb_path}")
            
            # Define embedding function using the loaded BGE-M3 model
            def bge_m3_embedding_function(texts):
                embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
                return embeddings.tolist()
            
            chroma = ChromaDB_VectorStore(config={
                'path': self.config.chromadb_path,
                'embedding_function': bge_m3_embedding_function,
                'collection_metadata': {'hnsw:space': 'cosine'}
            })
            logger.info("✓ ChromaDB initialized")
            
            # Initialize Vanna with ChromaDB for vector storage
            self.vanna = chroma
            logger.info("✓ Vanna instance created with ChromaDB backend and DeepSeek LLM")
            
            # Test MySQL connection
            if not self.config.test_mysql_connection():
                logger.warning("MySQL connection test failed - check credentials")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vanna: {e}")
            raise
    
    def add_sql_context(self, sql: str, description: str = None):
        """
        Add SQL context to Vanna knowledge base
        
        Args:
            sql: SQL statement or DDL
            description: Optional description of what this SQL does
        """
        try:
            if hasattr(self.vanna, 'add_ddl'):
                self.vanna.add_ddl(sql)
                logger.info(f"Added DDL to ChromaDB: {description or sql[:50]}")
            elif hasattr(self.vanna, 'add_sql_context'):
                self.vanna.add_sql_context(sql, description)
                logger.info(f"Added SQL context: {description or sql[:50]}")
        except Exception as e:
            logger.error(f"Failed to add SQL context: {e}")
    
    def add_documentation(self, text: str, metadata: dict = None):
        """
        Add documentation to knowledge base
        
        Args:
            text: Documentation text
            metadata: Optional metadata
        """
        try:
            if hasattr(self.vanna, 'add_documentation'):
                # Pass only 'text' since ChromaDB_VectorStore.add_documentation doesn't use metadata
                self.vanna.add_documentation(text)
                logger.info(f"Added documentation: {text[:50]}")
        except Exception as e:
            logger.error(f"Failed to add documentation: {e}")
    
    def generate_sql(self, question: str) -> Optional[str]:
        """
        Generate SQL from natural language question using DeepSeek
        
        Args:
            question: Natural language question
            
        Returns:
            Generated SQL string or None
        """
        try:
            if not self.deepseek_client:
                logger.error("DeepSeek client not initialized - no API key provided")
                return None
            
            logger.info(f"Generating SQL for: {question}")
            
            # Create prompt for DeepSeek
            system_prompt = "You are a SQL expert. Generate SQL queries based on natural language questions. Return only the SQL query without any markdown formatting or explanations."
            
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            sql = response.choices[0].message.content.strip()
            logger.info(f"Generated SQL: {sql}")
            return sql
        except Exception as e:
            logger.error(f"Failed to generate SQL: {e}")
            return None
    
    def execute_sql(self, sql: str) -> Optional[list]:
        """
        Execute SQL query against MySQL database
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Query results as list of dicts or None
        """
        try:
            conn = mysql.connector.connect(
                host=self.config.mysql_host,
                user=self.config.mysql_user,
                password=self.config.mysql_password,
                database=self.config.mysql_database,
                port=self.config.mysql_port
            )
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            logger.info(f"Query executed successfully, returned {len(results)} rows")
            return results
        except Exception as e:
            logger.error(f"Failed to execute SQL: {e}")
            return None
    
    def get_embedding(self, text: str) -> list:
        """
        Get embedding for text using BGE-M3
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embeddings = self.embedding_model.encode(text, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def add_table_context(self, table_description: str, ddl: str, column_info: dict = None):
        """
        Add table metadata to knowledge base
        
        Args:
            table_description: Description of the table
            ddl: CREATE TABLE DDL statement
            column_info: Dictionary of column metadata
        """
        try:
            # Add DDL as SQL context
            self.add_sql_context(ddl, table_description)
            
            # Add column information as documentation
            if column_info:
                col_docs = self._format_column_documentation(column_info)
                self.add_documentation(col_docs, {"type": "column_metadata"})
            
            logger.info(f"Added table context: {table_description}")
        except Exception as e:
            logger.error(f"Failed to add table context: {e}")
    
    @staticmethod
    def _format_column_documentation(column_info: dict) -> str:
        """Format column information as documentation"""
        doc_lines = ["Column Information:"]
        for col_name, col_data in column_info.items():
            col_type = col_data.get('type', 'unknown')
            col_comment = col_data.get('comment', '')
            doc_lines.append(f"- {col_name} ({col_type}): {col_comment}")
        return "\n".join(doc_lines)


def initialize_vanna(
    mysql_host: str,
    mysql_user: str,
    mysql_password: str,
    mysql_database: str,
    mysql_port: int = 3306,
    chromadb_path: str = './data/chromadb',
    embedding_model: str = 'BAAI/bge-m3',
    deepseek_api_key: str = None,
    deepseek_api_base: str = None
) -> VannaInstance:
    """Factory function to initialize Vanna with DeepSeek LLM"""
    config = VannaConfig(
        mysql_host=mysql_host,
        mysql_user=mysql_user,
        mysql_password=mysql_password,
        mysql_database=mysql_database,
        mysql_port=mysql_port,
        chromadb_path=chromadb_path,
        embedding_model=embedding_model
    )
    return VannaInstance(
        config,
        deepseek_api_key=deepseek_api_key,
        deepseek_api_base=deepseek_api_base
    )
