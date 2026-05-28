"""
Catalog loader - Extract table and column metadata from dbt catalog.json
and prepare them for Vanna embedding and storage in ChromaDB.

Supports both local file paths and HTTP URLs.
"""
import json
import os
import logging
import requests
from typing import Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class CatalogLoader:
    def __init__(self, catalog_path: str):
        """
        Load catalog from either HTTP URL or local file path.
        
        Args:
            catalog_path: Either a local file path or HTTP(S) URL
        """
        self.catalog_path = catalog_path
        self.catalog = self._load_catalog()
    
    def _load_catalog(self) -> dict:
        """Load catalog from HTTP URL or local file."""
        if self.catalog_path.startswith(('http://', 'https://')):
            return self._load_from_http()
        else:
            return self._load_from_file()
    
    def _load_from_http(self) -> dict:
        """Fetch catalog from HTTP URL."""
        try:
            logger.info(f"Loading catalog from HTTP: {self.catalog_path}")
            response = requests.get(self.catalog_path, timeout=10)
            response.raise_for_status()
            catalog = response.json()
            logger.info(f"✓ Catalog loaded from HTTP")
            return catalog
        except requests.RequestException as e:
            logger.error(f"Failed to load catalog from HTTP: {e}")
            raise
    
    def _load_from_file(self) -> dict:
        """Load catalog from local file."""
        try:
            logger.info(f"Loading catalog from file: {self.catalog_path}")
            with open(self.catalog_path) as f:
                catalog = json.load(f)
            logger.info(f"✓ Catalog loaded from file")
            return catalog
        except FileNotFoundError as e:
            logger.error(f"Catalog file not found: {self.catalog_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in catalog file: {e}")
            raise

    def extract_table_descriptions(self):
        """
        Prioritaskan nodes (dbt models) — ini yang kohesif dan siap diquery.
        Fallback ke sources hanya jika tidak ada nodes sama sekali.
        """
        results = []

        # ── 1. dbt models (nodes) — PRIORITAS UTAMA ──────────────────────────
        for key, node in self.catalog.get('nodes', {}).items():
            meta = node.get('metadata', {})
            schema = meta.get('schema', '')
            table = meta.get('name', '')
            
            # Buat deskripsi dari nama model — ubah snake_case ke kalimat
            description = table.replace('_', ' ').title()
            
            columns = node.get('columns', {})
            col_list = [
                {
                    'name': col['name'],
                    'type': col.get('type', ''),
                    'comment': col.get('comment', '') or ''
                }
                for col in columns.values()
            ]
            results.append((table, schema, description, col_list))

        # ── 2. sources (raw tables) — hanya jika tidak ada nodes ─────────────
        if not results:
            for key, source in self.catalog.get('sources', {}).items():
                meta = source.get('metadata', {})
                schema = meta.get('schema', '')
                table = meta.get('name', '')
                description = f"Raw table: {table}"
                columns = source.get('columns', {})
                col_list = [
                    {
                        'name': col['name'],
                        'type': col.get('type', ''),
                        'comment': col.get('comment', '') or ''
                    }
                    for col in columns.values()
                ]
                results.append((table, schema, description, col_list))

        return results

    def get_table_schema_ddl(self) -> dict:
        """
        Generate DDL dari nodes dbt — dengan deskripsi kolom yang informatif.
        """
        ddl_map = {}

        # Nodes dbt models
        for key, node in self.catalog.get('nodes', {}).items():
            meta = node.get('metadata', {})
            schema = meta.get('schema', '')
            table = meta.get('name', '')
            full_key = f"{schema}.{table}"
            columns = node.get('columns', {})

            col_defs = []
            for col in sorted(columns.values(), key=lambda c: c.get('index', 0)):
                col_name = col['name']
                col_type = col.get('type', 'varchar')
                comment = col.get('comment', '')
                # Tambahkan comment sebagai inline SQL comment jika ada
                comment_str = f"  -- {comment}" if comment else ""
                col_defs.append(f"  {col_name} {col_type}{comment_str}")

            ddl = f"CREATE TABLE {full_key} (\n"
            ddl += ",\n".join(col_defs)
            ddl += "\n);"
            ddl_map[full_key] = ddl

        # Fallback ke sources jika tidak ada nodes
        if not ddl_map:
            for key, source in self.catalog.get('sources', {}).items():
                meta = source.get('metadata', {})
                schema = meta.get('schema', '')
                table = meta.get('name', '')
                full_key = f"{schema}.{table}"
                columns = source.get('columns', {})

                col_defs = []
                for col in sorted(columns.values(), key=lambda c: c.get('index', 0)):
                    col_defs.append(f"  {col['name']} {col.get('type', 'varchar')}")

                ddl = f"CREATE TABLE {full_key} (\n"
                ddl += ",\n".join(col_defs)
                ddl += "\n);"
                ddl_map[full_key] = ddl

        return ddl_map

    def get_column_metadata(self) -> list:
        """Return flat list of column metadata untuk endpoint /api/columns."""
        results = []
        
        # Prioritaskan nodes
        source = self.catalog.get('nodes', {}) or self.catalog.get('sources', {})
        for key, node in source.items():
            meta = node.get('metadata', {})
            table = f"{meta.get('schema', '')}.{meta.get('name', '')}"
            for col in node.get('columns', {}).values():
                results.append({
                    'table': table,
                    'name': col['name'],
                    'type': col.get('type', ''),
                    'comment': col.get('comment', '') or ''
                })
        return results


def load_dbt_catalog(catalog_path: str) -> CatalogLoader:
    """Factory function to load DBT catalog from local or HTTP source"""
    return CatalogLoader(catalog_path)