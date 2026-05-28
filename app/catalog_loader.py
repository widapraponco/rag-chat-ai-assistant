"""
Catalog loader - Extract table and column metadata from dbt catalog.json
and prepare them for Vanna embedding and storage in ChromaDB.
"""
import json
import os
from typing import Dict, List, Tuple
from pathlib import Path

class CatalogLoader:
    def __init__(self, catalog_path: str):
        with open(catalog_path) as f:
            self.catalog = json.load(f)

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
    """Factory function to load DBT catalog"""
    return CatalogLoader(catalog_path)
