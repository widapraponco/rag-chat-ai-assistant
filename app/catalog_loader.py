"""
Catalog loader - Extract table and column metadata from dbt catalog.json
and prepare them for Vanna embedding and storage in ChromaDB.
"""
import json
import os
from typing import Dict, List, Tuple
from pathlib import Path


class CatalogLoader:
    """Load and parse dbt catalog.json for Vanna integration"""
    
    def __init__(self, catalog_path: str):
        """
        Initialize catalog loader
        
        Args:
            catalog_path: Path to dbt catalog.json file
        """
        self.catalog_path = catalog_path
        self.catalog_data = self._load_catalog()
        
    def _load_catalog(self) -> Dict:
        """Load catalog.json file"""
        if not os.path.exists(self.catalog_path):
            raise FileNotFoundError(f"Catalog not found: {self.catalog_path}")
        
        with open(self.catalog_path, 'r') as f:
            return json.load(f)
    
    def get_sources(self) -> Dict[str, Dict]:
        """Extract all sources from catalog"""
        return self.catalog_data.get('sources', {})
    
    def get_models(self) -> Dict[str, Dict]:
        """Extract all models from catalog"""
        return self.catalog_data.get('nodes', {})
    
    def extract_table_descriptions(self) -> List[Tuple[str, str, str, Dict]]:
        """
        Extract table descriptions and column info for embedding
        
        Returns:
            List of tuples: (table_name, schema_name, description, columns_dict)
        """
        tables = []
        
        # Extract from sources
        for source_key, source_data in self.get_sources().items():
            metadata = source_data.get('metadata', {})
            table_name = metadata.get('name', '')
            schema_name = metadata.get('schema', '')
            database = metadata.get('database', '')
            columns = source_data.get('columns', {})
            
            # Create table description
            table_desc = f"Table: {table_name} in schema {schema_name}"
            if metadata.get('comment'):
                table_desc += f". {metadata['comment']}"
            
            # Extract column info
            column_info = {}
            for col_name, col_data in columns.items():
                col_type = col_data.get('type', 'unknown')
                col_comment = col_data.get('comment', '')
                column_info[col_name] = {
                    'type': col_type,
                    'comment': col_comment,
                    'index': col_data.get('index', 0)
                }
            
            tables.append((table_name, schema_name, table_desc, column_info))
        
        # Extract from models
        for model_key, model_data in self.get_models().items():
            metadata = model_data.get('metadata', {})
            table_name = metadata.get('name', '')
            schema_name = metadata.get('schema', '')
            database = metadata.get('database', '')
            columns = model_data.get('columns', {})
            
            # Create table description
            table_desc = f"Model: {table_name} in schema {schema_name}"
            if metadata.get('comment'):
                table_desc += f". {metadata['comment']}"
            
            # Extract column info
            column_info = {}
            for col_name, col_data in columns.items():
                col_type = col_data.get('type', 'unknown')
                col_comment = col_data.get('comment', '')
                column_info[col_name] = {
                    'type': col_type,
                    'comment': col_comment,
                    'index': col_data.get('index', 0)
                }
            
            tables.append((table_name, schema_name, table_desc, column_info))
        
        return tables
    
    def get_table_schema_ddl(self) -> Dict[str, str]:
        """
        Generate CREATE TABLE DDL statements from catalog
        
        Returns:
            Dict mapping table names to DDL statements
        """
        ddl_statements = {}
        
        for table_name, schema_name, desc, columns in self.extract_table_descriptions():
            full_table_name = f"{schema_name}.{table_name}" if schema_name else table_name
            
            # Build column definitions
            col_defs = []
            for col_name, col_info in sorted(columns.items(), key=lambda x: x[1]['index']):
                col_type = col_info.get('type', 'VARCHAR(255)')
                col_defs.append(f"  {col_name} {col_type}")
            
            # Generate DDL
            ddl = f"CREATE TABLE {full_table_name} (\n"
            ddl += ",\n".join(col_defs)
            ddl += "\n);"
            
            ddl_statements[full_table_name] = ddl
        
        return ddl_statements
    
    def get_column_metadata(self) -> List[Dict]:
        """
        Get detailed column metadata for embedding
        
        Returns:
            List of column metadata dictionaries
        """
        column_metadata = []
        
        for table_name, schema_name, desc, columns in self.extract_table_descriptions():
            for col_name, col_info in columns.items():
                full_table = f"{schema_name}.{table_name}" if schema_name else table_name
                
                metadata_entry = {
                    'table': full_table,
                    'schema': schema_name,
                    'table_name': table_name,
                    'column': col_name,
                    'type': col_info.get('type', 'unknown'),
                    'description': col_info.get('comment', f'{col_name} column'),
                    'full_text': f"Column {col_name} ({col_info.get('type', 'unknown')}) in {full_table}"
                }
                column_metadata.append(metadata_entry)
        
        return column_metadata


def load_dbt_catalog(catalog_path: str) -> CatalogLoader:
    """Factory function to load DBT catalog"""
    return CatalogLoader(catalog_path)
