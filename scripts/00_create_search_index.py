#!/usr/bin/env python3
"""
Azure AI Search Index Creation Script

Creates/updates search index schema for RAG system.
Input: configs/search_schema.json (auto-generated if not exists)
Output: Search index ready for use

Usage:
    python scripts/00_create_search_index.py
    python scripts/00_create_search_index.py --index-name custom-index-v2
"""

import json
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, Any
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_azure_config() -> Dict[str, str]:
    """Load Azure Search configuration from environment variables."""
    config = {
        'endpoint': os.getenv('AZURE_SEARCH_ENDPOINT'),
        'admin_key': os.getenv('AZURE_INSERTION_KEY'),  # Admin key for index creation
        'query_key': os.getenv('AZURE_SEARCH_KEY'),     # Query key for read operations
        'api_version': os.getenv('AZURE_SEARCH_API_VERSION', '2023-11-01')
    }
    
    # Validate required config
    missing = [k for k, v in config.items() if not v and k != 'query_key']
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")
    
    return config


def validate_azure_connection(config: Dict[str, str]) -> bool:
    """Validate connection to Azure Search service."""
    try:
        url = f"{config['endpoint']}/indexes"
        headers = {
            'api-key': config['admin_key'],
            'Content-Type': 'application/json'
        }
        params = {'api-version': config['api_version']}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            indexes = response.json().get('value', [])
            logging.info(f"✅ Connected to Azure Search. Found {len(indexes)} existing indexes.")
            return True
        else:
            logging.error(f"❌ Azure Search connection failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Failed to connect to Azure Search: {e}")
        return False


def get_embedding_dimensions() -> int:
    """Get embedding dimensions from environment config."""
    model_name = os.getenv('EMBEDDING_1_MODEL_NAME', 'text-embedding-3-small')
    
    # Common Azure OpenAI embedding model dimensions
    model_dims = {
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'text-embedding-ada-002': 1536,
    }
    
    dims = model_dims.get(model_name, 1536)
    logging.info(f"Using embedding dimensions: {dims} (model: {model_name})")
    return dims


def create_default_schema(index_name: str) -> Dict[str, Any]:
    """Create default search index schema."""
    embedding_dims = get_embedding_dimensions()

    schema = {
        "name": index_name,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "searchable": False, "filterable": False, "retrievable": True},
            {"name": "content", "type": "Edm.String", "searchable": True, "filterable": False, "retrievable": True, "sortable": False, "facetable": False, "analyzer": "es.microsoft"},
            {"name": "contentVector", "type": "Collection(Edm.Single)", "searchable": True, "filterable": False, "retrievable": True, "sortable": False, "facetable": False, "dimensions": embedding_dims, "vectorSearchProfile": "default-vector-profile"},
            {"name": "doc_id", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True, "sortable": True, "facetable": True},
            {"name": "page", "type": "Edm.Int32", "searchable": False, "filterable": True, "retrievable": True, "sortable": True, "facetable": True},
            {"name": "content_type", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True, "sortable": False, "facetable": True},
            {"name": "chunk_id", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True},
            {"name": "source_uri", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True},
            {"name": "emb_version", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True},
            {"name": "doc_hash", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True},
            {"name": "ingested_at", "type": "Edm.DateTimeOffset", "searchable": False, "filterable": False, "retrievable": True},
            {"name": "fetched_at", "type": "Edm.DateTimeOffset", "searchable": False, "filterable": False, "retrievable": True},
            {"name": "chunk_method", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True}
        ],
        "vectorSearch": {
            "algorithms": [
                {
                    "name": "default-hnsw",
                    "kind": "hnsw",
                    "hnswParameters": {
                        "metric": "cosine",
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500
                    }
                }
            ],
            "profiles": [
                {
                    "name": "default-vector-profile",
                    "algorithm": "default-hnsw"
                }
            ]
        },
        "semantic": {
            "configurations": [
                {
                    "name": "default",
                    "prioritizedFields": {
                        "titleField": None,
                        "prioritizedContentFields": [
                            {"fieldName": "content"}
                        ],
                        "prioritizedKeywordsFields": [
                            {"fieldName": "doc_id"}
                        ]
                    }
                }
            ]
        }
    }
    return schema


def save_schema_file(schema: Dict[str, Any], schema_file: str):
    """Save schema to configs/search_schema.json."""
    os.makedirs(os.path.dirname(schema_file), exist_ok=True)
    
    with open(schema_file, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Schema saved to: {schema_file}")


def load_schema_file(schema_file: str) -> Dict[str, Any]:
    """Load schema from configs/search_schema.json."""
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    logging.info(f"Schema loaded from: {schema_file}")
    return schema


def check_index_exists(config: Dict[str, str], index_name: str) -> bool:
    """Check if index already exists."""
    try:
        url = f"{config['endpoint']}/indexes/{index_name}"
        headers = {
            'api-key': config['admin_key'],
            'Content-Type': 'application/json'
        }
        params = {'api-version': config['api_version']}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        return response.status_code == 200
        
    except Exception as e:
        logging.error(f"Error checking if index exists: {e}")
        return False


def create_index(config: Dict[str, str], schema: Dict[str, Any]) -> bool:
    """Create the search index."""
    index_name = schema['name']
    
    try:
        # Check if index already exists
        if check_index_exists(config, index_name):
            logging.warning(f"⚠️  Index '{index_name}' already exists!")
            
            response = input(f"Do you want to delete and recreate it? [y/N]: ")
            if response.lower() != 'y':
                logging.info("Index creation cancelled.")
                return False
            
            # Delete existing index
            delete_url = f"{config['endpoint']}/indexes/{index_name}"
            headers = {
                'api-key': config['admin_key'],
                'Content-Type': 'application/json'
            }
            params = {'api-version': config['api_version']}
            
            delete_response = requests.delete(delete_url, headers=headers, params=params, timeout=10)
            if delete_response.status_code == 204:
                logging.info(f"✅ Deleted existing index '{index_name}'")
            else:
                logging.error(f"❌ Failed to delete index: {delete_response.status_code} - {delete_response.text}")
                return False
        
        # Create new index
        url = f"{config['endpoint']}/indexes"
        headers = {
            'api-key': config['admin_key'],
            'Content-Type': 'application/json'
        }
        params = {'api-version': config['api_version']}
        
        response = requests.post(url, headers=headers, params=params, json=schema, timeout=30)
        
        if response.status_code == 201:
            logging.info(f"✅ Successfully created index '{index_name}'")
            
            # Log index details
            logging.info(f"Index details:")
            logging.info(f"  - Fields: {len(schema['fields'])}")
            logging.info(f"  - Vector search enabled: {bool(schema.get('vectorSearch'))}")
            logging.info(f"  - Semantic search enabled: {bool(schema.get('semantic'))}")
            
            return True
        else:
            logging.error(f"❌ Failed to create index: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Error creating index: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create Azure AI Search index")
    parser.add_argument(
        '--index-name',
        default='rag-toy-index-v1',
        help='Name of the search index (default: rag-toy-index-v1)'
    )
    parser.add_argument(
        '--schema-file',
        default='configs/search_schema.json',
        help='Path to schema file (default: configs/search_schema.json)'
    )
    parser.add_argument(
        '--create-schema',
        action='store_true',
        help='Create default schema file and exit (don\'t create index)'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        # Load Azure configuration
        logging.info("Loading Azure Search configuration...")
        azure_config = load_azure_config()
        
        # Validate connection
        logging.info("Validating Azure Search connection...")
        if not validate_azure_connection(azure_config):
            return 1
        
        # Handle schema creation
        if args.create_schema:
            logging.info(f"Creating default schema: {args.schema_file}")
            schema = create_default_schema(args.index_name)
            save_schema_file(schema, args.schema_file)
            logging.info("✅ Default schema created. Review and modify as needed.")
            return 0
        
        # Load or create schema
        if os.path.exists(args.schema_file):
            logging.info(f"Loading existing schema: {args.schema_file}")
            schema = load_schema_file(args.schema_file)
            # Update index name if provided
            schema['name'] = args.index_name
        else:
            logging.info(f"Creating default schema: {args.schema_file}")
            schema = create_default_schema(args.index_name)
            save_schema_file(schema, args.schema_file)
        
        # Create the index
        logging.info(f"Creating Azure Search index: {args.index_name}")
        if create_index(azure_config, schema):
            logging.info("✅ Index creation completed successfully!")
            logging.info(f"Index URL: {azure_config['endpoint']}/indexes/{args.index_name}")
            return 0
        else:
            logging.error("❌ Index creation failed!")
            return 1
            
    except Exception as e:
        logging.error(f"❌ Script failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())