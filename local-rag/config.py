"""
config.py
---------
Centralized configuration. Every tunable value in the project lives here
(overridable via environment variables) instead of being scattered across
files as magic numbers — the first thing that breaks once more than one
person/environment touches a project.
"""

import os

# --- Models -----------------------------------------------------------------
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "llama3.2")

# --- Chunking -----------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))

# --- Storage -----------------------------------------------------------------
DB_DIR = os.getenv("RAG_DB_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "documents")
MANIFEST_PATH = os.getenv("RAG_MANIFEST_PATH", "ingest_manifest.json")

# --- Retrieval -----------------------------------------------------------------
TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# --- Observability -----------------------------------------------------------------
LOG_DIR = os.getenv("RAG_LOG_DIR", "logs")
QUERY_LOG_FILE = os.getenv("RAG_QUERY_LOG_FILE", "queries.jsonl")
