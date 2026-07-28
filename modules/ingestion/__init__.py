"""
Ingestion Module: Text extraction, PDF cleaning, text chunking, and ChromaDB vector persistence
"""
from .ingest import run_ingestion

__all__ = ["run_ingestion"]
