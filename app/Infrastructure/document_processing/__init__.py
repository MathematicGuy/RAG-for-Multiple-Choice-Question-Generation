"""
Document processing infrastructure package.
Provides document loaders and processors for the RAG system.
"""

from .pdf_processor import DocumentProcessor, create_document_processor

__all__ = [
    "DocumentProcessor",
    "create_document_processor"
]
