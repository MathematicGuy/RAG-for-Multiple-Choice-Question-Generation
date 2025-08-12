"""
PDF Document Processor for Clean Architecture.
Handles PDF document loading and preprocessing.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from Domain.entities import Document
from Domain.repositories import DocumentProcessorRepository


class DocumentProcessor(DocumentProcessorRepository):
    """PDF document processor implementation"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            separators: Custom separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    async def load_document(self, file_path: str) -> List[Document]:
        """
        Load and process a PDF document.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document entities

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() != '.pdf':
            raise ValueError(f"Unsupported file format: {path.suffix}")

        try:
            # Load PDF using PyPDFLoader
            loader = PyPDFLoader(file_path)
            langchain_docs = await asyncio.to_thread(loader.load)

            # Convert to domain documents
            documents = []
            for i, doc in enumerate(langchain_docs):
                domain_doc = Document(
                    content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "page_number": i + 1,
                        "total_pages": len(langchain_docs),
                        "file_name": path.name,
                        "file_path": str(path)
                    },
                    source=path.name
                )
                documents.append(domain_doc)

            return documents

        except Exception as e:
            raise RuntimeError(f"Failed to load document {file_path}: {e}")

    async def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        chunked_documents = []

        for doc in documents:
            try:
                # Split text into chunks
                chunks = await asyncio.to_thread(
                    self.text_splitter.split_text, doc.content
                )

                # Create chunked documents
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Skip empty chunks
                        chunked_doc = Document(
                            content=chunk,
                            metadata={
                                **doc.metadata,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "original_page": doc.metadata.get("page_number", 1)
                            },
                            source=doc.source
                        )
                        chunked_documents.append(chunked_doc)

            except Exception as e:
                print(f"⚠️ Failed to chunk document from {doc.source}: {e}")
                continue

        return chunked_documents

    async def process_document(self, file_path: str) -> List[Document]:
        """
        Complete document processing pipeline.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of processed and chunked documents
        """
        # Load document
        documents = await self.load_document(file_path)

        # Chunk documents
        chunked_documents = await self.chunk_documents(documents)

        return chunked_documents

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return [".pdf"]

    async def validate_document(self, file_path: str) -> bool:
        """
        Validate if document can be processed.

        Args:
            file_path: Path to the document

        Returns:
            True if document is valid
        """
        try:
            path = Path(file_path)

            # Check if file exists
            if not path.exists():
                return False

            # Check file format
            if path.suffix.lower() not in self.get_supported_formats():
                return False

            # Try to load first page to validate PDF
            loader = PyPDFLoader(file_path)
            docs = await asyncio.to_thread(loader.load)

            return len(docs) > 0 and bool(docs[0].page_content.strip())

        except Exception:
            return False

    async def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get document metadata and information.

        Args:
            file_path: Path to the document

        Returns:
            Dictionary with document information
        """
        try:
            path = Path(file_path)
            documents = await self.load_document(file_path)

            if not documents:
                return {}

            total_content = " ".join(doc.content for doc in documents)

            return {
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "total_pages": len(documents),
                "total_characters": len(total_content),
                "total_words": len(total_content.split()),
                "estimated_chunks": len(total_content) // self.chunk_size + 1,
                "supported_format": True
            }

        except Exception as e:
            return {
                "error": str(e),
                "supported_format": False
            }


# Factory function for easy instantiation
def create_document_processor(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> DocumentProcessor:
    """
    Create a document processor instance.

    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        separators: Custom separators

    Returns:
        DocumentProcessor instance
    """
    return DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
