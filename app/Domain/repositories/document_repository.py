"""
Abstract repository interface for document operations.
Defines the contract for document processing implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from pathlib import Path
from Domain.entities.document import Document


class DocumentRepository(ABC):
    """Abstract repository for document operations"""

    @abstractmethod
    async def load_documents_from_folder(self, folder_path: str) -> Tuple[List[Document], List[str]]:
        """
        Load documents from a folder

        Args:
            folder_path: Path to folder containing documents

        Returns:
            Tuple of (loaded documents, list of filenames)

        Raises:
            DocumentLoadError: If loading fails
        """
        pass

    @abstractmethod
    async def load_document_from_file(self, file_path: str) -> Document:
        """
        Load a single document from file

        Args:
            file_path: Path to the document file

        Returns:
            Loaded document

        Raises:
            DocumentLoadError: If loading fails
        """
        pass

    @abstractmethod
    async def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        Split documents into smaller chunks

        Args:
            documents: List of documents to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of document chunks

        Raises:
            DocumentProcessingError: If chunking fails
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats

        Returns:
            List of supported file extensions
        """
        pass

    @abstractmethod
    async def validate_document(self, document: Document) -> bool:
        """
        Validate document content and metadata

        Args:
            document: Document to validate

        Returns:
            True if valid, False otherwise
        """
        pass
