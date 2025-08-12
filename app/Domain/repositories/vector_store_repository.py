"""
Abstract repository interface for vector store operations.
Defines the contract for vector store implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from Domain.entities.document import Document


class VectorStoreRepository(ABC):
    """Abstract repository for vector store operations"""

    @abstractmethod
    async def store_documents(self, documents: List[Document]) -> int:
        """
        Store documents in the vector database

        Args:
            documents: List of documents to store

        Returns:
            Number of chunks created

        Raises:
            VectorStoreError: If storage fails
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        query: str,
        k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Document]:
        """
        Search for similar documents

        Args:
            query: Search query
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar documents

        Raises:
            VectorStoreError: If search fails
        """
        pass

    @abstractmethod
    async def search_diverse(
        self,
        query: str,
        k: int = 5,
        diversity_threshold: float = 0.7
    ) -> List[Document]:
        """
        Search for diverse documents to avoid redundancy

        Args:
            query: Search query
            k: Number of documents to retrieve
            diversity_threshold: Minimum diversity threshold

        Returns:
            List of diverse documents

        Raises:
            VectorStoreError: If search fails
        """
        pass

    @abstractmethod
    async def get_document_count(self) -> int:
        """
        Get total number of documents in the store

        Returns:
            Number of documents
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        Clear all documents from the vector store

        Raises:
            VectorStoreError: If clearing fails
        """
        pass

    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if vector store is initialized

        Returns:
            True if initialized, False otherwise
        """
        pass
