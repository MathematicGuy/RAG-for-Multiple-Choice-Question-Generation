"""
FAISS vector store implementation.
Implements VectorStoreRepository interface for FAISS vector database.
"""

import asyncio
from typing import List, Optional, Set
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document as LangChainDocument
from Domain.repositories.vector_store_repository import VectorStoreRepository
from Domain.entities.document import Document


class FAISSVectorStore(VectorStoreRepository):
    """FAISS vector store implementation"""

    def __init__(
        self,
        embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder",
        chunk_size: int = 500,
        diversity_threshold: float = 0.7
    ):
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.diversity_threshold = diversity_threshold
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_db: Optional[FAISS] = None
        self.chunker: Optional[SemanticChunker] = None
        self._initialized = False

    async def _initialize(self):
        """Initialize embeddings and chunker if not already done"""
        if not self._initialized:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name
            )

            # Initialize semantic chunker
            self.chunker = SemanticChunker(
                embeddings=self.embeddings,
                buffer_size=1,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
                add_start_index=True
            )

            self._initialized = True

    async def store_documents(self, documents: List[Document]) -> int:
        """
        Store documents in the vector database

        Args:
            documents: List of documents to store

        Returns:
            Number of chunks created
        """
        await self._initialize()

        if self.chunker is None or self.embeddings is None:
            raise RuntimeError("Vector store not properly initialized")

        # Convert to LangChain format
        langchain_docs = []
        for doc in documents:
            langchain_doc = LangChainDocument(
                page_content=doc.content,
                metadata=doc.metadata.copy()
            )
            langchain_doc.metadata["source"] = doc.source
            langchain_docs.append(langchain_doc)

        # Split documents into chunks
        chunks = self.chunker.split_documents(langchain_docs)

        # Create or update vector database
        if self.vector_db is None:
            self.vector_db = FAISS.from_documents(chunks, embedding=self.embeddings)
        else:
            # Add to existing database
            new_db = FAISS.from_documents(chunks, embedding=self.embeddings)
            self.vector_db.merge_from(new_db)

        return len(chunks)

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
        """
        if not self.is_initialized() or self.vector_db is None:
            return []

        # Perform similarity search with scores
        results = self.vector_db.similarity_search_with_score(query, k=k*2)

        # Filter by similarity threshold and convert to domain entities
        documents = []
        for langchain_doc, score in results:
            # FAISS returns distance, lower is better
            # Convert to similarity score (higher is better)
            similarity_score = 1.0 / (1.0 + score)

            if similarity_score >= similarity_threshold and len(documents) < k:
                doc = Document(
                    content=langchain_doc.page_content,
                    metadata=langchain_doc.metadata.copy(),
                    source=langchain_doc.metadata.get("source", "Unknown")
                )
                documents.append(doc)

        return documents

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
        """
        if not self.is_initialized():
            return []

        # Get more candidates than needed for diversity selection
        candidates = await self.search_similar(query, k=k*3, similarity_threshold=0.2)

        if not candidates:
            return []

        # Select diverse documents
        selected = [candidates[0]]  # Always include the most relevant

        for candidate in candidates[1:]:
            if len(selected) >= k:
                break

            # Check diversity with already selected documents
            is_diverse = True
            for selected_doc in selected:
                similarity = self._calculate_text_similarity(
                    candidate.content,
                    selected_doc.content
                )
                if similarity > diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(candidate)

        return selected[:k]

    async def get_document_count(self) -> int:
        """
        Get total number of documents in the store

        Returns:
            Number of documents
        """
        if not self.is_initialized() or self.vector_db is None:
            return 0

        return self.vector_db.index.ntotal

    async def clear(self) -> None:
        """
        Clear all documents from the vector store
        """
        self.vector_db = None

    def is_initialized(self) -> bool:
        """
        Check if vector store is initialized

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized and self.vector_db is not None

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using word overlap

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def get_embedding_info(self) -> dict:
        """
        Get information about the embedding model

        Returns:
            Dictionary with embedding model information
        """
        return {
            "model_name": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "diversity_threshold": self.diversity_threshold,
            "initialized": self._initialized,
            "document_count": asyncio.create_task(self.get_document_count()) if self._initialized else 0
        }
