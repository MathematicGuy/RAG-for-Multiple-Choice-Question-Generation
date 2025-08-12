"""
Document Management Controllers for Clean Architecture.
Handles document upload, processing, and management.
"""

import time
from typing import Dict, Any, List
from pathlib import Path
from fastapi import HTTPException, BackgroundTasks, UploadFile, File

from Infrastructure.document_processing import DocumentProcessor
from Infrastructure.vector_stores.faiss_store import FAISSVectorStore
from config.settings import Settings


class DocumentController:
    """Controller for document management endpoints"""

    def __init__(
        self,
        document_processor: DocumentProcessor,
        vector_store: FAISSVectorStore,
        settings: Settings
    ):
        """
        Initialize document controller.

        Args:
            document_processor: Document processing service
            vector_store: Vector store for document storage
            settings: Application settings
        """
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.settings = settings

    async def upload_document(
        self,
        file: UploadFile,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """
        Upload and process a document.

        Args:
            file: Uploaded file
            background_tasks: Background task manager

        Returns:
            Upload response dictionary

        Raises:
            HTTPException: For validation or processing errors
        """
        try:
            # Validate file
            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail="No filename provided"
                )

            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are supported"
                )

            # Validate file size (10MB limit)
            max_size = 10 * 1024 * 1024  # 10MB
            content = await file.read()
            if len(content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail="File too large (max 10MB)"
                )

            # Reset file pointer
            await file.seek(0)

            # Save uploaded file
            file_path = self.settings.tmp_dir / file.filename

            # Ensure tmp directory exists
            self.settings.tmp_dir.mkdir(parents=True, exist_ok=True)

            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Start background processing
            background_tasks.add_task(
                self._process_document_background,
                str(file_path),
                file.filename
            )

            return {
                "message": f"Document {file.filename} uploaded successfully",
                "filename": file.filename,
                "file_size": len(content),
                "processing": "started in background",
                "upload_time": time.time()
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Upload failed: {str(e)}"
            )

    async def _process_document_background(
        self,
        file_path: str,
        filename: str
    ) -> None:
        """
        Process document in background.

        Args:
            file_path: Path to the uploaded file
            filename: Original filename
        """
        try:
            print(f"📄 Processing document: {filename}")

            # Process document
            documents = await self.document_processor.process_document(file_path)

            if not documents:
                print(f"⚠️ No content extracted from {filename}")
                return

            # Store in vector database
            chunks_created = await self.vector_store.store_documents(documents)

            print(f"✅ Processed {filename}: {len(documents)} chunks stored")

            # Clean up temporary file
            Path(file_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"❌ Failed to process document {filename}: {e}")
            # Clean up on error
            Path(file_path).unlink(missing_ok=True)

    async def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get document information.

        Args:
            file_path: Path to the document

        Returns:
            Document information dictionary
        """
        try:
            return await self.document_processor.get_document_info(file_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get document info: {str(e)}"
            )

    async def validate_document(self, file_path: str) -> bool:
        """
        Validate if document can be processed.

        Args:
            file_path: Path to the document

        Returns:
            True if document is valid
        """
        try:
            return await self.document_processor.validate_document(file_path)
        except Exception:
            return False

    async def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return self.document_processor.get_supported_formats()

    async def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.

        Returns:
            Vector store statistics
        """
        try:
            stats = {
                "is_initialized": self.vector_store.is_initialized(),
                "embedding_model": getattr(self.vector_store, 'embedding_model', 'unknown'),
            }

            if self.vector_store.is_initialized():
                doc_count = await self.vector_store.get_document_count()
                stats["document_count"] = doc_count

            return stats

        except Exception as e:
            return {
                "error": str(e),
                "is_initialized": False
            }


# Dependency injection helpers
from typing import Optional

_document_controller: Optional[DocumentController] = None


def set_document_controller(controller: DocumentController) -> None:
    """Set the global document controller instance"""
    global _document_controller
    _document_controller = controller


def get_document_controller() -> DocumentController:
    """Get the document controller dependency"""
    if _document_controller is None:
        raise HTTPException(
            status_code=503,
            detail="Document controller not initialized"
        )
    return _document_controller
