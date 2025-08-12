"""
System Information Controllers for Clean Architecture.
Handles system health, status, and configuration endpoints.
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import HTTPException

from Application.dto import HealthCheckResponseDTO
from config.settings import Settings


class SystemController:
    """Controller for system information endpoints"""

    def __init__(
        self,
        settings: Settings,
        start_time: Optional[float] = None
    ):
        """
        Initialize system controller.

        Args:
            settings: Application settings
            start_time: Application start time
        """
        self.settings = settings
        self.start_time = start_time or time.time()

    async def get_health_check(
        self,
        llm_service_healthy: bool = False,
        vector_store_healthy: bool = False
    ) -> HealthCheckResponseDTO:
        """
        Get system health status.

        Args:
            llm_service_healthy: LLM service health status
            vector_store_healthy: Vector store health status

        Returns:
            Health check response DTO
        """
        current_time = time.time()
        uptime = current_time - self.start_time

        services = {
            "llm_service": llm_service_healthy,
            "vector_store": vector_store_healthy,
            "document_processor": True  # Always true for now
        }

        overall_status = "healthy" if all(services.values()) else "unhealthy"

        return HealthCheckResponseDTO(
            status=overall_status,
            timestamp=datetime.fromtimestamp(current_time),
            services=services,
            version=self.settings.api.version,
            uptime_seconds=uptime
        )

    async def get_system_info(
        self,
        vector_store_doc_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive system information.

        Args:
            vector_store_doc_count: Number of documents in vector store

        Returns:
            System information dictionary
        """
        info = {
            "system": "RAG MCQ Generation API",
            "version": self.settings.api.version,
            "architecture": "Clean Architecture",
            "environment": self.settings.environment,
            "uptime_seconds": time.time() - self.start_time,
            "components": {
                "llm_provider": self.settings.llm.provider,
                "llm_model": self.settings.llm.model,
                "embedding_model": self.settings.embedding.model,
                "vector_store": self.settings.vector_store.type
            }
        }

        if vector_store_doc_count is not None:
            info["vector_store_documents"] = vector_store_doc_count
        else:
            info["vector_store_documents"] = "unknown"

        return info

    async def get_configuration(self) -> Dict[str, Any]:
        """
        Get non-sensitive configuration information.

        Returns:
            Configuration dictionary
        """
        return {
            "api": {
                "title": self.settings.api.title,
                "version": self.settings.api.version,
                "host": self.settings.api.host,
                "port": self.settings.api.port,
                "debug": self.settings.api.debug
            },
            "llm": {
                "provider": self.settings.llm.provider,
                "model": self.settings.llm.model,
                "max_retries": self.settings.llm.max_retries,
                "timeout": self.settings.llm.timeout
            },
            "embedding": {
                "model": self.settings.embedding.model,
                "chunk_size": self.settings.embedding.chunk_size,
                "diversity_threshold": self.settings.embedding.diversity_threshold
            },
            "vector_store": {
                "type": self.settings.vector_store.type,
                "similarity_threshold": self.settings.vector_store.similarity_threshold,
                "max_results": self.settings.vector_store.max_results
            },
            "quality": {
                "min_question_length": self.settings.quality.min_question_length,
                "max_question_length": self.settings.quality.max_question_length,
                "minimum_score": self.settings.quality.minimum_score
            },
            "environment": self.settings.environment
        }

    def get_uptime(self) -> float:
        """Get application uptime in seconds"""
        return time.time() - self.start_time

    def get_start_time(self) -> datetime:
        """Get application start time"""
        return datetime.fromtimestamp(self.start_time)


# Dependency injection helpers
_system_controller: Optional[SystemController] = None


def set_system_controller(controller: SystemController) -> None:
    """Set the global system controller instance"""
    global _system_controller
    _system_controller = controller


def get_system_controller() -> SystemController:
    """Get the system controller dependency"""
    if _system_controller is None:
        raise HTTPException(
            status_code=503,
            detail="System controller not initialized"
        )
    return _system_controller
