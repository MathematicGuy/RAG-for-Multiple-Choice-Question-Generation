"""
DTOs package initialization.
"""

from .request_dto import MCQRequestDTO, BatchMCQRequestDTO, DocumentUploadRequestDTO
from .response_dto import (
    MCQResponseDTO,
    BatchMCQResponseDTO,
    DocumentProcessResponseDTO,
    HealthCheckResponseDTO,
    ErrorResponseDTO
)

__all__ = [
    # Request DTOs
    'MCQRequestDTO',
    'BatchMCQRequestDTO',
    'DocumentUploadRequestDTO',
    # Response DTOs
    'MCQResponseDTO',
    'BatchMCQResponseDTO',
    'DocumentProcessResponseDTO',
    'HealthCheckResponseDTO',
    'ErrorResponseDTO'
]
