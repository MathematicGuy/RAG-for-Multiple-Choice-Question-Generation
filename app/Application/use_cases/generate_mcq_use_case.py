"""
Generate MCQ use case.
Contains the application logic for generating single MCQ questions.
"""

import time
from typing import Optional
from Domain.services.mcq_generation_service import MCQGenerationService
from Application.dto.request_dto import MCQRequestDTO
from Application.dto.response_dto import MCQResponseDTO


class GenerateMCQUseCase:
    """Use case for generating a single MCQ question"""

    def __init__(self, mcq_service: MCQGenerationService):
        self.mcq_service = mcq_service

    async def execute(self, request: MCQRequestDTO) -> MCQResponseDTO:
        """
        Execute MCQ generation use case

        Args:
            request: MCQ generation request

        Returns:
            MCQ response DTO

        Raises:
            ValueError: If generation fails
            RuntimeError: If service is not available
        """
        try:
            # Generate MCQ using the service
            mcq = await self.mcq_service.generate_mcq(
                topic=request.topic,
                difficulty=request.difficulty,
                question_type=request.question_type,
                context_query=request.context_query
            )

            # Convert to response DTO
            return MCQResponseDTO.from_entity(mcq)

        except ValueError as e:
            raise ValueError(f"MCQ generation failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Service error during MCQ generation: {str(e)}")

    async def validate_request(self, request: MCQRequestDTO) -> bool:
        """
        Validate the request before processing

        Args:
            request: Request to validate

        Returns:
            True if valid

        Raises:
            ValueError: If request is invalid
        """
        if not request.topic or not request.topic.strip():
            raise ValueError("Topic is required and cannot be empty")

        if request.context_query is not None and len(request.context_query) > 500:
            raise ValueError("Context query cannot exceed 500 characters")

        return True
