"""
Batch generate MCQ use case.
Contains the application logic for generating multiple MCQ questions.
"""

import time
import asyncio
from typing import List
from Domain.services.mcq_generation_service import MCQGenerationService
from Application.dto.request_dto import BatchMCQRequestDTO
from Application.dto.response_dto import BatchMCQResponseDTO


class BatchGenerateMCQUseCase:
    """Use case for generating multiple MCQ questions"""

    def __init__(self, mcq_service: MCQGenerationService):
        self.mcq_service = mcq_service

    async def execute(self, request: BatchMCQRequestDTO) -> BatchMCQResponseDTO:
        """
        Execute batch MCQ generation use case

        Args:
            request: Batch MCQ generation request

        Returns:
            Batch MCQ response DTO

        Raises:
            ValueError: If generation fails
            RuntimeError: If service is not available
        """
        start_time = time.time()

        try:
            # Validate request
            await self.validate_request(request)

            # Generate MCQs using the service
            mcqs = await self.mcq_service.generate_batch(
                topics=request.topics,
                questions_per_topic=request.questions_per_topic,
                difficulties=request.difficulties,
                question_types=request.question_types
            )

            # Calculate generation time
            generation_time = time.time() - start_time

            # Convert to response DTO
            return BatchMCQResponseDTO.from_entities(
                topics=request.topics,
                mcqs=mcqs,
                generation_time=generation_time
            )

        except ValueError as e:
            raise ValueError(f"Batch MCQ generation failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Service error during batch MCQ generation: {str(e)}")

    async def validate_request(self, request: BatchMCQRequestDTO) -> bool:
        """
        Validate the batch request before processing

        Args:
            request: Request to validate

        Returns:
            True if valid

        Raises:
            ValueError: If request is invalid
        """
        if not request.topics:
            raise ValueError("Topics list is required and cannot be empty")

        if len(request.topics) > 20:
            raise ValueError("Cannot process more than 20 topics at once")

        if request.questions_per_topic < 1 or request.questions_per_topic > 10:
            raise ValueError("Questions per topic must be between 1 and 10")

        total_questions = len(request.topics) * request.questions_per_topic
        if total_questions > 100:
            raise ValueError("Total questions cannot exceed 100")

        for topic in request.topics:
            if not topic or not topic.strip():
                raise ValueError("All topics must be non-empty")

        return True

    async def get_generation_estimate(self, request: BatchMCQRequestDTO) -> dict:
        """
        Get an estimate for batch generation

        Args:
            request: Batch request to estimate

        Returns:
            Dictionary with estimation details
        """
        total_questions = len(request.topics) * request.questions_per_topic
        estimated_time_per_question = 5  # seconds
        estimated_total_time = total_questions * estimated_time_per_question

        return {
            "total_questions": total_questions,
            "estimated_time_seconds": estimated_total_time,
            "estimated_time_minutes": estimated_total_time / 60,
            "topics_count": len(request.topics),
            "questions_per_topic": request.questions_per_topic
        }
