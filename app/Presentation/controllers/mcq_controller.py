"""
MCQ Generation Controllers for Clean Architecture.
Handles HTTP requests and responses for MCQ generation.
"""

from typing import List, Optional
from fastapi import HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse

from Application.dto import (
    MCQRequestDTO,
    BatchMCQRequestDTO,
    MCQResponseDTO,
    BatchMCQResponseDTO
)
from Application.use_cases import GenerateMCQUseCase, BatchGenerateMCQUseCase
from Domain.entities import MCQQuestion


class MCQController:
    """Controller for MCQ generation endpoints"""

    def __init__(
        self,
        generate_use_case: GenerateMCQUseCase,
        batch_generate_use_case: BatchGenerateMCQUseCase
    ):
        """
        Initialize MCQ controller.

        Args:
            generate_use_case: Single MCQ generation use case
            batch_generate_use_case: Batch MCQ generation use case
        """
        self.generate_use_case = generate_use_case
        self.batch_generate_use_case = batch_generate_use_case

    async def generate_single_mcq(self, request: MCQRequestDTO) -> MCQResponseDTO:
        """
        Generate a single MCQ question.

        Args:
            request: MCQ generation request

        Returns:
            MCQ response DTO

        Raises:
            HTTPException: For validation or processing errors
        """
        try:
            return await self.generate_use_case.execute(request)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    async def generate_batch_mcq(self, request: BatchMCQRequestDTO) -> BatchMCQResponseDTO:
        """
        Generate multiple MCQ questions.

        Args:
            request: Batch MCQ generation request

        Returns:
            Batch MCQ response DTO

        Raises:
            HTTPException: For validation or processing errors
        """
        try:
            return await self.batch_generate_use_case.execute(request)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    def validate_mcq_request(self, request: MCQRequestDTO) -> None:
        """
        Validate MCQ generation request.

        Args:
            request: Request to validate

        Raises:
            HTTPException: If validation fails
        """
        if not request.topic or len(request.topic.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Topic must be at least 3 characters long"
            )

        if request.context and len(request.context) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Context text is too long (max 10000 characters)"
            )

        if request.num_options and (request.num_options < 2 or request.num_options > 6):
            raise HTTPException(
                status_code=400,
                detail="Number of options must be between 2 and 6"
            )

    def validate_batch_request(self, request: BatchMCQRequestDTO) -> None:
        """
        Validate batch MCQ generation request.

        Args:
            request: Batch request to validate

        Raises:
            HTTPException: If validation fails
        """
        if request.count < 1 or request.count > 50:
            raise HTTPException(
                status_code=400,
                detail="Question count must be between 1 and 50"
            )

        # Validate the base MCQ request
        self.validate_mcq_request(request)


# Dependency injection helpers
_mcq_controller: Optional[MCQController] = None


def set_mcq_controller(controller: MCQController) -> None:
    """Set the global MCQ controller instance"""
    global _mcq_controller
    _mcq_controller = controller


def get_mcq_controller() -> MCQController:
    """Get the MCQ controller dependency"""
    if _mcq_controller is None:
        raise HTTPException(
            status_code=503,
            detail="MCQ controller not initialized"
        )
    return _mcq_controller
