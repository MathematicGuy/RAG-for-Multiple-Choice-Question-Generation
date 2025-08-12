"""
Abstract repository interface for LLM operations.
Defines the contract for LLM service implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional
from Domain.entities.enums import DifficultyLevel, QuestionType


class LLMRepository(ABC):
    """Abstract repository for LLM operations"""

    @abstractmethod
    async def generate_mcq(
        self,
        context: str,
        topic: str,
        difficulty: DifficultyLevel,
        question_type: QuestionType
    ) -> str:
        """
        Generate MCQ using LLM

        Args:
            context: Relevant context for the question
            topic: The topic for the question
            difficulty: Difficulty level of the question
            question_type: Type of question to generate

        Returns:
            Raw LLM response as string

        Raises:
            LLMServiceError: If LLM service fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check LLM service health

        Returns:
            True if service is healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Get information about the current model

        Returns:
            Dictionary containing model information
        """
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for given text

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        pass
