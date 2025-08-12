"""
Domain entities for the RAG MCQ system.
This module contains the core business entities that represent the fundamental concepts.
"""

from enum import Enum


class QuestionType(Enum):
    """Enumeration of different question types"""
    DEFINITION = "definition"
    COMPARISON = "comparison"
    APPLICATION = "application"
    ANALYSIS = "analysis"
    EVALUATION = "evaluation"


class DifficultyLevel(Enum):
    """Enumeration of difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
