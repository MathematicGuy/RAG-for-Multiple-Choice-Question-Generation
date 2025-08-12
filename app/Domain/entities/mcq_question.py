"""
MCQ Question domain entities.
Contains the core business entities for Multiple Choice Questions.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from Domain.entities.enums import QuestionType, DifficultyLevel


@dataclass
class MCQOption:
    """Data class for MCQ options"""
    label: str
    text: str
    is_correct: bool

    def __post_init__(self):
        """Validate the option after initialization"""
        if not self.label or not self.text:
            raise ValueError("Option label and text cannot be empty")
        if self.label not in ['A', 'B', 'C', 'D']:
            raise ValueError("Option label must be A, B, C, or D")


@dataclass
class MCQQuestion:
    """Data class for Multiple Choice Question"""
    question: str
    context: str
    options: List[MCQOption]
    explanation: str
    difficulty: str
    topic: str
    question_type: str
    source: str
    confidence_score: float = 0.0

    def __post_init__(self):
        """Validate the MCQ after initialization"""
        if not self.question or len(self.question.strip()) < 10:
            raise ValueError("Question must be at least 10 characters long")

        if len(self.options) != 4:
            raise ValueError("MCQ must have exactly 4 options")

        correct_count = sum(1 for opt in self.options if opt.is_correct)
        if correct_count != 1:
            raise ValueError("MCQ must have exactly one correct answer")

        if not self.explanation or len(self.explanation.strip()) < 20:
            raise ValueError("Explanation must be at least 20 characters long")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "question": self.question,
            "context": self.context,
            "options": {opt.label: opt.text for opt in self.options},
            "correct_answer": next(opt.label for opt in self.options if opt.is_correct),
            "explanation": self.explanation,
            "difficulty": self.difficulty,
            "topic": self.topic,
            "question_type": self.question_type,
            "source": self.source,
            "confidence_score": self.confidence_score
        }

    def get_correct_answer(self) -> str:
        """Get the label of the correct answer"""
        return next(opt.label for opt in self.options if opt.is_correct)

    def get_incorrect_options(self) -> List[MCQOption]:
        """Get all incorrect options (distractors)"""
        return [opt for opt in self.options if not opt.is_correct]

    def is_valid(self) -> bool:
        """Check if the MCQ is valid"""
        try:
            self.__post_init__()
            return True
        except ValueError:
            return False
