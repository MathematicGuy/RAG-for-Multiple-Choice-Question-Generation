"""
Quality validation service for MCQ questions.
Contains business logic for validating MCQ quality.
"""

from typing import List, Tuple
from Domain.entities.mcq_question import MCQQuestion


class QualityValidationService:
    """Service for validating MCQ quality"""

    def __init__(self):
        self.min_question_length = 10
        self.max_question_length = 200
        self.min_explanation_length = 20
        self.min_option_length = 3
        self.max_option_length = 150

    def validate_mcq(self, mcq: MCQQuestion) -> Tuple[bool, List[str]]:
        """
        Validate MCQ and return validation result with issues

        Args:
            mcq: MCQ question to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check question length
        if len(mcq.question) < self.min_question_length:
            issues.append(f"Question too short (minimum {self.min_question_length} characters)")
        elif len(mcq.question) > self.max_question_length:
            issues.append(f"Question too long (maximum {self.max_question_length} characters)")

        # Check options count
        if len(mcq.options) != 4:
            issues.append("Must have exactly 4 options")

        # Check for single correct answer
        correct_count = sum(1 for opt in mcq.options if opt.is_correct)
        if correct_count != 1:
            issues.append("Must have exactly one correct answer")

        # Check explanation
        if len(mcq.explanation) < self.min_explanation_length:
            issues.append(f"Explanation too short (minimum {self.min_explanation_length} characters)")

        # Check for distinct options
        option_texts = [opt.text for opt in mcq.options]
        if len(set(option_texts)) != len(option_texts):
            issues.append("Options must be distinct")

        # Check individual option lengths
        for i, option in enumerate(mcq.options):
            if len(option.text) < self.min_option_length:
                issues.append(f"Option {option.label} too short (minimum {self.min_option_length} characters)")
            elif len(option.text) > self.max_option_length:
                issues.append(f"Option {option.label} too long (maximum {self.max_option_length} characters)")

        # Check for empty or whitespace-only content
        if not mcq.question.strip():
            issues.append("Question cannot be empty or whitespace only")

        if not mcq.explanation.strip():
            issues.append("Explanation cannot be empty or whitespace only")

        for option in mcq.options:
            if not option.text.strip():
                issues.append(f"Option {option.label} cannot be empty or whitespace only")

        return len(issues) == 0, issues

    def calculate_quality_score(self, mcq: MCQQuestion) -> float:
        """
        Calculate quality score from 0 to 100

        Args:
            mcq: MCQ question to score

        Returns:
            Quality score between 0.0 and 100.0
        """
        is_valid, issues = self.validate_mcq(mcq)

        if not is_valid:
            # Return low score based on number of issues
            penalty = min(len(issues) * 10, 70)
            return max(0.0, 30.0 - penalty)

        # Start with base score for valid MCQ
        score = 70.0

        # Bonus for good explanation length
        if len(mcq.explanation) > 50:
            score += 10

        # Bonus for appropriate question length
        if 20 <= len(mcq.question) <= 100:
            score += 10

        # Bonus for diverse option lengths (indicates good distractors)
        option_lengths = [len(opt.text) for opt in mcq.options]
        length_variance = max(option_lengths) - min(option_lengths)
        if 10 <= length_variance <= 50:  # Good variance in option lengths
            score += 5

        # Bonus for balanced option distribution
        avg_length = sum(option_lengths) / len(option_lengths)
        if all(abs(length - avg_length) < avg_length * 0.5 for length in option_lengths):
            score += 5

        return min(score, 100.0)

    def get_quality_category(self, score: float) -> str:
        """
        Get quality category based on score

        Args:
            score: Quality score

        Returns:
            Quality category string
        """
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Very Poor"

    def suggest_improvements(self, mcq: MCQQuestion) -> List[str]:
        """
        Suggest improvements for the MCQ

        Args:
            mcq: MCQ question to analyze

        Returns:
            List of improvement suggestions
        """
        suggestions = []
        is_valid, issues = self.validate_mcq(mcq)

        # Address validation issues first
        suggestions.extend(issues)

        # Additional suggestions for improvement
        if len(mcq.question) < 30:
            suggestions.append("Consider making the question more detailed and specific")

        if len(mcq.explanation) < 50:
            suggestions.append("Provide a more comprehensive explanation")

        # Check for option balance
        option_lengths = [len(opt.text) for opt in mcq.options]
        if max(option_lengths) - min(option_lengths) > 100:
            suggestions.append("Balance the length of options to avoid obvious answers")

        return suggestions
