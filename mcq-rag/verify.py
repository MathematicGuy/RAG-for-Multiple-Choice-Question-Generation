from util import MCQQuestion, DifficultyLevel
from typing import Tuple, List

class QualityValidator:
    """Validates the quality of generated MCQ questions"""

    def __init__(self):
        self.min_question_length = 10
        self.max_question_length = 200
        self.min_explanation_length = 20

    def validate_mcq(self, mcq: MCQQuestion) -> Tuple[bool, List[str]]:
        """Validate MCQ and return validation result with issues"""
        issues = []

        # Check question length
        if len(mcq.question) < self.min_question_length:
            issues.append("Question too short")
        elif len(mcq.question) > self.max_question_length:
            issues.append("Question too long")

        # Check options count
        if len(mcq.options) != 4:
            issues.append("Must have exactly 4 options")

        # Check for single correct answer
        correct_count = sum(1 for opt in mcq.options if opt.is_correct)
        if correct_count != 1:
            issues.append("Must have exactly one correct answer")

        # Check explanation
        if len(mcq.explanation) < self.min_explanation_length:
            issues.append("Explanation too short")

        # Check for distinct options
        option_texts = [opt.text for opt in mcq.options]
        if len(set(option_texts)) != len(option_texts):
            issues.append("Options must be distinct")

        return len(issues) == 0, issues

    def calculate_quality_score(self, mcq: MCQQuestion) -> float:
        """Calculate quality score from 0 to 100"""
        is_valid, issues = self.validate_mcq(mcq)

        if not is_valid:
            return 0.0

        # Start with base score
        score = 70.0

        # Bonus for good explanation length
        if len(mcq.explanation) > 50:
            score += 10

        # Bonus for appropriate question length
        if 20 <= len(mcq.question) <= 100:
            score += 10

        # Bonus for diverse option lengths (indicates good distractors)
        option_lengths = [len(opt.text) for opt in mcq.options]
        if max(option_lengths) - min(option_lengths) < 50:  # Similar lengths
            score += 10

        return min(score, 100.0)


class DifficultyAnalyzer:
    """Analyzes and adjusts question difficulty"""

    def __init__(self):
        self.difficulty_keywords = {
            DifficultyLevel.EASY: ["là gì", "định nghĩa", "ví dụ", "đơn giản"],
            DifficultyLevel.MEDIUM: ["so sánh", "khác biệt", "ứng dụng", "khi nào"],
            DifficultyLevel.HARD: ["phân tích", "đánh giá", "tối ưu", "thiết kế"],
            DifficultyLevel.EXPERT: ["tổng hợp", "sáng tạo", "nghiên cứu", "phát triển"]
        }

    def assess_difficulty(self, question: str, context: str) -> DifficultyLevel:
        """Assess question difficulty based on content analysis"""
        question_lower = question.lower()

        # Count difficulty indicators
        difficulty_scores = {}
        for level, keywords in self.difficulty_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            difficulty_scores[level] = score

        # Return highest scoring difficulty
        if not any(difficulty_scores.values()):
            return DifficultyLevel.MEDIUM

        return max(difficulty_scores.keys(), key=lambda k: difficulty_scores[k])
