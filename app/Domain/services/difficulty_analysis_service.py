"""
Difficulty analysis service for MCQ questions.
Contains business logic for analyzing and adjusting question difficulty.
"""

from typing import Dict, List, Optional
from Domain.entities.enums import DifficultyLevel


class DifficultyAnalysisService:
    """Service for analyzing and adjusting question difficulty"""

    def __init__(self):
        self.difficulty_keywords = {
            DifficultyLevel.EASY: [
                "là gì", "định nghĩa", "ví dụ", "đơn giản", "cơ bản",
                "what is", "define", "example", "simple", "basic"
            ],
            DifficultyLevel.MEDIUM: [
                "so sánh", "khác biệt", "ứng dụng", "khi nào", "tại sao",
                "compare", "difference", "application", "when", "why"
            ],
            DifficultyLevel.HARD: [
                "phân tích", "đánh giá", "tối ưu", "thiết kế", "cách thức",
                "analyze", "evaluate", "optimize", "design", "how"
            ],
            DifficultyLevel.EXPERT: [
                "tổng hợp", "sáng tạo", "nghiên cứu", "phát triển", "suy luận",
                "synthesize", "create", "research", "develop", "infer"
            ]
        }

        self.complexity_indicators = {
            "high": ["algorithm", "implementation", "optimization", "architecture"],
            "medium": ["process", "method", "approach", "technique"],
            "low": ["definition", "example", "basic", "simple"]
        }

    def assess_difficulty(self, question: str, context: str = "") -> DifficultyLevel:
        """
        Assess question difficulty based on content analysis

        Args:
            question: The question text to analyze
            context: Additional context for analysis

        Returns:
            Estimated difficulty level
        """
        question_lower = question.lower()
        context_lower = context.lower() if context else ""
        combined_text = f"{question_lower} {context_lower}"

        # Count difficulty indicators
        difficulty_scores = {}
        for level, keywords in self.difficulty_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            difficulty_scores[level] = score

        # Add complexity analysis
        complexity_score = self._analyze_complexity(combined_text)

        # Adjust scores based on complexity
        if complexity_score >= 3:
            difficulty_scores[DifficultyLevel.EXPERT] += 2
            difficulty_scores[DifficultyLevel.HARD] += 1
        elif complexity_score >= 2:
            difficulty_scores[DifficultyLevel.HARD] += 1
            difficulty_scores[DifficultyLevel.MEDIUM] += 1
        else:
            difficulty_scores[DifficultyLevel.EASY] += 1

        # Return highest scoring difficulty
        if not any(difficulty_scores.values()):
            return DifficultyLevel.MEDIUM

        return max(difficulty_scores.keys(), key=lambda k: difficulty_scores[k])

    def _analyze_complexity(self, text: str) -> int:
        """
        Analyze text complexity based on various indicators

        Args:
            text: Text to analyze

        Returns:
            Complexity score (0-5)
        """
        complexity_score = 0

        # Check for high complexity indicators
        for indicator in self.complexity_indicators["high"]:
            if indicator in text:
                complexity_score += 2

        # Check for medium complexity indicators
        for indicator in self.complexity_indicators["medium"]:
            if indicator in text:
                complexity_score += 1

        # Analyze sentence structure
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

        if avg_sentence_length > 20:
            complexity_score += 1
        elif avg_sentence_length > 15:
            complexity_score += 0.5

        # Check for technical terms (words longer than 8 characters)
        words = text.split()
        technical_words = [w for w in words if len(w) > 8]
        technical_ratio = len(technical_words) / max(len(words), 1)

        if technical_ratio > 0.3:
            complexity_score += 2
        elif technical_ratio > 0.2:
            complexity_score += 1

        return min(int(complexity_score), 5)

    def suggest_difficulty_adjustments(
        self,
        current_question: str,
        target_difficulty: DifficultyLevel,
        current_difficulty: Optional[DifficultyLevel] = None
    ) -> List[str]:
        """
        Suggest adjustments to reach target difficulty

        Args:
            current_question: Current question text
            target_difficulty: Desired difficulty level
            current_difficulty: Current assessed difficulty (optional)

        Returns:
            List of adjustment suggestions
        """
        if current_difficulty is None:
            current_difficulty = self.assess_difficulty(current_question)

        suggestions = []

        if current_difficulty == target_difficulty:
            suggestions.append("Question difficulty is already at target level")
            return suggestions

        if target_difficulty.value == "easy":
            suggestions.extend([
                "Use simpler vocabulary and shorter sentences",
                "Focus on basic definitions and examples",
                "Ask for direct recall of information",
                "Use familiar contexts and scenarios"
            ])
        elif target_difficulty.value == "medium":
            suggestions.extend([
                "Include comparison or application questions",
                "Ask about relationships between concepts",
                "Require some analysis or interpretation",
                "Use moderate technical vocabulary"
            ])
        elif target_difficulty.value == "hard":
            suggestions.extend([
                "Require analysis and evaluation",
                "Include complex scenarios or problems",
                "Ask for optimization or design decisions",
                "Use advanced technical concepts"
            ])
        elif target_difficulty.value == "expert":
            suggestions.extend([
                "Require synthesis of multiple concepts",
                "Include research or creative thinking",
                "Ask for novel solutions or approaches",
                "Use cutting-edge or specialized knowledge"
            ])

        return suggestions

    def get_difficulty_distribution_recommendation(self, total_questions: int) -> Dict[DifficultyLevel, int]:
        """
        Get recommended difficulty distribution for a given number of questions

        Args:
            total_questions: Total number of questions to generate

        Returns:
            Dictionary with recommended count for each difficulty level
        """
        # Standard distribution: 30% Easy, 40% Medium, 25% Hard, 5% Expert
        distribution = {
            DifficultyLevel.EASY: max(1, int(total_questions * 0.30)),
            DifficultyLevel.MEDIUM: max(1, int(total_questions * 0.40)),
            DifficultyLevel.HARD: max(1, int(total_questions * 0.25)),
            DifficultyLevel.EXPERT: max(0, int(total_questions * 0.05))
        }

        # Adjust for small numbers
        if total_questions <= 4:
            distribution = {
                DifficultyLevel.EASY: 1,
                DifficultyLevel.MEDIUM: max(1, total_questions - 2),
                DifficultyLevel.HARD: 1 if total_questions > 2 else 0,
                DifficultyLevel.EXPERT: 0
            }

        # Ensure total matches
        current_total = sum(distribution.values())
        if current_total < total_questions:
            distribution[DifficultyLevel.MEDIUM] += (total_questions - current_total)
        elif current_total > total_questions:
            # Reduce from expert first, then hard, then easy
            excess = current_total - total_questions
            for level in [DifficultyLevel.EXPERT, DifficultyLevel.HARD, DifficultyLevel.EASY]:
                if excess <= 0:
                    break
                reduction = min(excess, distribution[level])
                distribution[level] -= reduction
                excess -= reduction

        return distribution
