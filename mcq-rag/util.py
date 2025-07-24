from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from prompt import PromptTemplateManager

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


@dataclass
class MCQOption:
    """Data class for MCQ options"""
    label: str
    text: str
    is_correct: bool


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

class ContextAwareRetriever:
    """Enhanced retriever with context awareness and diversity"""

    def __init__(self, vector_db: FAISS, diversity_threshold: float = 0.7):
        self.vector_db = vector_db
        self.diversity_threshold = diversity_threshold

    def retrieve_diverse_contexts(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents with semantic diversity"""
        # Get more candidates than needed
        candidates = self.vector_db.similarity_search(query, k=k*2)

        if not candidates:
            return []

        # Select diverse documents
        selected = [candidates[0]]  # Always include the most relevant

        for candidate in candidates[1:]:
            if len(selected) >= k:
                break

            # Check diversity with already selected documents
            is_diverse = True
            for selected_doc in selected:
                similarity = self._calculate_similarity(candidate.page_content,
                                                        selected_doc.page_content)
                if similarity > self.diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(candidate)

        return selected[:k]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simplified implementation)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

def debug_prompt_templates():
    """Debug function to test prompt template generation"""
    print("🔍 Testing Prompt Templates:")
    prompt_manager = PromptTemplateManager()

    for question_type in QuestionType:
        try:
            template = prompt_manager.get_template(question_type)
            print(f"  {question_type.value}: ✅ Template loaded ({len(template)} chars)")
        except Exception as e:
            print(f"  {question_type.value}: ❌ Error - {e}")
