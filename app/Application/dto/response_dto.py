"""
Response DTOs for MCQ generation.
Data Transfer Objects for API responses.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime


class MCQResponseDTO(BaseModel):
    """DTO for single MCQ response"""
    question: str
    options: Dict[str, str]
    correct_answer: str
    explanation: str
    confidence_score: float
    topic: str
    difficulty: str
    question_type: str
    source: Optional[str] = None

    @classmethod
    def from_entity(cls, mcq) -> "MCQResponseDTO":
        """Convert from domain entity to DTO"""
        return cls(
            question=mcq.question,
            options={opt.label: opt.text for opt in mcq.options},
            correct_answer=next(opt.label for opt in mcq.options if opt.is_correct),
            explanation=mcq.explanation,
            confidence_score=mcq.confidence_score,
            topic=mcq.topic,
            difficulty=mcq.difficulty,
            question_type=mcq.question_type,
            source=mcq.source
        )

    class Config:
        schema_extra = {
            "example": {
                "question": "What is machine learning?",
                "options": {
                    "A": "A type of artificial intelligence",
                    "B": "A programming language",
                    "C": "A database system",
                    "D": "A web framework"
                },
                "correct_answer": "A",
                "explanation": "Machine learning is a subset of artificial intelligence...",
                "confidence_score": 85.5,
                "topic": "Machine Learning",
                "difficulty": "medium",
                "question_type": "definition",
                "source": "ml_textbook.pdf"
            }
        }


class BatchMCQResponseDTO(BaseModel):
    """DTO for batch MCQ response"""
    topics: List[str]
    questions: List[MCQResponseDTO]
    total_questions: int
    successful_generations: int
    failed_generations: int
    average_confidence: float
    generation_time_seconds: float
    timestamp: datetime

    @classmethod
    def from_entities(cls, topics: List[str], mcqs: List, generation_time: float) -> "BatchMCQResponseDTO":
        """Convert from domain entities to DTO"""
        questions = [MCQResponseDTO.from_entity(mcq) for mcq in mcqs]
        avg_confidence = sum(mcq.confidence_score for mcq in mcqs) / len(mcqs) if mcqs else 0.0

        return cls(
            topics=topics,
            questions=questions,
            total_questions=len(topics) * (len(mcqs) // len(topics) if topics else 0),
            successful_generations=len(mcqs),
            failed_generations=max(0, len(topics) - len(mcqs)),
            average_confidence=avg_confidence,
            generation_time_seconds=generation_time,
            timestamp=datetime.utcnow()
        )

    class Config:
        schema_extra = {
            "example": {
                "topics": ["Machine Learning", "Deep Learning"],
                "questions": [],  # Would contain MCQResponseDTO examples
                "total_questions": 6,
                "successful_generations": 5,
                "failed_generations": 1,
                "average_confidence": 82.3,
                "generation_time_seconds": 45.2,
                "timestamp": "2025-08-11T10:30:00Z"
            }
        }


class DocumentProcessResponseDTO(BaseModel):
    """DTO for document processing response"""
    filename: str
    total_pages: int
    chunks_created: int
    processing_time_seconds: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime

    class Config:
        schema_extra = {
            "example": {
                "filename": "textbook.pdf",
                "total_pages": 150,
                "chunks_created": 300,
                "processing_time_seconds": 12.5,
                "success": True,
                "error_message": None,
                "timestamp": "2025-08-11T10:30:00Z"
            }
        }


class HealthCheckResponseDTO(BaseModel):
    """DTO for health check response"""
    status: str
    timestamp: datetime
    services: Dict[str, bool]
    version: str
    uptime_seconds: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-08-11T10:30:00Z",
                "services": {
                    "llm_service": True,
                    "vector_store": True,
                    "document_processor": True
                },
                "version": "1.0.0",
                "uptime_seconds": 3600.0
            }
        }


class ErrorResponseDTO(BaseModel):
    """DTO for error responses"""
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None
    details: Optional[Dict] = None

    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Topic cannot be empty",
                "timestamp": "2025-08-11T10:30:00Z",
                "request_id": "req_123456",
                "details": {
                    "field": "topic",
                    "value": ""
                }
            }
        }
