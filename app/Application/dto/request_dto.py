"""
Request DTOs for MCQ generation.
Data Transfer Objects for API requests.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from Domain.entities.enums import DifficultyLevel, QuestionType


class MCQRequestDTO(BaseModel):
    """DTO for single MCQ generation request"""
    topic: str = Field(..., min_length=1, max_length=200, description="Topic for the MCQ question")
    difficulty: DifficultyLevel = Field(DifficultyLevel.MEDIUM, description="Difficulty level")
    question_type: QuestionType = Field(QuestionType.DEFINITION, description="Type of question")
    context_query: Optional[str] = Field(None, max_length=500, description="Optional specific context query")

    @validator('topic')
    def topic_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Topic cannot be empty or whitespace only')
        return v.strip()

    @validator('context_query')
    def context_query_validator(cls, v):
        if v is not None and not v.strip():
            return None
        return v.strip() if v else None

    class Config:
        schema_extra = {
            "example": {
                "topic": "Machine Learning",
                "difficulty": "medium",
                "question_type": "definition",
                "context_query": "supervised learning algorithms"
            }
        }


class BatchMCQRequestDTO(BaseModel):
    """DTO for batch MCQ generation request"""
    topics: List[str] = Field(..., description="List of topics")
    questions_per_topic: int = Field(1, ge=1, le=10, description="Number of questions per topic")
    difficulties: Optional[List[DifficultyLevel]] = Field(
        None,
        description="List of difficulty levels to cycle through"
    )
    question_types: Optional[List[QuestionType]] = Field(
        None,
        description="List of question types to cycle through"
    )

    @validator('topics')
    def topics_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('Topics list cannot be empty')

        if len(v) > 20:
            raise ValueError('Topics list cannot exceed 20 items')

        cleaned_topics = []
        for topic in v:
            if not topic.strip():
                raise ValueError('Topics cannot be empty or whitespace only')
            cleaned_topics.append(topic.strip())

        return cleaned_topics

    @validator('difficulties')
    def difficulties_validator(cls, v):
        if v is not None and len(v) == 0:
            return None
        return v

    @validator('question_types')
    def question_types_validator(cls, v):
        if v is not None and len(v) == 0:
            return None
        return v

    class Config:
        schema_extra = {
            "example": {
                "topics": ["Machine Learning", "Deep Learning", "Neural Networks"],
                "questions_per_topic": 3,
                "difficulties": ["easy", "medium", "hard"],
                "question_types": ["definition", "application"]
            }
        }


class DocumentUploadRequestDTO(BaseModel):
    """DTO for document upload request"""
    file_path: str = Field(..., description="Path to the uploaded document")
    process_immediately: bool = Field(True, description="Whether to process the document immediately")
    chunk_size: Optional[int] = Field(500, ge=100, le=2000, description="Chunk size for document splitting")

    @validator('file_path')
    def file_path_validator(cls, v):
        if not v.strip():
            raise ValueError('File path cannot be empty')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "file_path": "/path/to/document.pdf",
                "process_immediately": True,
                "chunk_size": 500
            }
        }
