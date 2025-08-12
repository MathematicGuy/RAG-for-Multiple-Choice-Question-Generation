# RAG MCQ System: Migration Plan to Together AI API with Clean Architecture

## Overview

This plan outlines the migration from local LLM models (HuggingFace Transformers) to Together AI API while implementing Clean Architecture using the Service-Controller-Repository Pattern for the backend logic.

## Current System Analysis

### Current Architecture
- **Monolithic Structure**: Single `EnhancedRAGMCQGenerator` class handles all responsibilities
- **Local LLM**: Uses HuggingFace Transformers with quantization (Qwen2.5-3B-Instruct)
- **Embedding**: Vietnamese-bi-encoder (remains unchanged)
- **Vector Store**: FAISS for similarity search
- **API Layer**: FastAPI with direct dependency injection

### Current Issues
- **Single Responsibility Violation**: One class handles PDF processing, embedding, LLM inference, and validation
- **Tight Coupling**: Direct dependency on HuggingFace models
- **Resource Intensive**: Local model requires GPU/CPU resources
- **Scalability Issues**: Limited by local hardware

## Target Architecture: Clean Architecture with Together AI

### 1. Domain Layer (Core Business Logic)
```
Domain/
├── entities/
│   ├── mcq_question.py          # MCQQuestion, MCQOption entities
│   ├── document.py              # Document entity
│   └── enums.py                 # QuestionType, DifficultyLevel enums
├── repositories/
│   ├── llm_repository.py        # Abstract LLM interface
│   ├── document_repository.py   # Abstract document interface
│   └── vector_store_repository.py # Abstract vector store interface
└── services/
    ├── mcq_generation_service.py # Core MCQ generation logic
    ├── quality_validation_service.py # Quality validation service
    └── difficulty_analysis_service.py # Difficulty analysis service
```

### 2. Infrastructure Layer (External Dependencies)
```
Infrastructure/
├── llm/
│   ├── together_ai_client.py    # Together AI API client
│   ├── huggingface_client.py    # Fallback local LLM (optional)
│   └── llm_factory.py           # Factory for LLM providers
├── vector_stores/
│   ├── faiss_store.py           # FAISS implementation
│   └── vector_store_factory.py  # Factory for vector stores
├── document_processing/
│   ├── pdf_processor.py         # PDF document processing
│   └── text_chunker.py          # Semantic chunking
└── embeddings/
    └── huggingface_embeddings.py # HuggingFace embeddings wrapper
```

### 3. Application Layer (Use Cases)
```
Application/
├── use_cases/
│   ├── generate_mcq_use_case.py # Single MCQ generation
│   ├── batch_generate_use_case.py # Batch MCQ generation
│   ├── process_documents_use_case.py # Document processing
│   └── build_vector_db_use_case.py # Vector database building
├── dto/
│   ├── mcq_request_dto.py       # Request DTOs
│   └── mcq_response_dto.py      # Response DTOs
└── interfaces/
    ├── llm_service_interface.py # LLM service interface
    └── document_service_interface.py # Document service interface
```

### 4. Presentation Layer (API Controllers)
```
Presentation/
├── controllers/
│   ├── mcq_controller.py        # MCQ generation endpoints
│   ├── document_controller.py   # Document upload endpoints
│   └── health_controller.py     # Health check endpoints
├── middleware/
│   ├── auth_middleware.py       # API authentication
│   ├── error_handler.py         # Global error handling
│   └── request_logger.py        # Request logging
└── validators/
    ├── mcq_request_validator.py # Request validation
    └── file_upload_validator.py # File upload validation
```

## Migration Strategy

### Phase 1: Refactor to Clean Architecture (Week 1-2)

#### Step 1.1: Extract Domain Entities
- Move `MCQQuestion`, `MCQOption` to `Domain/entities/`
- Create `Document` entity for document handling
- Move enums to `Domain/entities/enums.py`

#### Step 1.2: Create Repository Interfaces
```python
# Domain/repositories/llm_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from Domain.entities.mcq_question import MCQQuestion
from Domain.entities.enums import DifficultyLevel, QuestionType

class LLMRepository(ABC):
    @abstractmethod
    async def generate_mcq(
        self,
        context: str,
        topic: str,
        difficulty: DifficultyLevel,
        question_type: QuestionType
    ) -> str:
        """Generate MCQ using LLM"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check LLM service health"""
        pass
```

#### Step 1.3: Create Service Layer
```python
# Domain/services/mcq_generation_service.py
from Domain.repositories.llm_repository import LLMRepository
from Domain.repositories.vector_store_repository import VectorStoreRepository
from Domain.services.quality_validation_service import QualityValidationService

class MCQGenerationService:
    def __init__(
        self,
        llm_repository: LLMRepository,
        vector_store_repository: VectorStoreRepository,
        quality_service: QualityValidationService
    ):
        self.llm_repository = llm_repository
        self.vector_store_repository = vector_store_repository
        self.quality_service = quality_service

    async def generate_mcq(self, topic: str, difficulty: DifficultyLevel, question_type: QuestionType) -> MCQQuestion:
        # 1. Retrieve relevant context
        context = await self.vector_store_repository.search_similar(topic)

        # 2. Generate MCQ using LLM
        raw_response = await self.llm_repository.generate_mcq(context, topic, difficulty, question_type)

        # 3. Parse and validate
        mcq = self._parse_mcq_response(raw_response)
        quality_score = self.quality_service.calculate_quality_score(mcq)
        mcq.confidence_score = quality_score

        return mcq
```

### Phase 2: Implement Together AI Integration (Week 3)

#### Step 2.1: Create Together AI Client
```python
# Infrastructure/llm/together_ai_client.py
import aiohttp
from typing import Dict, Any
from Domain.repositories.llm_repository import LLMRepository

class TogetherAIClient(LLMRepository):
    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.together.xyz/v1"

    async def generate_mcq(self, context: str, topic: str, difficulty: DifficultyLevel, question_type: QuestionType) -> str:
        prompt = self._build_prompt(context, topic, difficulty, question_type)

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 800,
                "temperature": 0.7
            }

            async with session.post(f"{self.base_url}/completions", headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Together AI API error: {response.status}")

                result = await response.json()
                return result["choices"][0]["text"].strip()

    def _build_prompt(self, context: str, topic: str, difficulty: DifficultyLevel, question_type: QuestionType) -> str:
        return f"""
        Hãy tạo 1 câu hỏi trắc nghiệm dựa trên nội dung sau đây.
        Nội dung: {context}
        Chủ đề: {topic}
        Mức độ: {difficulty.value}
        Loại câu hỏi: {question_type.value}

        QUAN TRỌNG: Chỉ trả về JSON hợp lệ, không có text bổ sung:
        {{
            "question": "Câu hỏi rõ ràng về {topic}",
            "options": {{
                "A": "Đáp án A",
                "B": "Đáp án B",
                "C": "Đáp án C",
                "D": "Đáp án D"
            }},
            "correct_answer": "A",
            "explanation": "Giải thích tại sao đáp án A đúng"
        }}
        """
```

#### Step 2.2: Create LLM Factory
```python
# Infrastructure/llm/llm_factory.py
from typing import Dict, Any
from Domain.repositories.llm_repository import LLMRepository
from Infrastructure.llm.together_ai_client import TogetherAIClient
from Infrastructure.llm.huggingface_client import HuggingFaceClient

class LLMFactory:
    @staticmethod
    def create_llm(provider: str, config: Dict[str, Any]) -> LLMRepository:
        if provider == "together_ai":
            return TogetherAIClient(
                api_key=config["api_key"],
                model=config.get("model", "Qwen/Qwen2.5-7B-Instruct-Turbo")
            )
        elif provider == "huggingface":
            return HuggingFaceClient(
                model_name=config["model_name"],
                use_quantization=config.get("use_quantization", True)
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
```

### Phase 3: Update Application Layer (Week 4)

#### Step 3.1: Create Use Cases
```python
# Application/use_cases/generate_mcq_use_case.py
from Domain.services.mcq_generation_service import MCQGenerationService
from Application.dto.mcq_request_dto import MCQRequestDTO
from Application.dto.mcq_response_dto import MCQResponseDTO

class GenerateMCQUseCase:
    def __init__(self, mcq_service: MCQGenerationService):
        self.mcq_service = mcq_service

    async def execute(self, request: MCQRequestDTO) -> MCQResponseDTO:
        mcq = await self.mcq_service.generate_mcq(
            topic=request.topic,
            difficulty=request.difficulty,
            question_type=request.question_type
        )

        return MCQResponseDTO.from_entity(mcq)
```

#### Step 3.2: Create DTOs
```python
# Application/dto/mcq_request_dto.py
from pydantic import BaseModel
from Domain.entities.enums import DifficultyLevel, QuestionType

class MCQRequestDTO(BaseModel):
    topic: str
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    question_type: QuestionType = QuestionType.DEFINITION
    context_query: str = None

# Application/dto/mcq_response_dto.py
from pydantic import BaseModel
from typing import Dict
from Domain.entities.mcq_question import MCQQuestion

class MCQResponseDTO(BaseModel):
    question: str
    options: Dict[str, str]
    correct_answer: str
    explanation: str
    confidence_score: float
    topic: str
    difficulty: str
    question_type: str

    @classmethod
    def from_entity(cls, mcq: MCQQuestion) -> "MCQResponseDTO":
        return cls(
            question=mcq.question,
            options={opt.label: opt.text for opt in mcq.options},
            correct_answer=next(opt.label for opt in mcq.options if opt.is_correct),
            explanation=mcq.explanation,
            confidence_score=mcq.confidence_score,
            topic=mcq.topic,
            difficulty=mcq.difficulty,
            question_type=mcq.question_type
        )
```

### Phase 4: Update Presentation Layer (Week 5)

#### Step 4.1: Refactor Controllers
```python
# Presentation/controllers/mcq_controller.py
from fastapi import APIRouter, Depends, HTTPException
from Application.use_cases.generate_mcq_use_case import GenerateMCQUseCase
from Application.dto.mcq_request_dto import MCQRequestDTO
from Application.dto.mcq_response_dto import MCQResponseDTO

router = APIRouter(prefix="/api/v1/mcq", tags=["MCQ Generation"])

@router.post("/generate", response_model=MCQResponseDTO)
async def generate_mcq(
    request: MCQRequestDTO,
    use_case: GenerateMCQUseCase = Depends()
) -> MCQResponseDTO:
    try:
        return await use_case.execute(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-batch")
async def generate_batch_mcq(
    request: BatchMCQRequestDTO,
    use_case: BatchGenerateMCQUseCase = Depends()
):
    try:
        return await use_case.execute(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### Step 4.2: Dependency Injection Setup
```python
# main.py
from fastapi import FastAPI, Depends
from Infrastructure.llm.llm_factory import LLMFactory
from Domain.services.mcq_generation_service import MCQGenerationService
from Application.use_cases.generate_mcq_use_case import GenerateMCQUseCase

app = FastAPI()

# Configuration
config = {
    "llm_provider": "together_ai",
    "together_ai": {
        "api_key": "your_api_key",
        "model": "Qwen/Qwen2.5-7B-Instruct-Turbo"
    }
}

# Dependency injection
def get_llm_repository():
    return LLMFactory.create_llm(
        provider=config["llm_provider"],
        config=config[config["llm_provider"]]
    )

def get_mcq_service(llm_repo = Depends(get_llm_repository)):
    return MCQGenerationService(llm_repo, vector_store_repo, quality_service)

def get_generate_mcq_use_case(service = Depends(get_mcq_service)):
    return GenerateMCQUseCase(service)
```

## Key Benefits of the New Architecture

### 1. **Separation of Concerns**
- **Domain Layer**: Pure business logic, no external dependencies
- **Infrastructure Layer**: External API integrations, database access
- **Application Layer**: Use case orchestration
- **Presentation Layer**: HTTP handling, validation

### 2. **Testability**
- Easy unit testing with mock repositories
- Integration testing with test doubles
- Independent testing of each layer

### 3. **Flexibility**
- Easy to switch between Together AI and local models
- Support for multiple LLM providers
- Configurable without code changes

### 4. **Scalability**
- Async/await support for better performance
- Horizontal scaling with API-based LLM
- Better resource utilization

### 5. **Maintainability**
- Clear responsibility boundaries
- Easy to add new features
- Better error handling and logging

## Configuration Management

### Environment Configuration
```yaml
# config/production.yaml
llm:
  provider: "together_ai"
  together_ai:
    api_key: "${TOGETHER_AI_API_KEY}"
    model: "Qwen/Qwen2.5-7B-Instruct-Turbo"
    max_tokens: 800
    temperature: 0.7

embedding:
  model: "bkai-foundation-models/vietnamese-bi-encoder"

vector_store:
  type: "faiss"
  similarity_threshold: 0.3

quality:
  minimum_score: 60.0
  good_score: 75.0
  excellent_score: 90.0
```

## Error Handling Strategy

### 1. **Graceful Degradation**
- Fallback to local model if Together AI fails
- Retry mechanism with exponential backoff
- Circuit breaker pattern for API calls

### 2. **Comprehensive Logging**
- Request/response logging
- Performance metrics
- Error tracking and alerting

### 3. **Health Checks**
- LLM service health monitoring
- Database connectivity checks
- API endpoint health status

## Performance Considerations

### 1. **Caching Strategy**
- Response caching for similar requests
- Vector embedding caching
- Template caching

### 2. **Rate Limiting**
- Together AI API rate limits
- Request throttling
- Fair usage policies

### 3. **Monitoring**
- Response time tracking
- API usage metrics
- Resource utilization monitoring

## Migration Timeline

- **Week 1-2**: Domain and Repository setup
- **Week 3**: Together AI integration
- **Week 4**: Application layer implementation
- **Week 5**: Presentation layer updates
- **Week 6**: Testing and deployment

## Files to be Modified/Created

### New Files (35+ files)
```
Domain/
├── entities/ (3 files)
├── repositories/ (3 files)
└── services/ (3 files)

Infrastructure/
├── llm/ (3 files)
├── vector_stores/ (2 files)
├── document_processing/ (2 files)
└── embeddings/ (1 file)

Application/
├── use_cases/ (4 files)
├── dto/ (4 files)
└── interfaces/ (2 files)

Presentation/
├── controllers/ (3 files)
├── middleware/ (3 files)
└── validators/ (2 files)

Config/
├── settings.py
├── dependencies.py
└── environments/ (3 files)
```

### Modified Files
- `app.py` (main FastAPI application)
- `requirements.txt` (add new dependencies)
- `docker-compose.yaml` (update configuration)
- `README.md` (update documentation)

This migration plan ensures a smooth transition from local LLM models to Together AI API while implementing Clean Architecture principles that will make the system more maintainable, testable, and scalable.
