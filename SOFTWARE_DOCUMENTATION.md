# 📚 RAG MCQ System - Software Documentation

> **For IT Students**: This guide explains the Clean Architecture implementation of our RAG-based Multiple Choice Question generation system. Everything is explained step-by-step!

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Explanation](#architecture-explanation)
3. [Folder Structure Guide](#folder-structure-guide)
4. [File-by-File Breakdown](#file-by-file-breakdown)
5. [How to Extend the Code](#how-to-extend-the-code)
6. [Common Development Tasks](#common-development-tasks)
7. [Troubleshooting Guide](#troubleshooting-guide)

---

## 🎯 System Overview

### What Does This System Do?
Our system takes PDF documents, processes them, and generates multiple-choice questions (MCQs) based on the content. Think of it like an AI teacher that reads your textbook and creates quiz questions for you!

### Key Features:
- 📄 **Upload PDFs** → System reads and understands the content
- 🤖 **AI Question Generation** → Uses Together AI to create smart questions
- 🔍 **Smart Search** → Finds relevant content using vector search
- 📊 **Quality Control** → Ensures questions meet quality standards
- 🌐 **Web API** → Easy to use via HTTP requests

### Technology Stack:
- **Backend**: Python + FastAPI (like a smart waiter taking orders)
- **AI**: Together AI API (the brain that creates questions)
- **Search**: FAISS Vector Database (like Google for your documents)
- **Documents**: PDF processing (reads your files)

---

## 🏗️ Architecture Explanation

### What is Clean Architecture?
Imagine building a house with separate rooms for different purposes:
- **Living Room** (Presentation): Where guests interact
- **Kitchen** (Application): Where work gets done
- **Bedroom** (Domain): Your private business rules
- **Basement** (Infrastructure): All the utilities and connections

### Our 4 Layers Explained:

```
🌐 PRESENTATION LAYER (Controllers)
    ↓ "Handle user requests"
📋 APPLICATION LAYER (Use Cases & DTOs)
    ↓ "Organize business workflows"
🧠 DOMAIN LAYER (Business Logic)
    ↓ "Core business rules"
🔧 INFRASTRUCTURE LAYER (External Tools)
    ↓ "Talk to databases, APIs, files"
```

#### Layer 1: Presentation Layer 🌐
**Purpose**: Handle HTTP requests and responses
**Analogy**: Like a restaurant waiter who takes your order and brings your food
**Contains**: Controllers that handle web requests

#### Layer 2: Application Layer 📋
**Purpose**: Coordinate business operations
**Analogy**: Like a restaurant manager who organizes the kitchen workflow
**Contains**: Use cases (business workflows) and DTOs (data containers)

#### Layer 3: Domain Layer 🧠
**Purpose**: Core business logic and rules
**Analogy**: Like the chef's recipes and cooking techniques
**Contains**: Entities (data models), Services (business logic), Repository interfaces

#### Layer 4: Infrastructure Layer 🔧
**Purpose**: Handle external systems (databases, APIs, files)
**Analogy**: Like the restaurant's suppliers, payment systems, and utilities
**Contains**: Database connections, API clients, file processors

### Why This Architecture?
1. **Easy to Test**: Each layer can be tested separately
2. **Easy to Change**: Swap out databases or APIs without breaking everything
3. **Easy to Understand**: Clear separation of responsibilities
4. **Easy to Scale**: Add new features without messy code

---

## 📁 Folder Structure Guide

```
app/
├── 📋 Application/           # Application Layer - Workflows & Data Transfer
├── 🧠 Domain/               # Domain Layer - Business Rules
├── 🔧 Infrastructure/       # Infrastructure Layer - External Systems
├── 🌐 Presentation/         # Presentation Layer - Web Controllers
├── ⚙️ config/              # Configuration Settings
├── 📄 main.py              # Application Entry Point
├── 📦 requirements.txt     # Python Dependencies
└── 📚 Documentation Files
```

### Detailed Folder Breakdown:

### 📋 **Application/** - The Workflow Organizer
Think of this as your project manager who organizes tasks and handles paperwork.

```
Application/
├── dto/                    # Data Transfer Objects (DTOs)
│   ├── request_dto.py     # What users send to us
│   └── response_dto.py    # What we send back to users
└── use_cases/             # Business Workflows
    ├── generate_mcq_use_case.py      # Single question workflow
    └── batch_generate_mcq_use_case.py # Multiple questions workflow
```

**What are DTOs?**
- Like forms you fill out at the doctor's office
- They ensure data is properly formatted and validated
- Example: When user asks for a question, they fill out a "request form" (DTO)

**What are Use Cases?**
- Step-by-step procedures for completing tasks
- Like recipes: "First do this, then do that, finally return the result"
- Example: "Generate MCQ" use case coordinates all the steps to create a question

### 🧠 **Domain/** - The Brain Center
This is where all the smart business logic lives. No external dependencies!

```
Domain/
├── entities/              # Core Data Models
│   ├── mcq_question.py   # What a question looks like
│   ├── document.py       # What a document looks like
│   └── enums.py          # Predefined choices (like difficulty levels)
├── repositories/         # Interfaces for Data Storage
│   ├── llm_repository.py        # How to talk to AI
│   ├── vector_store_repository.py # How to search documents
│   └── document_repository.py   # How to handle documents
└── services/             # Business Logic
    ├── mcq_generation_service.py    # Smart question creation logic
    ├── quality_validation_service.py # Quality checking logic
    └── difficulty_analysis_service.py # Difficulty assessment logic
```

**Entities**: Like blueprints for objects
- MCQQuestion: Defines what makes a valid question
- Document: Defines what makes a valid document

**Repositories**: Like contracts/interfaces
- Define HOW to do things, but not the actual implementation
- Example: "There must be a way to generate text, but I don't care if it's OpenAI or Together AI"

**Services**: The smart business logic
- MCQGenerationService: "How to create good questions"
- QualityValidationService: "How to check if a question is good enough"

### 🔧 **Infrastructure/** - The Tools and Connections
This layer handles all the "dirty work" of connecting to external systems.

```
Infrastructure/
├── llm/                  # AI Language Model Connections
│   ├── together_ai_client.py    # Talk to Together AI API
│   └── llm_factory.py          # Choose which AI to use
├── vector_stores/        # Document Search System
│   └── faiss_store.py          # FAISS vector database
└── document_processing/  # File Processing
    └── pdf_processor.py        # Read and process PDFs
```

**LLM (Large Language Model)**:
- together_ai_client.py: Actually calls Together AI API
- llm_factory.py: Decides which AI service to use (like a factory that makes cars)

**Vector Stores**:
- faiss_store.py: Implements smart document search using embeddings
- Think of it like Google search for your documents

**Document Processing**:
- pdf_processor.py: Reads PDF files and breaks them into chunks

### 🌐 **Presentation/** - The Front Desk
This is where users interact with your system through web requests.

```
Presentation/
└── controllers/          # Web Request Handlers
    ├── mcq_controller.py       # Handle question generation requests
    ├── document_controller.py  # Handle file upload requests
    └── system_controller.py    # Handle system info requests
```

**Controllers**: Like receptionists at different desks
- MCQController: "I handle all question-related requests"
- DocumentController: "I handle all file-related requests"
- SystemController: "I handle system status and info requests"

### ⚙️ **config/** - The Settings Department
```
config/
└── settings.py           # All configuration in one place
```

**Settings**: Like a control panel for your application
- Database connections, API keys, default values
- Environment-specific settings (development vs production)

---

## 📄 File-by-File Breakdown

### 🎯 Core Application Files

#### **main.py** - The Application Starter
```python
# This is like the main switch that turns on your entire application
# It:
# 1. Sets up all the components
# 2. Connects them together
# 3. Starts the web server
# 4. Handles the application lifecycle
```

**What it does:**
- Initializes all services when app starts
- Sets up dependency injection (connecting components)
- Defines web routes (URLs)
- Handles application shutdown

**Key sections:**
- `initialize_services()`: Sets up all components
- `@app.get("/api/...")`: Defines web endpoints
- `lifespan()`: Manages app startup/shutdown

#### **requirements.txt** - The Shopping List
```text
# Lists all Python packages your app needs
# Like a shopping list for your development environment
```

### 📋 Application Layer Files

#### **dto/request_dto.py** - Input Forms
```python
# Defines what data users must provide
# Like forms at the DMV - specific fields required
class MCQRequestDTO:
    topic: str          # What topic to generate questions about
    context: str        # Background text to base questions on
    difficulty: str     # How hard should the questions be
    num_options: int    # How many answer choices
```

**Purpose**: Validate and structure incoming data

#### **dto/response_dto.py** - Output Packages
```python
# Defines what data we send back to users
# Like a package label - tells you what's inside
class MCQResponseDTO:
    question: str       # The generated question
    options: List[str]  # Answer choices
    correct_answer: str # The right answer
    metadata: dict      # Extra information
```

**Purpose**: Structure and validate outgoing data

#### **use_cases/generate_mcq_use_case.py** - Single Question Workflow
```python
# Step-by-step process to create one question
# Like a recipe for making a sandwich
async def execute(request: MCQRequestDTO) -> MCQResponseDTO:
    # 1. Validate the request
    # 2. Find relevant content
    # 3. Generate the question
    # 4. Check quality
    # 5. Return formatted response
```

#### **use_cases/batch_generate_mcq_use_case.py** - Multiple Questions Workflow
```python
# Process to create many questions at once
# Like a recipe for making a whole meal
async def execute(request: BatchMCQRequestDTO) -> BatchMCQResponseDTO:
    # 1. Validate the batch request
    # 2. Generate each question
    # 3. Collect all results
    # 4. Return batch response
```

### 🧠 Domain Layer Files

#### **entities/mcq_question.py** - Question Blueprint
```python
# Defines what makes a valid MCQ question
# Like a template for creating questions
class MCQQuestion:
    question_text: str      # The actual question
    options: List[MCQOption] # Answer choices
    correct_answer_index: int # Which option is correct
    difficulty: DifficultyLevel # How hard it is

    def validate(self) -> bool:
        # Check if question meets our standards
```

#### **entities/document.py** - Document Blueprint
```python
# Defines what makes a valid document
class Document:
    content: str        # The text content
    metadata: dict      # Information about the document
    source: str         # Where it came from
```

#### **entities/enums.py** - Predefined Choices
```python
# Like multiple choice options for the system itself
class QuestionType(Enum):
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
```

#### **repositories/llm_repository.py** - AI Interface Contract
```python
# Abstract contract for AI services
# Says "You must be able to generate text" but doesn't say how
class LLMRepository(ABC):
    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass
```

#### **services/mcq_generation_service.py** - Question Creation Logic
```python
# The smart logic for creating good questions
class MCQGenerationService:
    def __init__(self, llm_repo, vector_store_repo, quality_service):
        # Inject dependencies (like hiring specialists)

    async def generate_mcq(self, topic: str, context: str) -> MCQQuestion:
        # 1. Find relevant content using vector search
        # 2. Create a smart prompt for the AI
        # 3. Generate question using AI
        # 4. Validate quality
        # 5. Return polished question
```

### 🔧 Infrastructure Layer Files

#### **llm/together_ai_client.py** - AI API Communication
```python
# Actually talks to Together AI API
# Like a translator who speaks to foreign services
class TogetherAIClient(LLMRepository):
    async def generate_text(self, prompt: str) -> str:
        # 1. Format the request for Together AI
        # 2. Send HTTP request
        # 3. Handle response and errors
        # 4. Return generated text
```

#### **llm/llm_factory.py** - AI Service Selector
```python
# Chooses which AI service to use
# Like a factory that makes different types of cars
class LLMFactory:
    @staticmethod
    def create_llm(provider: str, config: dict) -> LLMRepository:
        if provider == "together_ai":
            return TogetherAIClient(config)
        elif provider == "openai":
            return OpenAIClient(config)  # Future implementation
        else:
            raise ValueError(f"Unknown provider: {provider}")
```

#### **vector_stores/faiss_store.py** - Document Search Engine
```python
# Implements smart document search
# Like Google search for your documents
class FAISSVectorStore(VectorStoreRepository):
    async def search_similar(self, query: str) -> List[Document]:
        # 1. Convert query to vector representation
        # 2. Find similar document chunks
        # 3. Return most relevant content

    async def store_documents(self, documents: List[Document]) -> int:
        # 1. Convert documents to vectors
        # 2. Store in FAISS index
        # 3. Return number of chunks created
```

#### **document_processing/pdf_processor.py** - File Reader
```python
# Reads and processes PDF files
# Like a librarian who organizes books
class DocumentProcessor:
    async def load_document(self, file_path: str) -> List[Document]:
        # 1. Read PDF file
        # 2. Extract text from each page
        # 3. Split into manageable chunks
        # 4. Return structured documents
```

### 🌐 Presentation Layer Files

#### **controllers/mcq_controller.py** - Question Request Handler
```python
# Handles web requests for question generation
# Like a waiter taking orders for questions
class MCQController:
    async def generate_single_mcq(self, request: MCQRequestDTO) -> MCQResponseDTO:
        # 1. Validate the request
        # 2. Call the use case
        # 3. Handle any errors
        # 4. Return formatted response
```

#### **controllers/document_controller.py** - File Upload Handler
```python
# Handles document upload and processing
# Like a clerk who handles document submissions
class DocumentController:
    async def upload_document(self, file: UploadFile) -> dict:
        # 1. Validate file type and size
        # 2. Save file temporarily
        # 3. Start background processing
        # 4. Return upload confirmation
```

#### **controllers/system_controller.py** - System Info Handler
```python
# Handles system status and information requests
# Like an information desk
class SystemController:
    async def get_health_check(self) -> HealthCheckResponseDTO:
        # 1. Check all system components
        # 2. Gather health status
        # 3. Return system status report
```

### ⚙️ Configuration Files

#### **config/settings.py** - Application Settings
```python
# Central place for all configuration
# Like a control panel for your app
class Settings:
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LLM settings
    llm_provider: str = "together_ai"
    llm_model: str = "meta-llama/Llama-2-7b-chat-hf"

    # Vector store settings
    embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder"
    chunk_size: int = 500
```

---

## 🛠️ How to Extend the Code

### 🎯 Common Extension Scenarios

#### 1. Adding a New AI Provider (e.g., OpenAI)

**Step 1: Create the Implementation**
```python
# Infrastructure/llm/openai_client.py
from Domain.repositories import LLMRepository
import openai

class OpenAIClient(LLMRepository):
    def __init__(self, config: dict):
        self.api_key = config["api_key"]
        self.model = config["model"]
        openai.api_key = self.api_key

    async def generate_text(self, prompt: str) -> str:
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def health_check(self) -> bool:
        try:
            # Test API connection
            await openai.Model.alist()
            return True
        except:
            return False
```

**Step 2: Update the Factory**
```python
# Infrastructure/llm/llm_factory.py
from .openai_client import OpenAIClient

class LLMFactory:
    @staticmethod
    def create_llm(provider: str, config: dict) -> LLMRepository:
        if provider == "together_ai":
            return TogetherAIClient(config)
        elif provider == "openai":           # <- Add this
            return OpenAIClient(config)      # <- Add this
        else:
            raise ValueError(f"Unknown provider: {provider}")
```

**Step 3: Update Configuration**
```python
# config/settings.py
class LLMSettings(BaseModel):
    provider: str = "together_ai"  # Can now be "openai" too
    model: str = "meta-llama/Llama-2-7b-chat-hf"
    openai_api_key: str = ""       # <- Add this
    # ... other settings
```

**That's it!** Now you can switch AI providers by changing the configuration.

#### 2. Adding a New Question Type (e.g., Fill-in-the-Blank)

**Step 1: Update Enums**
```python
# Domain/entities/enums.py
class QuestionType(Enum):
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_IN_BLANK = "fill_in_blank"  # <- Add this
```

**Step 2: Update Entity**
```python
# Domain/entities/mcq_question.py
class MCQQuestion(BaseModel):
    question_text: str
    options: Optional[List[MCQOption]] = None  # Make optional for fill-in-blank
    correct_answer: str  # For fill-in-blank, this is the answer text
    question_type: QuestionType
    # ... other fields

    def validate(self) -> bool:
        if self.question_type == QuestionType.FILL_IN_BLANK:
            # No options needed for fill-in-blank
            return len(self.question_text) > 10 and bool(self.correct_answer)
        else:
            # Existing validation for multiple choice
            return len(self.options) >= 2
```

**Step 3: Update Service Logic**
```python
# Domain/services/mcq_generation_service.py
async def generate_mcq(self, request: MCQRequestDTO) -> MCQQuestion:
    # ... existing code ...

    if request.question_type == QuestionType.FILL_IN_BLANK:
        prompt = self._create_fill_blank_prompt(context, topic)
        response = await self.llm_repository.generate_text(prompt)
        question = self._parse_fill_blank_response(response)
    else:
        # Existing multiple choice logic
        prompt = self._create_mcq_prompt(context, topic, request.num_options)
        # ... rest of existing code

    return question

def _create_fill_blank_prompt(self, context: str, topic: str) -> str:
    return f"""
    Based on this content about {topic}:
    {context}

    Create a fill-in-the-blank question. Format:
    Question: [Question with _____ for blank]
    Answer: [Correct answer]
    """
```

**Step 4: Update DTOs**
```python
# Application/dto/request_dto.py
class MCQRequestDTO(BaseModel):
    topic: str
    context: Optional[str] = None
    question_type: QuestionType = QuestionType.SINGLE_CHOICE
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    num_options: Optional[int] = 4  # Make optional since fill-in-blank doesn't need it
```

#### 3. Adding Document Validation Service

**Step 1: Create Domain Service**
```python
# Domain/services/document_validation_service.py
from Domain.entities import Document
from typing import List, Dict

class DocumentValidationService:
    def __init__(self):
        self.min_content_length = 100
        self.max_content_length = 50000

    async def validate_document(self, document: Document) -> bool:
        """Check if document is suitable for MCQ generation"""
        if not document.content or len(document.content.strip()) < self.min_content_length:
            return False

        if len(document.content) > self.max_content_length:
            return False

        # Check for educational content indicators
        educational_keywords = ['learn', 'study', 'concept', 'theory', 'definition']
        content_lower = document.content.lower()

        if not any(keyword in content_lower for keyword in educational_keywords):
            return False

        return True

    async def get_validation_report(self, document: Document) -> Dict[str, any]:
        """Get detailed validation report"""
        return {
            "is_valid": await self.validate_document(document),
            "content_length": len(document.content),
            "estimated_questions": len(document.content) // 500,  # Rough estimate
            "language_detected": self._detect_language(document.content),
            "readability_score": self._calculate_readability(document.content)
        }
```

**Step 2: Integrate into Use Case**
```python
# Application/use_cases/document_upload_use_case.py
class DocumentUploadUseCase:
    def __init__(
        self,
        document_processor: DocumentProcessor,
        validation_service: DocumentValidationService,
        vector_store: VectorStoreRepository
    ):
        self.document_processor = document_processor
        self.validation_service = validation_service
        self.vector_store = vector_store

    async def execute(self, file_path: str) -> Dict[str, any]:
        # 1. Load document
        documents = await self.document_processor.load_document(file_path)

        # 2. Validate each document chunk
        validation_results = []
        valid_documents = []

        for doc in documents:
            is_valid = await self.validation_service.validate_document(doc)
            validation_results.append(is_valid)
            if is_valid:
                valid_documents.append(doc)

        # 3. Store only valid documents
        if valid_documents:
            chunks_stored = await self.vector_store.store_documents(valid_documents)
        else:
            chunks_stored = 0

        return {
            "total_pages": len(documents),
            "valid_pages": len(valid_documents),
            "chunks_stored": chunks_stored,
            "validation_summary": {
                "passed": sum(validation_results),
                "failed": len(validation_results) - sum(validation_results)
            }
        }
```

#### 4. Adding Caching Layer

**Step 1: Create Cache Repository Interface**
```python
# Domain/repositories/cache_repository.py
from abc import ABC, abstractmethod
from typing import Optional, Any
import json

class CacheRepository(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        pass

    @abstractmethod
    async def set(self, key: str, value: str, expire_seconds: int = 3600) -> bool:
        """Set value with expiration"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
```

**Step 2: Create Redis Implementation**
```python
# Infrastructure/cache/redis_cache.py
import redis.asyncio as redis
from Domain.repositories import CacheRepository
import json

class RedisCache(CacheRepository):
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def get(self, key: str) -> Optional[str]:
        try:
            value = await self.redis.get(key)
            return value.decode() if value else None
        except Exception:
            return None

    async def set(self, key: str, value: str, expire_seconds: int = 3600) -> bool:
        try:
            await self.redis.set(key, value, ex=expire_seconds)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception:
            return False
```

**Step 3: Add Caching to MCQ Service**
```python
# Domain/services/mcq_generation_service.py
import hashlib
import json

class MCQGenerationService:
    def __init__(
        self,
        llm_repository: LLMRepository,
        vector_store_repository: VectorStoreRepository,
        quality_service: QualityValidationService,
        difficulty_service: DifficultyAnalysisService,
        cache_repository: Optional[CacheRepository] = None  # <- Add this
    ):
        # ... existing code ...
        self.cache_repository = cache_repository

    async def generate_mcq(self, request: MCQRequestDTO) -> MCQQuestion:
        # Create cache key based on request parameters
        cache_key = self._create_cache_key(request)

        # Try to get from cache first
        if self.cache_repository:
            cached_result = await self.cache_repository.get(cache_key)
            if cached_result:
                return MCQQuestion.parse_raw(cached_result)

        # Generate new question if not in cache
        question = await self._generate_new_mcq(request)

        # Store in cache
        if self.cache_repository and question:
            await self.cache_repository.set(
                cache_key,
                question.json(),
                expire_seconds=3600  # Cache for 1 hour
            )

        return question

    def _create_cache_key(self, request: MCQRequestDTO) -> str:
        """Create unique cache key for request"""
        request_data = {
            "topic": request.topic,
            "context": request.context[:100] if request.context else "",  # First 100 chars
            "question_type": request.question_type.value,
            "difficulty": request.difficulty.value,
            "num_options": request.num_options
        }
        request_json = json.dumps(request_data, sort_keys=True)
        return f"mcq:{hashlib.md5(request_json.encode()).hexdigest()}"
```

### 🔧 Extension Guidelines

#### **DO's:**
1. **Follow the Layer Rules:**
   - Domain layer should NOT import from other layers
   - Each layer only imports from layers below it
   - Use dependency injection for external dependencies

2. **Use Interfaces:**
   - Always create repository interfaces in Domain layer
   - Implement interfaces in Infrastructure layer
   - This makes testing and swapping implementations easy

3. **Add Tests:**
   ```python
   # tests/domain/test_mcq_generation_service.py
   async def test_generate_mcq():
       # Arrange: Set up mock dependencies
       mock_llm = MockLLMRepository()
       mock_vector_store = MockVectorStoreRepository()
       service = MCQGenerationService(mock_llm, mock_vector_store)

       # Act: Call the method
       result = await service.generate_mcq(request)

       # Assert: Check the result
       assert result.question_text is not None
       assert len(result.options) == 4
   ```

4. **Update Configuration:**
   - Add new settings to `config/settings.py`
   - Use environment variables for sensitive data
   - Provide sensible defaults

#### **DON'Ts:**
1. **Don't mix layers:**
   ```python
   # ❌ BAD: Domain importing from Infrastructure
   from Infrastructure.llm import TogetherAIClient  # Wrong!

   # ✅ GOOD: Domain using interface
   from Domain.repositories import LLMRepository  # Correct!
   ```

2. **Don't hardcode values:**
   ```python
   # ❌ BAD
   api_key = "sk-1234567890"  # Hardcoded

   # ✅ GOOD
   api_key = settings.llm.api_key  # From configuration
   ```

3. **Don't skip validation:**
   ```python
   # ❌ BAD
   async def generate_mcq(self, topic: str):
       # Directly use topic without validation

   # ✅ GOOD
   async def generate_mcq(self, request: MCQRequestDTO):
       # Request is already validated by DTO
   ```

### 🧪 Testing Your Extensions

#### **Unit Test Example:**
```python
# tests/test_mcq_service.py
import pytest
from unittest.mock import AsyncMock
from Domain.services import MCQGenerationService
from Domain.entities import MCQQuestion, QuestionType

@pytest.mark.asyncio
async def test_generate_mcq_success():
    # Arrange
    mock_llm = AsyncMock()
    mock_llm.generate_text.return_value = "Sample question with options..."

    mock_vector_store = AsyncMock()
    mock_vector_store.search_similar.return_value = [sample_document]

    service = MCQGenerationService(mock_llm, mock_vector_store, None, None)

    # Act
    result = await service.generate_mcq(sample_request)

    # Assert
    assert isinstance(result, MCQQuestion)
    assert result.question_type == QuestionType.SINGLE_CHOICE
    mock_llm.generate_text.assert_called_once()
```

#### **Integration Test Example:**
```python
# tests/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from main import app

def test_generate_mcq_endpoint():
    client = TestClient(app)

    response = client.post(
        "/api/v1/mcq/generate",
        json={
            "topic": "Python Programming",
            "context": "Python is a programming language...",
            "question_type": "single_choice",
            "difficulty": "medium"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "options" in data
    assert len(data["options"]) == 4
```

---

## 🔧 Common Development Tasks

### Task 1: Adding a New API Endpoint

**Example: Add endpoint to get question statistics**

1. **Add DTO (if needed):**
```python
# Application/dto/response_dto.py
class QuestionStatsResponseDTO(BaseModel):
    total_questions_generated: int
    questions_by_difficulty: Dict[str, int]
    questions_by_type: Dict[str, int]
    average_generation_time: float
```

2. **Add Use Case:**
```python
# Application/use_cases/get_question_stats_use_case.py
class GetQuestionStatsUseCase:
    def __init__(self, stats_repository: StatsRepository):
        self.stats_repository = stats_repository

    async def execute(self) -> QuestionStatsResponseDTO:
        stats = await self.stats_repository.get_question_stats()
        return QuestionStatsResponseDTO(**stats)
```

3. **Add Controller Method:**
```python
# Presentation/controllers/mcq_controller.py
async def get_question_stats(self) -> QuestionStatsResponseDTO:
    return await self.get_stats_use_case.execute()
```

4. **Add Route:**
```python
# main.py
@app.get("/api/v1/mcq/stats", response_model=QuestionStatsResponseDTO)
async def get_question_statistics(
    mcq_controller: MCQController = Depends(get_mcq_controller)
):
    """Get question generation statistics"""
    return await mcq_controller.get_question_stats()
```

### Task 2: Adding Database Persistence

1. **Create Entity:**
```python
# Domain/entities/question_history.py
class QuestionHistory(BaseModel):
    id: Optional[str] = None
    question_text: str
    topic: str
    difficulty: DifficultyLevel
    created_at: datetime
    user_id: Optional[str] = None
```

2. **Create Repository Interface:**
```python
# Domain/repositories/question_history_repository.py
class QuestionHistoryRepository(ABC):
    @abstractmethod
    async def save_question(self, question: QuestionHistory) -> str:
        pass

    @abstractmethod
    async def get_questions_by_user(self, user_id: str) -> List[QuestionHistory]:
        pass
```

3. **Implement Repository:**
```python
# Infrastructure/persistence/sqlite_question_history.py
import sqlite3
import aiosqlite

class SQLiteQuestionHistoryRepository(QuestionHistoryRepository):
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def save_question(self, question: QuestionHistory) -> str:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO questions (question_text, topic, difficulty, created_at, user_id) VALUES (?, ?, ?, ?, ?)",
                (question.question_text, question.topic, question.difficulty.value, question.created_at, question.user_id)
            )
            await db.commit()
            return str(cursor.lastrowid)
```

### Task 3: Adding Authentication

1. **Create User Entity:**
```python
# Domain/entities/user.py
class User(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool = True
    created_at: datetime
```

2. **Create Auth Service:**
```python
# Domain/services/auth_service.py
class AuthService:
    def __init__(self, user_repository: UserRepository, token_service: TokenService):
        self.user_repository = user_repository
        self.token_service = token_service

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        user = await self.user_repository.get_by_username(username)
        if user and self._verify_password(password, user.password_hash):
            return user
        return None

    async def create_token(self, user: User) -> str:
        return await self.token_service.create_access_token(user.id)
```

3. **Add Auth Middleware:**
```python
# Infrastructure/auth/jwt_auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    token = credentials.credentials
    user_id = verify_token(token)  # Implement token verification
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await user_repository.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user
```

4. **Protect Endpoints:**
```python
# main.py
@app.post("/api/v1/mcq/generate", response_model=MCQResponseDTO)
async def generate_mcq(
    request: MCQRequestDTO,
    current_user: User = Depends(get_current_user),  # Add this
    mcq_controller: MCQController = Depends(get_mcq_controller)
):
    """Generate MCQ (requires authentication)"""
    return await mcq_controller.generate_single_mcq(request)
```

---

## 🚨 Troubleshooting Guide

### Common Errors and Solutions

#### **1. Import Errors**
```
Error: ModuleNotFoundError: No module named 'Domain'
```

**Solution:**
```python
# Make sure you're running from the app directory
cd app
python main.py

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/app:$PYTHONPATH
```

#### **2. Dependency Injection Errors**
```
Error: Service not initialized
```

**Solution:**
Check that services are properly initialized in `main.py`:
```python
# Make sure this is called before app startup
await initialize_services(settings)
```

#### **3. Configuration Errors**
```
Error: Together AI API key not found
```

**Solution:**
```bash
# Set environment variable
export TOGETHER_API_KEY=your_api_key_here

# Or create token file
mkdir tokens
echo "your_api_key_here" > tokens/together_api_key.txt
```

#### **4. Database/Vector Store Errors**
```
Error: FAISS index not found
```

**Solution:**
```python
# Initialize vector store properly
vector_store = FAISSVectorStore(
    embedding_model="bkai-foundation-models/vietnamese-bi-encoder"
)
# Make sure to load documents first
await vector_store.store_documents(documents)
```

#### **5. Async/Await Errors**
```
Error: RuntimeError: This event loop is already running
```

**Solution:**
```python
# Don't mix sync and async code
# ❌ BAD
def sync_function():
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_function())

# ✅ GOOD
async def async_function():
    return await some_async_operation()
```

### Debug Tips

1. **Enable Debug Logging:**
```python
# config/settings.py
logging_level: str = "DEBUG"

# In your service
import logging
logger = logging.getLogger(__name__)

async def generate_mcq(self, request):
    logger.debug(f"Generating MCQ for topic: {request.topic}")
    # ... rest of code
```

2. **Use Health Checks:**
```bash
# Check if all services are working
curl http://localhost:8000/health

# Check system info
curl http://localhost:8000/api/v1/system/info
```

3. **Test Individual Components:**
```python
# Test LLM connection
async def test_llm():
    client = TogetherAIClient(config)
    response = await client.generate_text("Hello, world!")
    print(response)

# Test vector store
async def test_vector_store():
    store = FAISSVectorStore()
    results = await store.search_similar("machine learning")
    print(results)
```

---

## 📚 Learning Resources

### For Understanding Clean Architecture:
1. **Uncle Bob's Clean Architecture** (Book)
2. **FastAPI Documentation** (Official docs)
3. **Python Async Programming** (Real Python tutorials)

### For Extending This System:
1. **Domain-Driven Design** (Eric Evans)
2. **Test-Driven Development** (Kent Beck)
3. **API Design Patterns** (Microsoft docs)

### Python Libraries Used:
- **FastAPI**: Web framework
- **Pydantic**: Data validation
- **LangChain**: Document processing
- **FAISS**: Vector search
- **aiohttp**: Async HTTP client

---

## 🎓 Conclusion

This RAG MCQ system demonstrates Clean Architecture principles in a real-world application. Key takeaways:

1. **Separation of Concerns**: Each layer has a specific responsibility
2. **Dependency Inversion**: High-level modules don't depend on low-level details
3. **Testability**: Components can be tested in isolation
4. **Extensibility**: New features can be added without breaking existing code
5. **Maintainability**: Code is organized and easy to understand

Remember: Clean Architecture might seem complex at first, but it makes your code much easier to work with as your project grows. Start small, follow the patterns, and gradually build up your understanding!

**Happy coding! 🚀**