"""
Main FastAPI application with Clean Architecture.
This file replaces the old app.py and implements the new architecture.
"""

import time
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configuration
from config.settings import Settings, get_settings

# DTOs
from Application.dto import (
    MCQRequestDTO,
    BatchMCQRequestDTO,
    MCQResponseDTO,
    BatchMCQResponseDTO,
    HealthCheckResponseDTO,
    ErrorResponseDTO
)

# Domain
from Domain.services import MCQGenerationService, QualityValidationService, DifficultyAnalysisService
from Domain.entities import Document

# Infrastructure
from Infrastructure.llm import LLMFactory
from Infrastructure.vector_stores.faiss_store import FAISSVectorStore

# Use cases
from Application.use_cases.generate_mcq_use_case import GenerateMCQUseCase
from Application.use_cases.batch_generate_mcq_use_case import BatchGenerateMCQUseCase

# Global variables for dependency injection
llm_repository = None
vector_store_repository = None
mcq_generation_service = None
quality_validation_service = None
difficulty_analysis_service = None
generate_mcq_use_case = None
batch_generate_mcq_use_case = None
app_start_time = None


async def initialize_services(settings: Settings):
    """Initialize all services and repositories"""
    global llm_repository, vector_store_repository, mcq_generation_service
    global quality_validation_service, difficulty_analysis_service
    global generate_mcq_use_case, batch_generate_mcq_use_case, app_start_time

    print("🔧 Initializing RAG MCQ System with Clean Architecture...")
    app_start_time = time.time()

    try:
        # Initialize LLM Repository
        llm_config = {
            "api_key": settings.get_together_api_key(),
            "model": settings.llm.model,
            "max_retries": settings.llm.max_retries,
            "timeout": settings.llm.timeout
        }
        llm_repository = LLMFactory.create_llm(settings.llm.provider, llm_config)
        print("✅ LLM Repository initialized")

        # Initialize Vector Store Repository
        vector_store_repository = FAISSVectorStore(
            embedding_model=settings.embedding.model,
            chunk_size=settings.embedding.chunk_size,
            diversity_threshold=settings.embedding.diversity_threshold
        )
        print("✅ Vector Store Repository initialized")

        # Initialize Domain Services
        quality_validation_service = QualityValidationService()
        difficulty_analysis_service = DifficultyAnalysisService()

        mcq_generation_service = MCQGenerationService(
            llm_repository=llm_repository,
            vector_store_repository=vector_store_repository,
            quality_service=quality_validation_service,
            difficulty_service=difficulty_analysis_service
        )
        print("✅ Domain Services initialized")

        # Initialize Use Cases
        generate_mcq_use_case = GenerateMCQUseCase(mcq_generation_service)
        batch_generate_mcq_use_case = BatchGenerateMCQUseCase(mcq_generation_service)
        print("✅ Use Cases initialized")

        print("🎉 System initialization complete!")

    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        raise


async def cleanup_services():
    """Cleanup services on shutdown"""
    global llm_repository

    print("🧹 Cleaning up services...")

    if llm_repository and hasattr(llm_repository, 'close'):
        try:
            await llm_repository.close()
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")

    print("✅ Cleanup complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    settings = get_settings()
    await initialize_services(settings)
    yield
    await cleanup_services()


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()

    app = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        description="RAG-based Multiple Choice Question Generation API with Clean Architecture",
        docs_url=settings.api.docs_url,
        openapi_url=settings.api.openapi_url,
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# Dependency injection functions
def get_generate_mcq_use_case() -> GenerateMCQUseCase:
    """Get generate MCQ use case"""
    if generate_mcq_use_case is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return generate_mcq_use_case


def get_batch_generate_mcq_use_case() -> BatchGenerateMCQUseCase:
    """Get batch generate MCQ use case"""
    if batch_generate_mcq_use_case is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return batch_generate_mcq_use_case


def get_vector_store() -> FAISSVectorStore:
    """Get vector store repository"""
    if vector_store_repository is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return vector_store_repository


# API Routes

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "RAG MCQ Generation API with Clean Architecture",
        "version": get_settings().api.version,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheckResponseDTO)
async def health_check():
    """Health check endpoint"""
    settings = get_settings()
    current_time = time.time()
    uptime = current_time - app_start_time if app_start_time else None

    # Check service health
    services = {
        "llm_service": False,
        "vector_store": False,
        "document_processor": False  
    }

    try:
        if llm_repository:
            services["llm_service"] = await llm_repository.health_check()
    except Exception:
        services["llm_service"] = False

    try:
        if vector_store_repository:
            services["vector_store"] = vector_store_repository.is_initialized()
    except Exception:
        services["vector_store"] = False

    overall_status = "healthy" if all(services.values()) else "unhealthy"

    return HealthCheckResponseDTO(
        status=overall_status,
        timestamp=datetime.fromtimestamp(current_time),
        services=services,
        version=settings.api.version,
        uptime_seconds=uptime
    )


@app.post("/api/v1/mcq/gen", response_model=MCQResponseDTO)
async def generate_mcq(
    request: MCQRequestDTO,
    use_case: GenerateMCQUseCase = Depends(get_generate_mcq_use_case)
):
    """Generate a single MCQ question"""
    try:
        return await use_case.execute(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/mcq/gen-batch", response_model=BatchMCQResponseDTO)
async def generate_batch_mcq(
    request: BatchMCQRequestDTO,
    use_case: BatchGenerateMCQUseCase = Depends(get_batch_generate_mcq_use_case)
):
    """Generate multiple MCQ questions"""
    try:
        return await use_case.execute(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    vector_store: FAISSVectorStore = Depends(get_vector_store)
):
    """Upload and process a document"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Save uploaded file
        settings = get_settings()
        file_path = settings.tmp_dir / file.filename

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process document in background
        background_tasks.add_task(process_document_background, str(file_path), vector_store)

        return {
            "message": f"Document {file.filename} uploaded successfully",
            "filename": file.filename,
            "processing": "started in background"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_document_background(file_path: str, vector_store: FAISSVectorStore):
    """Process document in background"""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from pathlib import Path

        # Load PDF
        loader = PyPDFLoader(file_path)
        langchain_docs = loader.load()

        # Convert to domain documents
        documents = []
        for doc in langchain_docs:
            domain_doc = Document(
                content=doc.page_content,
                metadata=doc.metadata,
                source=str(Path(file_path).name)
            )
            documents.append(domain_doc)

        # Store in vector database
        chunks_created = await vector_store.store_documents(documents)
        print(f"✅ Processed {Path(file_path).name}: {len(documents)} pages, {chunks_created} chunks")

    except Exception as e:
        print(f"❌ Failed to process document {file_path}: {e}")


@app.get("/api/v1/system/info")
async def get_system_info():
    """Get system information"""
    settings = get_settings()

    info = {
        "system": "RAG MCQ Generation API",
        "version": settings.api.version,
        "architecture": "Clean Architecture",
        "environment": settings.environment,
        "components": {
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model,
            "embedding_model": settings.embedding.model,
            "vector_store": settings.vector_store.type
        }
    }

    if vector_store_repository:
        try:
            doc_count = await vector_store_repository.get_document_count()
            info["vector_store_documents"] = doc_count
        except Exception:
            info["vector_store_documents"] = "unknown"

    return info


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponseDTO(
            error="ValueError",
            message=str(exc),
            timestamp=datetime.fromtimestamp(time.time())
        ).dict()
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    """Handle RuntimeError exceptions"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponseDTO(
            error="RuntimeError",
            message=str(exc),
            timestamp=datetime.fromtimestamp(time.time())
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()

    uvicorn.run(
        "app_clean:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level=settings.logging.level.lower()
    )
