"""
Main FastAPI application with Clean Architecture using Controllers.
This is the final version that replaces app.py with proper separation of concerns.
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

# Controllers
from Presentation.controllers import (
    MCQController,
    DocumentController,
    SystemController,
    set_mcq_controller,
    set_document_controller,
    set_system_controller,
    get_mcq_controller,
    get_document_controller,
    get_system_controller
)

# Domain
from Domain.services import MCQGenerationService, QualityValidationService, DifficultyAnalysisService

# Infrastructure
from Infrastructure.llm import LLMFactory
from Infrastructure.vector_stores.faiss_store import FAISSVectorStore
from Infrastructure.document_processing import create_document_processor

# Use cases
from Application.use_cases.generate_mcq_use_case import GenerateMCQUseCase
from Application.use_cases.batch_generate_mcq_use_case import BatchGenerateMCQUseCase

# Global application start time
app_start_time = None


async def initialize_services(settings: Settings):
    """Initialize all services, repositories, and controllers"""
    global app_start_time

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

        # Initialize Document Processor
        document_processor = create_document_processor(
            chunk_size=settings.embedding.chunk_size,
            chunk_overlap=settings.embedding.chunk_overlap
        )
        print("✅ Document Processor initialized")

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

        # Initialize Controllers
        mcq_controller = MCQController(
            generate_use_case=generate_mcq_use_case,
            batch_generate_use_case=batch_generate_mcq_use_case
        )

        document_controller = DocumentController(
            document_processor=document_processor,
            vector_store=vector_store_repository,
            settings=settings
        )

        system_controller = SystemController(
            settings=settings,
            start_time=app_start_time
        )

        # Set global controllers for dependency injection
        set_mcq_controller(mcq_controller)
        set_document_controller(document_controller)
        set_system_controller(system_controller)

        print("✅ Controllers initialized")
        print("🎉 System initialization complete!")

        # Store references for cleanup
        return {
            "llm_repository": llm_repository,
            "vector_store_repository": vector_store_repository,
            "document_processor": document_processor
        }

    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        raise


async def cleanup_services(services: dict):
    """Cleanup services on shutdown"""
    print("🧹 Cleaning up services...")

    llm_repository = services.get("llm_repository")
    if llm_repository and hasattr(llm_repository, 'close'):
        try:
            await llm_repository.close()
        except Exception as e:
            print(f"⚠️ Error during LLM cleanup: {e}")

    print("✅ Cleanup complete")


# Global services reference for cleanup
_services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global _services
    settings = get_settings()
    _services = await initialize_services(settings)
    yield
    await cleanup_services(_services)


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


# API Routes

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "RAG MCQ Generation API with Clean Architecture",
        "version": get_settings().api.version,
        "docs": "/docs",
        "architecture": "Clean Architecture with Controllers"
    }


@app.get("/health", response_model=HealthCheckResponseDTO)
async def health_check(
    system_controller: SystemController = Depends(get_system_controller)
):
    """Health check endpoint"""
    try:
        # Check service health
        llm_healthy = False
        vector_store_healthy = False

        # Check LLM service
        llm_repository = _services.get("llm_repository")
        if llm_repository:
            try:
                llm_healthy = await llm_repository.health_check()
            except Exception:
                llm_healthy = False

        # Check vector store
        vector_store = _services.get("vector_store_repository")
        if vector_store:
            try:
                vector_store_healthy = vector_store.is_initialized()
            except Exception:
                vector_store_healthy = False

        return await system_controller.get_health_check(
            llm_service_healthy=llm_healthy,
            vector_store_healthy=vector_store_healthy
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# MCQ Generation Endpoints

@app.post("/api/v1/mcq/generate", response_model=MCQResponseDTO)
async def generate_mcq(
    request: MCQRequestDTO,
    mcq_controller: MCQController = Depends(get_mcq_controller)
):
    """Generate a single MCQ question"""
    return await mcq_controller.generate_single_mcq(request)


@app.post("/api/v1/mcq/generate-batch", response_model=BatchMCQResponseDTO)
async def generate_batch_mcq(
    request: BatchMCQRequestDTO,
    mcq_controller: MCQController = Depends(get_mcq_controller)
):
    """Generate multiple MCQ questions"""
    return await mcq_controller.generate_batch_mcq(request)


# Document Management Endpoints

@app.post("/api/v1/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_controller: DocumentController = Depends(get_document_controller)
):
    """Upload and process a document"""
    return await document_controller.upload_document(file, background_tasks)


@app.get("/api/v1/documents/formats")
async def get_supported_formats(
    document_controller: DocumentController = Depends(get_document_controller)
):
    """Get list of supported document formats"""
    formats = await document_controller.get_supported_formats()
    return {"supported_formats": formats}


@app.get("/api/v1/documents/stats")
async def get_document_stats(
    document_controller: DocumentController = Depends(get_document_controller)
):
    """Get document and vector store statistics"""
    return await document_controller.get_vector_store_stats()


# System Information Endpoints

@app.get("/api/v1/system/info")
async def get_system_info(
    system_controller: SystemController = Depends(get_system_controller)
):
    """Get comprehensive system information"""
    try:
        # Get vector store document count
        vector_store = _services.get("vector_store_repository")
        doc_count = None
        if vector_store:
            try:
                doc_count = await vector_store.get_document_count()
            except Exception:
                doc_count = None

        return await system_controller.get_system_info(
            vector_store_doc_count=doc_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")


@app.get("/api/v1/system/config")
async def get_system_config(
    system_controller: SystemController = Depends(get_system_controller)
):
    """Get non-sensitive system configuration"""
    return await system_controller.get_configuration()


@app.get("/api/v1/system/uptime")
async def get_system_uptime(
    system_controller: SystemController = Depends(get_system_controller)
):
    """Get system uptime information"""
    return {
        "uptime_seconds": system_controller.get_uptime(),
        "start_time": system_controller.get_start_time().isoformat(),
        "current_time": time.time()
    }


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

    print(f"🚀 Starting RAG MCQ API server...")
    print(f"   Environment: {settings.environment}")
    print(f"   LLM Provider: {settings.llm.provider}")
    print(f"   LLM Model: {settings.llm.model}")
    print(f"   Embedding Model: {settings.embedding.model}")
    print(f"   Host: {settings.api.host}:{settings.api.port}")
    print(f"   Docs: http://{settings.api.host}:{settings.api.port}/docs")

    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level=settings.logging.level.lower()
    )
