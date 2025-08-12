# RAG MCQ System Migration - Clean Architecture Implementation

## 🎯 Migration Overview

This migration transforms the RAG MCQ system from a monolithic architecture using local LLM models to a Clean Architecture implementation using Together AI API. The system maintains the same embedding models while improving code organization, testability, and maintainability.

## 🏗️ Architecture

### Clean Architecture Layers

```
┌─ Presentation Layer (Controllers)
│  ├─ MCQController
│  ├─ DocumentController
│  └─ SystemController
│
├─ Application Layer (Use Cases & DTOs)
│  ├─ Use Cases: GenerateMCQUseCase, BatchGenerateMCQUseCase
│  └─ DTOs: Request/Response models
│
├─ Domain Layer (Business Logic)
│  ├─ Entities: MCQQuestion, MCQOption, Document
│  ├─ Services: MCQGenerationService, QualityValidationService
│  └─ Repository Interfaces
│
└─ Infrastructure Layer (External Dependencies)
   ├─ LLM: Together AI Client
   ├─ Vector Stores: FAISS Implementation
   └─ Document Processing: PDF Processor
```

## 🚀 New Features

### 1. Together AI Integration
- **Async HTTP Client**: Fast, non-blocking API calls
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Handling**: Comprehensive error handling and validation
- **Health Checks**: Monitor API connectivity

### 2. Clean Architecture Benefits
- **Separation of Concerns**: Each layer has a single responsibility
- **Dependency Inversion**: High-level modules don't depend on low-level modules
- **Testability**: Easy to unit test with dependency injection
- **Maintainability**: Clear structure and interfaces

### 3. Enhanced Monitoring
- **Health Endpoints**: System status and component health
- **Structured Logging**: JSON-formatted logs with context
- **Performance Metrics**: Response times and success rates
- **Configuration Management**: Environment-based settings

## 📁 Project Structure

```
api-cloud/
├─ main.py                          # New main application (replaces app.py)
├─ config/
│  └─ settings.py                   # Centralized configuration
├─ Domain/
│  ├─ entities/                     # Business entities
│  ├─ repositories/                 # Repository interfaces
│  └─ services/                     # Business logic services
├─ Application/
│  ├─ dto/                         # Data transfer objects
│  └─ use_cases/                   # Application use cases
├─ Infrastructure/
│  ├─ llm/                         # LLM implementations
│  ├─ vector_stores/               # Vector store implementations
│  └─ document_processing/         # Document processing
├─ Presentation/
│  └─ controllers/                 # API controllers
└─ requirements.txt                # Updated dependencies
```

## 🔧 Configuration

### Environment Variables

```bash
# Together AI Configuration
TOGETHER_API_KEY=your_api_key_here
LLM_MODEL=meta-llama/Llama-2-7b-chat-hf

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Embedding Configuration
EMBEDDING_MODEL=bkai-foundation-models/vietnamese-bi-encoder
EMBEDDING_CHUNK_SIZE=500
EMBEDDING_CHUNK_OVERLAP=50

# Vector Store Configuration
VECTOR_STORE_TYPE=faiss
VECTOR_STORE_SIMILARITY_THRESHOLD=0.3
VECTOR_STORE_MAX_RESULTS=5

# Quality Settings
QUALITY_MINIMUM_SCORE=60.0
QUALITY_GOOD_SCORE=75.0
QUALITY_EXCELLENT_SCORE=90.0

# Logging Configuration
LOGGING_LEVEL=INFO
LOGGING_FORMAT=json
```

### Token Files (Alternative)
If environment variables are not available, place API keys in:
- `tokens/together_api_key.txt`
- `tokens/langsmith_api_key.txt`

## 🚀 Running the New System

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy and modify environment file
cp env.example .env
# Edit .env with your configuration
```

### 3. Start the Server
```bash
# Development mode
python main.py

# Production mode with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Access Documentation
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **System Info**: http://localhost:8000/api/v1/system/info

## 📋 API Endpoints

### MCQ Generation
- `POST /api/v1/mcq/generate` - Generate single MCQ
- `POST /api/v1/mcq/generate-batch` - Generate multiple MCQs

### Document Management
- `POST /api/v1/documents/upload` - Upload and process documents
- `GET /api/v1/documents/formats` - Get supported formats
- `GET /api/v1/documents/stats` - Get document statistics

### System Information
- `GET /health` - Health check
- `GET /api/v1/system/info` - System information
- `GET /api/v1/system/config` - Configuration details
- `GET /api/v1/system/uptime` - Uptime information

## 🔄 Migration from Old System

### Key Changes

1. **LLM Provider**:
   - **Old**: Local Transformers with quantization
   - **New**: Together AI API with async HTTP calls

2. **Architecture**:
   - **Old**: Monolithic `EnhancedRAGMCQGenerator` class
   - **New**: Clean Architecture with separate layers

3. **Dependencies**:
   - **Added**: `aiohttp`, `dependency-injector`, `structlog`, `tenacity`
   - **Maintained**: Embedding models and vector store (FAISS)

4. **Configuration**:
   - **Old**: Hardcoded settings in classes
   - **New**: Environment-based configuration management

### Compatibility

- **API Endpoints**: Maintained backward compatibility
- **Request/Response Format**: Same JSON structure
- **Embedding Models**: No changes to Vietnamese bi-encoder
- **Vector Store**: FAISS remains the same

## 🧪 Testing the Migration

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Generate MCQ
```bash
curl -X POST "http://localhost:8000/api/v1/mcq/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Machine Learning",
    "context": "Machine learning is a subset of artificial intelligence...",
    "question_type": "single_choice",
    "difficulty": "medium",
    "num_options": 4
  }'
```

### 3. Upload Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf"
```

## 🔍 Monitoring and Debugging

### Logs
The system uses structured logging with JSON format:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "mcq_generation",
  "message": "MCQ generated successfully",
  "topic": "Machine Learning",
  "question_type": "single_choice",
  "processing_time": 2.5
}
```

### Health Monitoring
Monitor system health via `/health` endpoint:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "llm_service": true,
    "vector_store": true,
    "document_processor": true
  },
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Together AI API Key**:
   - Ensure `TOGETHER_API_KEY` is set correctly
   - Check token file exists: `tokens/together_api_key.txt`

2. **LLM Connection Errors**:
   - Check internet connectivity
   - Verify API key permissions
   - Monitor rate limits

3. **Vector Store Issues**:
   - Ensure embedding model is downloaded
   - Check FAISS installation
   - Verify document processing

4. **Import Errors**:
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

### Debug Mode
Enable debug logging:
```bash
export LOGGING_LEVEL=DEBUG
export API_DEBUG=true
python main.py
```

## 📈 Performance Improvements

### Together AI vs Local LLM
- **Startup Time**: Faster (no model loading)
- **Memory Usage**: Lower (no local model storage)
- **Scalability**: Better (API handles scaling)
- **Response Time**: Comparable or faster
- **Cost**: Pay-per-use model

### Architecture Benefits
- **Code Organization**: Better separation of concerns
- **Testing**: Easier unit and integration testing
- **Maintenance**: Cleaner dependencies and interfaces
- **Extensibility**: Easy to add new LLM providers or features

## 🔮 Future Enhancements

1. **Multiple LLM Providers**: Support for OpenAI, Anthropic, etc.
2. **Caching Layer**: Redis for frequently accessed content
3. **Async Document Processing**: Background job queue
4. **Advanced Monitoring**: Prometheus metrics and Grafana dashboards
5. **A/B Testing**: Compare different LLM models
6. **Rate Limiting**: API rate limiting and quotas

## 📝 Development Notes

### Adding New LLM Providers
1. Implement `LLMRepository` interface
2. Add to `LLMFactory`
3. Update configuration settings
4. Add provider-specific tests

### Adding New Features
1. Define in Domain layer (entities, services)
2. Create Application layer (use cases, DTOs)
3. Implement in Infrastructure layer
4. Add controllers in Presentation layer

### Testing Strategy
- **Unit Tests**: Domain and Application layers
- **Integration Tests**: Infrastructure components
- **E2E Tests**: Full API workflow
- **Performance Tests**: Load and stress testing

---

## 📞 Support

For issues or questions about the migration:
1. Check logs for specific error messages
2. Verify configuration settings
3. Test individual components (health checks)
4. Review the troubleshooting section

The migration maintains all existing functionality while providing a more robust, scalable, and maintainable architecture.
