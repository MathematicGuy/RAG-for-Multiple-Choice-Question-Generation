# Requirements Structure

This directory contains environment-specific dependency files for the Enhanced RAG MCQ Generator.

## File Structure

```
requirements/
├── base.txt     # Core dependencies shared across all environments
├── dev.txt      # Development dependencies (includes base.txt)
├── prod.txt     # Production dependencies (includes base.txt)
└── README.md    # This file
```

## Installation Instructions

### For Development Environment

```bash
# Install development dependencies (includes all base dependencies)
pip install -r requirements/dev.txt
```

### For Production Environment

```bash
# Install production dependencies (includes all base dependencies)
pip install -r requirements/prod.txt
```

### For Base Environment Only

```bash
# Install only core dependencies
pip install -r requirements/base.txt
```

## GPU Support

### CUDA 11.8 (Recommended)
```bash
pip install -r requirements/prod.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip install -r requirements/prod.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### CPU Only
The default installation uses CPU-only versions. For explicit CPU installation:
```bash
# Replace faiss-cpu with faiss-gpu if needed
pip install -r requirements/prod.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Dependency Categories

### Base Dependencies (`base.txt`)
- **AI/ML Core**: transformers, torch, accelerate, bitsandbytes
- **LangChain**: Complete LangChain ecosystem for RAG
- **Vector DB**: FAISS for similarity search
- **Document Processing**: PyPDF, unstructured for document parsing
- **Utilities**: numpy, pandas, tqdm, python-dotenv

### Development Dependencies (`dev.txt`)
- **Jupyter**: Full Jupyter ecosystem for interactive development
- **Code Quality**: black, isort, flake8, mypy for code formatting and linting
- **Testing**: pytest with async and coverage support
- **API Development**: FastAPI with full development features
- **Documentation**: mkdocs for documentation generation

### Production Dependencies (`prod.txt`)
- **Web Server**: FastAPI with uvicorn/gunicorn for production serving
- **Security**: python-jose, passlib for authentication
- **Monitoring**: psutil, structured logging
- **Optional**: Database drivers and caching solutions

## Version Strategy

- **Pinned Versions**: All dependencies use exact versions for reproducibility
- **Compatibility Tested**: Versions are tested to work together
- **Security**: Regular updates for security patches
- **Performance**: Optimized for the specific use case

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB (CPU inference)
- **Storage**: 10GB for models and dependencies
- **Python**: 3.8+

### Recommended for GPU
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8 or 12.1
- **Storage**: 20GB+ for models and cache

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi
   # Install matching PyTorch version
   ```

2. **Memory Issues**
   ```bash
   # Use CPU-only version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **BitsAndBytesConfig Issues**
   ```bash
   # Ensure CUDA is properly installed
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Environment Variables

Create a `.env` file in the API root:
```env
# Required tokens
HUGGING_FACE_TOKEN=your_token_here
LANGSMITH_API_KEY=your_key_here

# Optional configurations
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
```

## Docker Support

The requirements are optimized for Docker builds:

```dockerfile
# Use specific Python version
FROM python:3.10-slim

# Install production dependencies
COPY requirements/prod.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/prod.txt

# ... rest of Dockerfile
```

## Contribution Guidelines

When adding new dependencies:

1. Add to appropriate file (`base.txt`, `dev.txt`, or `prod.txt`)
2. Use exact version pinning
3. Test compatibility with existing dependencies
4. Update this README if needed
5. Consider security implications for production dependencies
