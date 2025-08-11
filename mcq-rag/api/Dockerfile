#? Base image: PyTorch with CUDA 12.1, cuDNN8, Python 3.11, development tools
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Upgrade pip and install requirements
RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy app code
COPY . /app

# Expose port used by uvicorn
EXPOSE 8080

# Environment vars
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    TOKENIZERS_PARALLELISM=false 
    
# Run uvicorn (adjust to your app module if different)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
