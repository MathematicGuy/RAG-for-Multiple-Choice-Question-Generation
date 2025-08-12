#? Base image: PyTorch with CUDA 12.1, cuDNN8, Python 3.11, development tools
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# Avoids interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements folder and install dependencies
COPY requirements.txt .

# Download wheels for caching
RUN pip download --no-cache-dir -r requirements.txt -d ./wheels

# Install from local wheels using pip (not uv to avoid errors)
RUN pip install --no-cache-dir --no-index --find-links=./wheels -r requirements.txt

# Copy the rest of the application code
COPY . .

# Optional: suppress transformer warnings
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=true

# Expose FastAPI port
EXPOSE 8000

# Run app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]