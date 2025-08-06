#? Stage 1: Build with Python 3.11 and CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set non-interactive mode for APT
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies: Python, curl, etc.
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1 \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

#? Stage 2: App installation
# Ensure pipx and uv is install
RUN python -m pip install pipx && \
    pipx install uv && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

# Make sure uv is in PATH (adjust if needed)
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy files within requirements folder
COPY requirements/ ./requirements/
# Pre-download wheels
RUN uv pip download -r requirements/prod.txt -d ./wheels && \
    uv pip install --system --no-index --find-links=./wheels -r requirements/prod.txt

# Copy the rest of the code
COPY . .

# Set environment variable for GPU use (optional)
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=true

EXPOSE 8000

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]