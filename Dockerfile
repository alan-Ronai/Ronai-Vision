## CPU-first base image. For GPU builds, switch to an appropriate CUDA base image
## and a gpu-specific requirements file (e.g. requirements-gpu.txt).
FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ca-certificates \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create a non-root user for running the app
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Run the FastAPI app by default; main.py is currently empty so use the API module
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
