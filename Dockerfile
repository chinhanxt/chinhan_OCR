# Dockerfile for Baidu Unlimited-OCR with Unsloth GPU Acceleration
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set non-interactive environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_NAME="baidu/Unlimited-OCR"

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade --force-reinstall --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
RUN pip install --no-cache-dir --upgrade --force-reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install Unsloth from source for latest VLM optimizations
RUN pip install --no-cache-dir --no-deps "unsloth @ git+https://github.com/unslothai/unsloth.git"

# Copy application files
COPY app.py /workspace/app.py

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Launch FastAPI web server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
