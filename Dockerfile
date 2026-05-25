FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (relaxed versions for compatibility)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    timm \
    "Pillow>=10.2.0,<10.3.0" \
    facenet-pytorch \
    opencv-python-headless

# Copy backend code
COPY backend/main.py .
COPY backend/model.py .
COPY backend/preprocessing.py .
COPY backend/models/ ./models/

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
