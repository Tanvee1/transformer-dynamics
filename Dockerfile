# Dockerfile for Transformer Dynamics Lab
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt-get/lists/*

# Install lightweight CPU-only PyTorch first (150MB instead of 1GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable for Python path
ENV PYTHONPATH=/app/backend

# Expose port
EXPOSE 7860

# Start FastAPI application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
