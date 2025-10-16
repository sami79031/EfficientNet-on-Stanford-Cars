FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements-gcp.txt .
RUN pip install --no-cache-dir -r requirements-gcp.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/input/stanford data/output/logs data/output/trials

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "training/train.py"]
