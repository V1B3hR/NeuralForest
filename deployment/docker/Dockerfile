# NeuralForest Docker Image
# Production-ready deployment container for NeuralForest

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy NeuralForest code
COPY . /app/

# Create directories for models and data
RUN mkdir -p /app/checkpoints /app/data /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NEURALFOREST_HOME=/app

# Expose API port (if running as a service)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from api import NeuralForestAPI; print('healthy')"

# Default command
CMD ["python", "-u"]
