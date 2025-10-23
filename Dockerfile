# Build stage
FROM python:3.10.14-slim-bullseye AS builder

# Set working directory
WORKDIR /application

# Copy requirements first for caching
COPY requirements.txt .

# Install build dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Runtime stage
FROM python:3.10.14-slim-bullseye

# Set working directory
WORKDIR /application

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the rest of the code
COPY . .

# Pre-warm heavy imports to reduce cold start
USER root
RUN python -c "import pandas; import numpy; import sklearn; import flask; import torch" \
    && chown -R appuser:appuser /application
USER appuser

# Expose port 5000 (Flask default)
EXPOSE 5000

# Run the app
CMD ["python", "application.py"]
