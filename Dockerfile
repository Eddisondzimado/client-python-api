FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including MySQL client libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data/models

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV GCS_BUCKET_NAME=client-support-chatbot-api.appspot.com

# Use a lightweight server and start immediately
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "3600", "predictor:app"]