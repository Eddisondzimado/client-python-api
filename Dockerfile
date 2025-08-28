FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
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

# Run the application
CMD ["python", "predictor.py"]