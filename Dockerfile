# Multi-stage Dockerfile for Zepto Data & AI Platform
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Default startup runs both FastAPI backend and Streamlit frontend
CMD ["./start.sh"]

