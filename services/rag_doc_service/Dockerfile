FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data models

# Copy the application code
COPY . .

# Create a volume mount point for the model
VOLUME /app/models

# Expose the port
EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]