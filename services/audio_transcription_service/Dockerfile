FROM python:3.9-slim

WORKDIR /app

# Install git and ffmpeg
RUN apt-get update && \
    apt-get install -y git ffmpeg curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy basic requirements (except gigaam)
COPY requirements.txt .


# Установка остальных зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install GigaAM with longform dependencies
RUN git clone https://github.com/salute-developers/GigaAM.git /tmp/GigaAM && \
    cd /tmp/GigaAM && \
    pip install -e . && \
    pip install ".[longform]" && \
    cd /app

# Copy application code
COPY . /app

# Create directory for audio uploads
RUN mkdir -p /app/uploads

EXPOSE 8004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"] 