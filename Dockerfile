FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces runs as non-root user (uid 1000)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# YOLO needs a writable config directory
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

# Give ownership to appuser
RUN chown -R appuser:appuser /app

USER appuser

# HuggingFace Spaces MUST expose port 7860
EXPOSE 7860

CMD ["python", "app.py"]