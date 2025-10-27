# Start from Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy code and requirements
COPY YOLO_DETECTION.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create volumes for images and results
VOLUME ["/app/images", "/app/results"]

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "YOLO_DETECTION:app", "--host", "0.0.0.0", "--port", "8000"]
