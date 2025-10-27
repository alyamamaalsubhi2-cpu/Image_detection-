# Start from a Python base image
FROM python:3.11

# Set working directory
WORKDIR /app

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
