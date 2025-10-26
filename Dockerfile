# Use NVIDIA CUDA runtime image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
# Set Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Install Python and git
RUN apt-get update && \
    apt-get install -y python3-pip git && \
    rm -rf /var/lib/apt/lists/*
# Check Python and pip version
RUN python3 --version
RUN pip3 --version
# Set working directory
WORKDIR /app
# Copy your code and requirements
COPY YOLO_DETECTION.py .
COPY requirements.txt .
# Install Python dependencies
RUN pip3 install -r requirements.txt
# Create volumes for images and results
VOLUME ["/app/images", "/app/results"]
# Expose FastAPI port
EXPOSE 8000
# Start FastAPI server
CMD ["uvicorn", "YOLO_DETECTION:app", "--host", "0.0.0.0", "--port", "8000"]
