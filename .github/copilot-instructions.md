# AI Agent Instructions for YOLO Detection Project

## Project Overview
This is a FastAPI-based object detection service using YOLOv8. The project provides an API endpoint for detecting objects in images using the YOLO (You Only Look Once) model.

## Key Components

### Main Application (`YOLO_DETECTION.py`)
- FastAPI application with two endpoints:
  - `/detection_images` (POST): Performs object detection
  - `/help` (GET): Provides API usage information
- Uses YOLOv8m model for detection
- Processes images and outputs:
  - Annotated images with bounding boxes
  - CSV files with detection results

## Critical Patterns & Conventions

### Object Detection Pipeline
```python
# Load model once at endpoint call
model = YOLO("yolov8m.pt")

# Process results format:
[x, y, width, height, class_name, confidence]
```

### File Paths & Volumes
- Input images expected in: `C:\Users\a.alsubhi\Desktop\Images`
- Results saved in:
  - Annotated images: `{input_folder}/annotated_image.jpg`
  - Detection data: `{results_dir}/detections.csv`

### Docker Setup
- Uses Python 3.12 slim base image
- Two mounted volumes:
  - `/app/images`: For input images
  - `/app/results`: For detection results
- FastAPI server runs on port 8000

## Dependencies
Key packages required (from `requirements.txt`):
- `fastapi`: Web framework
- `uvicorn[standard]`: ASGI server
- `opencv-python`: Image processing
- `ultralytics`: YOLO implementation

## Development Workflow
1. Setup:
   ```bash
   pip install -r requirements.txt
   ```
2. Run locally:
   ```bash
   uvicorn YOLO_DETECTION:app --host 0.0.0.0 --port 8000
   ```
3. Docker build:
   ```bash
   docker build -t yolo-detection .
   ```

## Common Patterns
- Image paths are hardcoded - when modifying, ensure all path references are updated
- Detection classes are predefined: ['car', 'bus', 'truck', 'motorbike', 'bicycle']
- Results are always saved both visually (annotated image) and numerically (CSV)

## Known Issues
- Possible variable name inconsistencies in the detection loop (classe, cls_id, classs_name)
- Missing error handling for file operations
- Hard-coded file paths need to be made configurable