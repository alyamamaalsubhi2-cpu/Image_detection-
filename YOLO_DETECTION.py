import cv2
import csv
import asyncio
from fastapi import FastAPI, HTTPException
from ultralytics import YOLO
from pathlib import Path

app = FastAPI()

# Define classes
CLASSES = ['car', 'bus', 'person', 'motorbike', 'bicycle']

# Load YOLO model (adjust device if needed)
model = YOLO("yolov8n.py")

RESULTS_DIR = Path("/app/images")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use pathlib everywhere (no os)
folder_path = RESULTS_DIR
allowed_ext = {'.jpg', '.jpeg', '.png'}

@app.post("/detection_images")
async def detect_images():
    
    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")
   
    image_files = []
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in allowed_ext:
            image_files.append(file)
    if not image_files:
        raise HTTPException(status_code=404, detail=f"No images found in {folder_path}")

    all_detections = []
    processed = 0


    for image_file in image_files:
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Could not read image {image_file.name}")
            continue

        try:
            # Run prediction asynchronously
            results = await asyncio.to_thread(model.predict, image)
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            continue

        # Process results
        for result in results:
            for box in result.boxes:
                try:
                    xy = box.xyxy[0]
                    x1, y1, x2, y2 = map(int, xy)
                    width = x2 - x1
                    height = y2 - y1
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0]) 
                    class_name = CLASSES[class_id] if 0 <= class_id < len(CLASSES) else f"class_{class_id}"

                    # Draw bounding box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 2)

                    all_detections.append([image_file.name, x1, y1, width, height, class_name, confidence])

                except Exception as e:
                    print(f"Bounding box parse error for {image_file.name}: {e}")
                    continue

        # Save annotated image next to inputs
        save_path = folder_path / f"annotated_{image_file.name}"
        cv2.imwrite(str(save_path), image)
        processed += 1

    # Write CSV using pathlib path
    csv_path = folder_path / "detections.csv"
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "x", "y", "width", "height", "class", "confidence"])
            writer.writerows(all_detections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write CSV: {e}")

    return {
        "message": f"Processed {processed} images successfully.",
        "csv_file": str(csv_path),
    }


@app.get("/help")
async def get_help():
    return {
        "message": "YOLO Object Detection API",
        "endpoints": {
            "/detection_images": "Run detection on images in the specified folder.",
            "/help": "Show this help message.",
        },
    }
