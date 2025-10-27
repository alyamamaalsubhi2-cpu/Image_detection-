import cv2
import csv
import asyncio
from fastapi import FastAPI, HTTPException
from ultralytics import YOLO
from pathlib import Path

app = FastAPI()

# Define classes
CLASSES = ['car', 'bus', 'truck', 'motorbike', 'bicycle']

# Preload YOLO model on GPU
model = YOLO("yolov8m.pt", device="cpu")


@app.post("/detection_images")
async def detect_images():
    """                                                     
    Run YOLO detection asynchronously .
    """
    folder_path = Path(r"C:\Users\a.alsubhi\Desktop\Images")
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    # Check if folder exists
    if not folder_path.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")

    # Find all image files step by step
    image_files = []
    for file in folder_path.iterdir():  # Go through each file in the folder
        if file.suffix.lower() in allowed_ext:  # Check if it is an allowed image
            image_files.append(file)

    if len(image_files) == 0:  # No valid images found
        raise HTTPException(status_code=404, detail=f"No images found in {folder_path}")

    all_detections = []
    processed = 0

    # Process each image asynchronously
    for image_file in image_files:
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Warning: Could not read image {image_file.name}")
            continue

        try:
            results = await asyncio.to_thread(model.predict, image)
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            continue

        # Extract detection boxes
        for result in results:
            for box in result.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width = x2 - x1
                    height = y2 - y1
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = CLASSES[class_id] if 0 <= class_id < len(CLASSES) else f"class_{class_id}"

                    # Draw rectangle and label on the image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        f"{class_name} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    # Save detection info for CSV
                    all_detections.append([image_file.name, x1, y1, width, height, class_name, confidence])
                except Exception as e:
                    print(f"Bounding box error for {image_file.name}: {e}")
                    continue

        # Save annotated image
        save_path = folder_path / f"annotated_{image_file.name}"
        cv2.imwrite(str(save_path), image)
        processed += 1

    # Write all detections to CSV
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
 async def help():
    return {
        "message": "YOLO Object Detection API",
        "endpoints": {"/detection_images": "Run detection on images in the specified folder.",
            "/help": "Show this help message.",
        },
    }
