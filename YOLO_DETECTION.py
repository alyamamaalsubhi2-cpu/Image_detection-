import cv2
from fastapi import FastAPI ,HTTPException
from ultralytics import YOLO
import os
import csv
import asyncio

app = FastAPI()

# Define classes
classes = ['car', 'bus', 'truck', 'motorbike', 'bicycle']

@app.post("/detection_images")
async def start_detection():
    model = YOLO("yolov8m.pt", device='cuda')  
    folder_path = r"C:\Users\a.alsubhi\Desktop\Images"
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")

    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in allowed_ext]

    if not files:
        raise HTTPException(status_code=404, detail=f"No images found in {folder_path}")

    all_objects = []
    processed = 0

    for fname in files:
        image_path = os.path.join(folder_path, fname)
        image = cv2.imread(image_path)
        if image is None:
            continue

        try:
            results = await asyncio.to_thread(model, image)
        except Exception as e:
            continue

        # results may be an iterable of result objects
        for result in results:
            boxes = getattr(result, "boxes", [])
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                except Exception:
                    continue
                width = x2 - x1
                height = y2 - y1
                confidence = float(box.conf[0]) 
                class_id = int(box.cls[0]) 
                class_name = classes[class_id] if 0 <= class_id < len(classes) else f"class_{class_id}"

                # draw box & label on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                all_objects.append([fname, x1, y1, width, height, class_name, confidence])

        # save annotated image
        save_path = os.path.join(folder_path, f"annotated_{fname}")
        cv2.imwrite(save_path, image)
        processed += 1

    # write CSV
    csv_path = os.path.join(folder_path, "detections.csv")
    try:
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "x", "y", "width", "height", "class", "confidence"])
            writer.writerows(all_objects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write CSV: {e}")

    return {"message": f"Processed {processed} images", "csv": csv_path}

@app.get("/help/")
async def read_me():
    return {
        "message": "This is a YOLO object detection API. Use the /detection_images endpoint to run detection on images in the specified folder path."
    }
