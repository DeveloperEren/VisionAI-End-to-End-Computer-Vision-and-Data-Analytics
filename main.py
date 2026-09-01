from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import pyodbc
import os
import time

app = FastAPI()
model = YOLO("best.pt")

OUTPUT_DIR = "detected_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DB_CONFIG = r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=ERENZEPHYRUS\MSSQLSERVER1;DATABASE=SecurityCamera;Trusted_Connection=yes;"

def save_to_db(vehicle_type: str, confidence: float, bbox: list):
    try:
        conn = pyodbc.connect(DB_CONFIG)
        cursor = conn.cursor()
        query = """
            INSERT INTO VehicleDetections (VehicleType, Confidence, BoundingBox)
            VALUES (?, ?, ?)
        """
        cursor.execute(query, (vehicle_type, confidence, str(bbox)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

@app.post("/detect/")
async def detect_vehicles(
    file: UploadFile = File(...),
    conf: float = Form(0.5),
    iou: float = Form(0.45),
    classes: str = Form("")
):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    class_names_to_ids = {name: id for id, name in model.names.items()}
    target_classes = None
    
    if classes:
        selected_names = [c.strip() for c in classes.split(",")]
        target_classes = [class_names_to_ids[name] for name in selected_names if name in class_names_to_ids]

    # Model çağrısına conf, iou ve target_classes eklendi
    results = model(img, conf=conf, iou=iou, imgsz=1280, classes=target_classes)
    
    detections = []
    for r in results:
        for box in r.boxes:
            box_conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            bbox = [int(x) for x in box.xyxy[0].tolist()]

            detections.append({
                "class": class_name,
                "confidence": round(box_conf, 2),
                "bbox": bbox
            })
            save_to_db(class_name, round(box_conf, 2), bbox)

    annotated_img = results[0].plot()
    timestamp = int(time.time())
    output_filename = f"detected_{timestamp}_{file.filename}"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_filepath, annotated_img)

    return {
        "filename": file.filename, 
        "total_detections": len(detections), 
        "detections": detections,
        "download_url": f"/download/{output_filename}"
    }

@app.get("/download/{filename}")
async def download_image(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg", filename=filename)
    return {"error": "Dosya bulunamadı."}