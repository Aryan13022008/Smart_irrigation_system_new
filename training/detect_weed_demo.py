"""
Smart Agriculture System - Weed Detection Demo Script
========================================================================
Loads the trained YOLO model, runs inference on a sample field image, 
and displays the detection results (bounding boxes, class names, confidence).
"""

from pathlib import Path
from ultralytics import YOLO
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "training" / "models"
YOLO_PATH = PROJECT_ROOT / "models" / "weed_model.pt"

def main():
    print("=" * 60)
    print("SMART AGRICULTURE - WEED DETECTION INFERENCE DEMO")
    print("=" * 60)

    # 1. Load YOLO Model (falls back to yolov8n if custom weights are pending)
    model_path = YOLO_PATH if YOLO_PATH.exists() else "yolov8n.pt"
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    # 2. Simulate or load a test crop image
    # (Replace with an actual image path from your project if available)
    sample_image_path = PROJECT_ROOT / "data" / "sample_crop.jpg"
    
    if not sample_image_path.exists():
        print("Sample image not found on disk. Creating a blank test canvas...")
        import numpy as np
        # Create a dummy 640x640 green image representing a field
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_img[:, :] = [34, 139, 34] # Forest green
        cv2.imwrite(str(sample_image_path), dummy_img)

    print(f"Running inference on: {sample_image_path}")

    # 3. Perform Inference
    results = model(str(sample_image_path), conf=0.3)

    # 4. Extract and Print Detection Metrics
    for r in results:
        boxes = r.boxes
        print(f"\nTotal detections found: {len(boxes)}")
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"  - Object {i+1}: Class='{class_name}', Confidence={conf:.2f}")

    # 5. Save Annotated Output
    annotated_img = results[0].plot()
    output_path = PROJECT_ROOT / "training" / "processed" / "detection_result.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated_img)
    print(f"\nSaved annotated result image to: {output_path}")

if __name__ == "__main__":
    main()