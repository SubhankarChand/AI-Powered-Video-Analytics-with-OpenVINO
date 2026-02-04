from ultralytics import YOLO
import shutil
import os

print("Downloading and Exporting standard YOLOv8n model...")

# 1. Load the official YOLOv8n model (it will download automatically)
model = YOLO("yolov8n.pt")

# 2. Export it to OpenVINO format
# This ensures the input/output shapes match EXACTLY what the code expects
model.export(format="openvino")

print("------------------------------------------------")
print("SUCCESS! Model exported to folder: 'yolov8n_openvino_model/'")
print("You can now run app.py")
print("------------------------------------------------")