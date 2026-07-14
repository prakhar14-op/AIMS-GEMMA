from ultralytics import YOLO

# 1. Download the base model
model = YOLO('yolov8n.pt') 

# 2. Export to ONNX format
# This will save 'yolov8n.onnx' in your current directory
model.export(format='onnx', imgsz=896) 

# 3. Verify it was created
import os
print("Does yolov8n.onnx exist?", os.path.exists('yolov8n.onnx'))