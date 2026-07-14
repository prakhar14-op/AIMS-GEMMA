# %%writefile config.py
# ==========================================
# SYSTEM CONFIGURATION
# ==========================================

# Path to the VIRAT video
VIDEO_PATH = "/kaggle/input/datasets/vincyjoel/cctv-01/CCTV 01/VIRAT_S_010003_05_000499_000523.mp4"

# Phase 1 & 2 Constants
IMAGE_SIZE = (896, 896)
TARGET_FPS = 1.0
BUFFER_MAX_LEN = 5

# Phase 2: YOLO Configuration
YOLO_MODEL = "yolov8n.onnx" # Ensure you have exported this to ONNX
YOLO_CONFIDENCE_THRESHOLD = 0.70
YOLO_DEVICE = "cpu"
TRIGGER_CLASSES = {"person"}
BENIGN_CLASSES = {"potted plant", "chair"}
RESTRICTED_ZONE_POLYGON = [(100, 100), (700, 100), (700, 700), (100, 700)]
ROI_POLYGON = RESTRICTED_ZONE_POLYGON