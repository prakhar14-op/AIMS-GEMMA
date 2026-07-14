
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL, YOLO_CONFIDENCE_THRESHOLD, YOLO_DEVICE, TRIGGER_CLASSES, BENIGN_CLASSES, RESTRICTED_ZONE_POLYGON

logger = logging.getLogger(__name__)

class YOLOTripwire:
    # CHANGE: Updated argument name to 'model_path' to match your pipeline.py
    def __init__(self, model_path=YOLO_MODEL, confidence_threshold=YOLO_CONFIDENCE_THRESHOLD, device=YOLO_DEVICE, roi_polygon=RESTRICTED_ZONE_POLYGON):
        self._model_name = model_path 
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._model = YOLO(self._model_name,task='detect') 
        self._loaded = True
        self._roi = np.array(roi_polygon, dtype=np.int32) if roi_polygon else None

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self._model(frame, conf=self._confidence_threshold, device=self._device, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None: continue
            for i in range(len(result.boxes)):
                class_id = int(result.boxes.cls[i])
                label = self._model.names[class_id]
                bbox = result.boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                detections.append({
                    "label": label,
                    "confidence": float(result.boxes.conf[i]),
                    "bbox": bbox,
                    "in_roi": self._bbox_in_roi(bbox)
                })
        return detections

    def _bbox_in_roi(self, bbox: list) -> bool:
        if self._roi is None: return True
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        return cv2.pointPolygonTest(self._roi, (cx, cy), False) >= 0

    def should_trigger_analysis(self, detections: list[dict]) -> bool:
        if not detections: return False
        in_zone = [d for d in detections if d.get("in_roi", True)]
        if not in_zone: return False
        detected_labels = {d["label"] for d in in_zone}
        if detected_labels & TRIGGER_CLASSES: return True
        return False