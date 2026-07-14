
import time
import logging
import sys
import cv2  # <-- ADDED CV2
import os
# Ensure the current directory is in the path
sys.path.append('/kaggle/working')

from camera import VisualIngestionPipeline
from tripwire import YOLOTripwire

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PipelineConnector")

def run_pipeline():
    # 1. Initialize Phase 1
    ingestion = VisualIngestionPipeline()
    ingestion.start()
    
    # 2. Initialize Phase 2
    tripwire = YOLOTripwire(model_path="yolov8n.onnx")
    
    logger.info("Pipeline connected: Phase 1 feeding into Phase 2.")
    
    try:
        while True:
            # 3. Pull frame from Phase 1
            snapshot = ingestion.get_current_buffer_snapshot()
            
            if snapshot:
                # 4. Use the robust detection logic from tripwire.py
                latest_frame = snapshot[-1]
                
                # Get full list of objects
                detections = tripwire.detect(latest_frame) 
                
                # --- DEBUG LOG (HEARTBEAT) ---
                persons_detected = [d for d in detections if d['label'] == 'person']
                persons_in_roi = [d for d in persons_detected if d.get('in_roi', False)]
                logger.info(f"Phase 2 Heartbeat | Objects: {len(detections)} | Persons: {len(persons_detected)} | Persons in ROI: {len(persons_in_roi)}")
                # -----------------------------
                
                # --- NEW: SAVE THE FRAME IF A PERSON IS DETECTED ---
                if len(persons_detected) > 0:
                    # Make a copy of the frame to draw on
                    annotated_frame = latest_frame.copy()
                    
                    # Draw a box for every person found
                    for p in persons_detected:
                        x1, y1, x2, y2 = p['bbox']
                        # Draw green rectangle
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Add label and confidence score
                        label_text = f"Person: {p['confidence']:.2f}"
                        cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Save the image to the Kaggle working directory
                    # Note: We convert RGB back to BGR for OpenCV to save the colors correctly
                    save_path = f"/kaggle/working/person_detected_{int(time.time())}.jpg"
                    cv2.imwrite(save_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                    logger.info(f"📸 Saved image with bounding box to: {save_path}")
                # ---------------------------------------------------

                # Reflex logic: should we wake up the Brain (Phase 3/4)?
                if tripwire.should_trigger_analysis(detections):
                    logger.warning("🚨 THREAT DETECTED: Waking Gemma...")
                    # Phase 3/4 integration will go here
            
            # Run inference ~5 times per second (0.2s) to keep up with the camera thread
            time.sleep(0.2) 
            
    except KeyboardInterrupt:
        logger.info("Stopping pipeline...")
        ingestion.stop()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        ingestion.stop()

if __name__ == "__main__":
    run_pipeline()