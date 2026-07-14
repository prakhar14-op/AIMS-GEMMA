
import time
import logging
import sys
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
                
                # Reflex logic: should we wake up the Brain (Phase 3/4)?
                if tripwire.should_trigger_analysis(detections):
                    logger.warning("🚨 THREAT DETECTED: Waking Gemma...")
                    # Get the 5-frame snapshot from the buffer
                    snapshot = ingestion.get_current_buffer_snapshot()
                    # Call the analyzer
                    verdict = analyzer.analyze_event(snapshot, detections)
                    logger.info(f"Gemma Verdict: {verdict}")
            
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