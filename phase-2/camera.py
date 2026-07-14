
import cv2
import time
import threading
import collections
import numpy as np
from config import VIDEO_PATH, IMAGE_SIZE, TARGET_FPS, BUFFER_MAX_LEN

class VisualIngestionPipeline:
    def __init__(self):
        # collections.deque automatically drops the oldest frames when maxlen is reached
        self.frame_buffer = collections.deque(maxlen=BUFFER_MAX_LEN)
        self.is_running = False
        self.capture_thread = None
        self.buffer_lock = threading.Lock()
        self.is_frozen = False

    def _ingestion_worker(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Fallback just in case OpenCV cannot read the video metadata
        if video_fps == 0 or video_fps is None:
            video_fps = 24.0 
            
        # Calculate the exact sleep needed to mimic a live CCTV camera
        realtime_sleep = 1.0 / video_fps
        
        print(f"[Phase 1] Initializing MP4 stream. Simulating live camera at {video_fps:.2f} FPS.")
        
        while self.is_running:
            if self.is_frozen:
                time.sleep(0.1)
                continue

            # Read sequentially: This is much more stable than cap.set() jumping
            ret, frame = cap.read()
            
            if not ret:
                print("[Phase 1] End of stream. Looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to frame 0
                continue
            
            # Spatial Normalization & RGB Conversion
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(rgb_frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            
            # Thread-safe buffer update
            with self.buffer_lock:
                self.frame_buffer.append(resized_frame)
            
            # Pace the ingestion to exactly match real-world time (The "Chef" speed limit)
            time.sleep(realtime_sleep)
            
        cap.release()

    def start(self):
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._ingestion_worker, daemon=True)
        self.capture_thread.start()
        print("[Phase 1] Ingestion background thread spawned.")

    def stop(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join()

    def get_current_buffer_snapshot(self):
        with self.buffer_lock:
            # Returns a copy of the current buffer for YOLO to process
            return list(self.frame_buffer)