import cv2
import time
import threading

import collections
import numpy as np

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
IS_KAGGLE = True  # Set to False when running locally with a physical webcam
VIDEO_PATH = "/kaggle/input/datasets/vincyjoel/cctv-01/CCTV 01/VIRAT_S_010003_05_000499_000523.mp4"  # Path to your test video if on Kaggle
TARGET_FPS = 1.0  # Extract exactly 1 frame per second
BUFFER_MAX_LEN = 5  # Maintain a rolling 5-second temporal window
IMAGE_SIZE = (896, 896)  # Native resolution expected by Gemma-4 Multimodal

class VisualIngestionPipeline:
    def __init__(self):
        # The rolling memory ring buffer (Thread-safe structure)
        self.frame_buffer = collections.deque(maxlen=BUFFER_MAX_LEN)
        
        # Thread management controls
        self.is_running = False
        self.capture_thread = None
        self.buffer_lock = threading.Lock()
        
        # State marker for Phase 2 Handoff
        self.is_frozen = False

    def initialize_camera(self):
        """Establishes the hardware or file capture connection."""
        if IS_KAGGLE:
            print(f"[Phase 1] Initializing file stream from: {VIDEO_PATH}")
            cap = cv2.VideoCapture(VIDEO_PATH)
        else:
            print("[Phase 1] Initializing physical hardware via V4L2 backend...")
            # Use Video4Linux2 backend for low-level Linux hardware control
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
            # CRITICAL: Force OS driver buffer size to 1.
            # Prevents driver-level lag when the processing loop slows down.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        if not cap.isOpened():
            raise IOError("[Phase 1] Critical Error: Could not open video source.")
            
        return cap

    def _ingestion_worker(self):
        """Background thread execution loop for zero-latency capture."""
        cap = self.initialize_camera()
        last_extraction_time = 0.0
        time_interval = 1.0 / TARGET_FPS

        while self.is_running:
            if self.is_frozen:
                # Phase 2 has requested a freeze; sleep thread briefly to avoid spinning CPU
                time.sleep(0.05)
                continue

            if IS_KAGGLE:
                # File streams need direct retrieval as cap.grab() doesn't drop frames the same way
                ret, frame = cap.read()
                if not ret:
                    print("[Phase 1] Video file reached end of stream. Looping...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            else:
                # Hardware stream: Vacuum the bus using high-speed grab to avoid latency spikes
                if not cap.grab():
                    continue
                    
            current_time = time.time()
            
            # Temporal Decimation: Evaluate if it is time to extract a frame
            if (current_time - last_extraction_time) >= time_interval:
                
                if not IS_KAGGLE:
                    # Decompress only the target frame from the hardware pointer
                    ret, frame = cap.retrieve()
                    if not ret:
                        continue
                
                # Spatial Normalization & Color Space Alignment
                # Convert OpenCV's native BGR layout to the AI's required RGB layout
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize using INTER_AREA interpolation (optimized for shrinking images)
                resized_frame = cv2.resize(rgb_frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                
                # Thread-safe write execution to the rolling buffer
                with self.buffer_lock:
                    self.frame_buffer.append(resized_frame)
                    
                last_extraction_time = current_time
                
            # Yield CPU execution briefly to prevent thread starvation
            time.sleep(0.001)

        cap.release()
        print("[Phase 1] Video ingestion pipeline safely terminated.")

    def start(self):
        """Spawns the continuous acquisition background thread."""
        if self.is_running:
            print("[Phase 1] Pipeline is already actively running.")
            return
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._ingestion_worker, daemon=True)
        self.capture_thread.start()
        print("[Phase 1] Ingestion background thread spawned successfully.")

    def stop(self):
        """Safely stops the thread loop."""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join()

    def get_current_buffer_snapshot(self):
        """Provides a safe deep-copy payload to hand off to downstream phases."""
        with self.buffer_lock:
            # Returns a snapshot list containing up to 5 historical normalized NumPy arrays
            return list(self.frame_buffer)

# ==========================================
# SIMULATION INTEGRATION (How to run it)
# ==========================================
if __name__ == "__main__":
    # Instantiate the system chassis
    pipeline = VisualIngestionPipeline()
    pipeline.start()
    
    try:
        # Simulate the runtime execution loop
        for seconds in range(8):
            time.sleep(1)
            snapshot = pipeline.get_current_buffer_snapshot()
            print(f"Time Elapsed: {seconds + 1}s | Active Buffer Payload Size: {len(snapshot)} frames")
            
            if len(snapshot) > 0:
                print(f" -> Current active frame tensor dimensions: {snapshot[-1].shape}")
                
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()