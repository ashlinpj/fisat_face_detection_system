# Configuration file for Face Detection System

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database settings
DATABASE_PATH = os.path.join(BASE_DIR, "database", "canteen.db")
FACES_DIR = os.path.join(BASE_DIR, "database", "faces")
# Directory for visit screenshots
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

# YOLO Model settings
YOLO_MODEL = "yolov8n.pt"  # General model, works well
CONFIDENCE_THRESHOLD = 0.4

# Face Recognition settings - LOWERED for better matching with older photos
FACE_RECOGNITION_THRESHOLD = 0.35  # Lower = more lenient matching (was 0.6)
FACE_EMBEDDING_MODEL = "Facenet"  # Facenet is faster and good for variations

# Performance settings - GPU OPTIMIZED
PROCESS_EVERY_N_FRAMES = 5  # Faster processing with GPU
DETECTION_SCALE = 1.0  # Full resolution with GPU acceleration
USE_GPU = True  # Enable GPU acceleration
USE_THREADED_RECOGNITION = True  # Run recognition in background thread
USE_DNN_DETECTOR = True  # Use GPU-accelerated DNN face detector
DNN_CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold for DNN detector

# Camera settings
CAMERA_INDEX =0 # Default camera (change if you have multiple cameras)
FRAME_WIDTH = 640  # Lower resolution = faster processing
FRAME_HEIGHT = 480
FPS = 60

# UI settings
WINDOW_TITLE = "College Canteen Face Detection System"

# Time settings
MIN_TIME_BETWEEN_LOGS = 30  # Minimum seconds between logging same person
