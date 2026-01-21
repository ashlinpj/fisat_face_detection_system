# Configuration file for Face Detection System

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database settings
DATABASE_PATH = os.path.join(BASE_DIR, "database", "canteen.db")
FACES_DIR = os.path.join(BASE_DIR, "database", "faces")

# YOLO Model settings
YOLO_MODEL = "yolov8n.pt"  # General model, works well
CONFIDENCE_THRESHOLD = 0.4

# Face Recognition settings - STRICT ACCURACY (Prevent False Positives)
FACE_RECOGNITION_THRESHOLD = 0.55  # STRICT threshold to prevent wrong identifications
FACE_EMBEDDING_MODEL = "Facenet"  # Facenet is faster and good for variations
USE_ADVANCED_PREPROCESSING = True  # Enable CLAHE and lighting normalization

# Multi-angle face capture settings
NUM_CAPTURE_ANGLES = 5  # Number of different angles to capture during registration
CAPTURE_ANGLES = [
    {"name": "Center", "description": "Look straight at the camera", "emoji": "😊"},
    {"name": "Turn Left", "description": "Turn your head slightly to the left", "emoji": "😏"},
    {"name": "Turn Right", "description": "Turn your head slightly to the right", "emoji": "😌"},
    {"name": "Look Up", "description": "Tilt your head slightly up", "emoji": "😄"},
    {"name": "Look Down", "description": "Tilt your head slightly down", "emoji": "🙂"}
]

# Performance settings - OPTIMIZED FOR SPEED
PROCESS_EVERY_N_FRAMES = 2  # Process every 2 frames (FASTER logging)
DETECTION_SCALE = 0.5  # Lower scale = faster detection
USE_GPU = True  # Enable GPU acceleration
USE_THREADED_RECOGNITION = True  # Run recognition in background thread

# Camera settings
CAMERA_INDEX = 0  # Default camera (change if you have multiple cameras)
FRAME_WIDTH = 640  # Lower resolution = faster processing
FRAME_HEIGHT = 480
FPS = 60

# UI settings
WINDOW_TITLE = "College Canteen Face Detection System"

# Time settings
MIN_TIME_BETWEEN_LOGS = 60  # Minimum 60 seconds (1 min) between logging same person
