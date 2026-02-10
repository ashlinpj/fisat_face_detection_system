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
USE_GPU = False  # Enable GPU acceleration
USE_THREADED_RECOGNITION = True  # Run recognition in background thread
USE_DNN_DETECTOR = True  # Use GPU-accelerated DNN face detector
DNN_CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold for DNN detector

# Camera settings
CAMERA_INDEX = 0  # Default camera (change if you have multiple cameras)
FRAME_WIDTH = 640  # Lower resolution = faster processing
FRAME_HEIGHT = 480
FPS = 60

# RTSP Stream settings (TEMPORARILY DISABLED)
USE_RTSP = False  # Set to True to use RTSP stream instead of webcam
RTSP_URL = "rtsp://10.255.48.170:8554/live"  # Default OBS RTSP stream URL
# Alternative RTSP URL examples:
# RTSP_URL = "rtsp://username:password@192.168.1.100:554/stream"
# RTSP_URL = "rtsp://admin:admin@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0"
RTSP_RECONNECT_ATTEMPTS = 5  # Number of reconnection attempts
RTSP_RECONNECT_DELAY = 2  # Seconds between reconnection attempts
RTSP_BUFFER_SIZE = 1  # Buffer size (lower = less latency)
RTSP_TRANSPORT = "tcp"  # tcp or udp (tcp is more reliable)

# YouTube Stream settings
USE_YOUTUBE = True  # Set to True to use YouTube stream as camera feed
YOUTUBE_URL = "https://youtube.com/live/Vr-ocieh1oo"  # lofi hip hop radio - 24/7 live stream (default test)
# Replace with your own YouTube stream URL:
# YOUTUBE_URL = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
# Example live stream URLs:
# YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
YOUTUBE_RECONNECT_ATTEMPTS = 3  # Number of reconnection attempts
YOUTUBE_RECONNECT_DELAY = 3  # Seconds between reconnection attempts
YOUTUBE_QUALITY = "best[height<=720]"  # Video quality (best, worst, best[height<=720], etc.)

# UI settings
WINDOW_TITLE = "College Canteen Face Detection System"

# Time settings
MIN_TIME_BETWEEN_LOGS = 30  # Minimum seconds between logging same person