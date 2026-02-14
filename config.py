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
USE_YOUTUBE = False  # Set to True to use YouTube stream as camera feed
YOUTUBE_URL = "https://youtube.com/live/Vr-ocieh1oo"  # lofi hip hop radio - 24/7 live stream (default test)
# Replace with your own YouTube stream URL:
# YOUTUBE_URL = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
# Example live stream URLs:
# YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
YOUTUBE_RECONNECT_ATTEMPTS = 3  # Number of reconnection attempts
YOUTUBE_RECONNECT_DELAY = 3  # Seconds between reconnection attempts
YOUTUBE_QUALITY = "best[height<=720]"  # Video quality (best, worst, best[height<=720], etc.)

# Video File settings
USE_VIDEO_FILE = True  # Set to True to use a pre-downloaded video file
VIDEO_FILE_PATH = "database/vid1.mp4"  # Path to video file (relative or absolute)
# Examples:
# VIDEO_FILE_PATH = "videos/sample.mp4"  # Relative path
# VIDEO_FILE_PATH = r"C:\Users\Videos\my_video.mp4"  # Absolute path (Windows)
# VIDEO_FILE_PATH = "/home/user/videos/sample.mp4"  # Absolute path (Linux/Mac)
VIDEO_LOOP = False  # Set to True to loop the video when it ends

# UI settings
WINDOW_TITLE = "College Canteen Face Detection System"

# Time settings
MIN_TIME_BETWEEN_LOGS = 30  # Minimum seconds between logging same person

# Video Processing Queue settings
VIDEO_FRAME_SKIP_INTERVAL = 5  # Process 1 frame every N frames (10 = faster processing)
VIDEO_MAX_DURATION = 10 * 60  # Maximum video duration in seconds (10 minutes)
VIDEO_SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv']  # Supported video formats
VIDEO_AUTO_WAIT_ENABLED = True  # Automatically wait and rescan when no videos found
VIDEO_WAIT_INTERVAL = 60  # Seconds to wait before rescanning empty folder

# ── Shared Folder / Network Queue settings ──
# Watch folder where another PC drops 10-minute .mp4 segments
VIDEO_WATCH_FOLDER = r"C:\cctv_storage\cam1"
# Internal working directories (auto-created)
VIDEO_PROCESSING_DIR = os.path.join(BASE_DIR, "processing")
VIDEO_ERROR_DIR = os.path.join(BASE_DIR, "error")
# File stability: file must not change size for this many seconds before processing
# Set to 0 to skip the stability check entirely (process immediately)
VIDEO_STABILITY_WAIT_SECONDS = 0
# How often to re-check file size during stability wait (seconds)
VIDEO_STABILITY_CHECK_INTERVAL = 5
# How often to scan the watch folder when idle (seconds)
VIDEO_SCAN_INTERVAL = 60

# ── Accuracy / Voting settings ──
VIDEO_MIN_CONSECUTIVE_FRAMES = 10 # Must see same person N frames in a row (was 3)
VIDEO_BLUR_THRESHOLD = 80.0        # Laplacian variance below this → skip blurry face
VIDEO_VOTE_RATIO = 0.40            # ≥40 % of track votes for identity to confirm
VIDEO_CONFIDENCE_FLOOR = 0.50      # Ignore recognitions below this confidence
VIDEO_RECOGNITION_THRESHOLD = 0.38 # DeepFace cosine distance threshold

# ── Async Video Recognition (live-feed style) ──
VIDEO_DISPLAY_CONFIDENCE = 0.70    # Only show name if confidence >= this (0.70 = 70%)
VIDEO_RECOGNITION_WORKERS = 2      # Background threads for embed+match (2 = good default)
VIDEO_RECHECK_INTERVAL = 5         # Re-queue a tracked face every N processed frames if low-conf

# ── Video Tab Visualization settings ──
VIDEO_PREVIEW_FPS = 10             # Max FPS for GUI frame preview (was 5, now faster)
VIDEO_PREVIEW_WIDTH = 640          # Preview display width in pixels