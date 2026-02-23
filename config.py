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

# UI / logging controls
SHOW_WINDOW = False  # Set False to show names only (no camera feed), True to show full camera feed with annotations
LOG_RECOGNITIONS = True  # Print detected faces to console
VERBOSE_RTSP_LOGS = False  # Set True to see dropped-frame warnings

# Frame buffer settings to avoid stale/overlapping frames
FRAME_BUFFER_SIZE = 2  # Keep only the freshest 1-2 frames in memory
FRAME_FETCH_TIMEOUT = 0.5  # Seconds to wait for a new frame before treating stream as stalled

# Camera settings
CAMERA_INDEX = 0  # Default camera (change if you have multiple cameras)
FRAME_WIDTH = 640  # Lower resolution = faster processing
FRAME_HEIGHT = 480
FPS = 60

# RTSP Stream settings
USE_RTSP = False  # Set to True to use RTSP stream instead of webcam
RTSP_URL = "rtsp://localhost:8554/live"  # Default OBS RTSP stream URL
# Alternative RTSP URL examples:
# RTSP_URL = "rtsp://username:password@192.168.1.100:554/stream"
# RTSP_URL = "rtsp://admin:admin@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0"
RTSP_RECONNECT_ATTEMPTS = 5  # Number of reconnection attempts
RTSP_RECONNECT_DELAY = 2  # Seconds between reconnection attempts
RTSP_BUFFER_SIZE = 100  # Buffer size (lower = less latency, 1 keeps freshest frame)
RTSP_TRANSPORT = "tcp"  # tcp is more reliable and avoids UDP packet loss artifacts
RTSP_TARGET_FPS = 7  # Request FPS from the source; keep realistic to avoid queueing
RTSP_RESTART_INTERVAL = 0  # Seconds; restart capture periodically to clear mosaic artifacts (set 0 to disable)

# OpenCV FFMPEG capture options to reduce latency and drop stale frames
# Format: key:value will be converted to "key;value" joined by "|"
RTSP_OPENCV_OPTIONS = {
	"rtsp_transport": RTSP_TRANSPORT,
	"fflags": "nobuffer",
	"max_delay": "500000",      # microseconds; keep low to avoid long buffering
	"buffer_size": "102400",    # bytes; small to keep latency low
	"stimeout": "2000000",      # microseconds to wait for packets
	"reorder_queue_size": "0",  # disable frame reordering to reduce lag
}

# Number of grab() calls before each read() to skip queued frames when processing lags
RTSP_PREGRAB_COUNT = 2

# UI settings
WINDOW_TITLE = "College Canteen Face Detection System"

# Time settings
MIN_TIME_BETWEEN_LOGS = 30  # Minimum seconds between logging same person
