"""Configuration file for Face Detection System"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database settings
DATABASE_PATH = os.path.join(BASE_DIR, "database", "canteen.db")
FACES_DIR = os.path.join(BASE_DIR, "database", "faces")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

# YOLO Model settings
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4

# Face Recognition settings
FACE_RECOGNITION_THRESHOLD = 0.35
FACE_EMBEDDING_MODEL = "Facenet"

# Performance settings
PROCESS_EVERY_N_FRAMES = 5
DETECTION_SCALE = 1.0
USE_GPU = True
USE_THREADED_RECOGNITION = True
USE_DNN_DETECTOR = True
DNN_CONFIDENCE_THRESHOLD = 0.7  # Increased to reduce false positives

# Face detection settings
CASCADE_MIN_NEIGHBORS = 5  # Higher = fewer false positives (4=permissive, 5-6=balanced, 7+=strict)
MIN_FACE_SIZE = 60  # Minimum face size in pixels (smaller = more sensitive, larger = fewer false positives)

# UI settings
SHOW_WINDOW = True
LOG_RECOGNITIONS = True
VERBOSE_RTSP_LOGS = False
WINDOW_TITLE = "College Canteen Face Detection System"

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

# Time settings
MIN_TIME_BETWEEN_LOGS = 30  # Minimum seconds between logging same person
VISIT_COOLDOWN_SECONDS = 300  # 5 minutes cooldown between logging same person

# Recognition constants
FACE_CENTER_DISTANCE_MULTIPLIER = 0.5  # Distance threshold for matching faces
DISPLAY_CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence to display recognition
CURRENT_FACES_TTL_SECONDS = 10  # Time-to-live for recognized face boxes
RECOGNITION_QUEUE_SIZE = 1  # Max size of recognition task queue

# Face enhancement settings
FACE_ENHANCEMENT_TARGET_SIZE = 224  # Target size for face enhancement
FACE_SHARPNESS_FACTOR = 1.5  # Sharpness enhancement multiplier
FACE_CONTRAST_FACTOR = 1.2  # Contrast enhancement multiplier
FACE_BRIGHTNESS_FACTOR = 1.1  # Brightness enhancement multiplier

# Detection thresholds
MIN_FACE_DETECTION_SIZE = 20  # Minimum face dimension in pixels for detection
FACE_ASPECT_RATIO_MIN = 0.5  # Minimum width/height ratio for valid face
FACE_ASPECT_RATIO_MAX = 2.0  # Maximum width/height ratio for valid face

# Frame processing
PROCESS_INTERVAL_FRAMES = 10  # Process every Nth frame for recognition
