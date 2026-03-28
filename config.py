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

# Face Recognition settings
FACE_RECOGNITION_THRESHOLD = 0.60  # Higher threshold for Facenet512's stronger embeddings
FACE_MATCH_MARGIN = 0.10  # Tighter margin improves discrimination between similar faces
DISPLAY_CONFIDENCE_THRESHOLD = 0.40  # Show recognized names with moderate confidence
FACE_EMBEDDING_MODEL = "Facenet512"  # 512-dim embeddings: 4x more discriminative than Facenet-128

# Multi-embedding gallery matching (uses all registration samples, not just average)
FACE_MULTI_SAMPLE_MATCH = True  # Enable gallery-based matching for better accuracy
FACE_GALLERY_TOP_K = 5  # Average top-K gallery hits for robust scoring
FACE_ADAPTIVE_MARGIN = True  # Scale margin up when top-2 candidates are close

# Performance settings - RTSP optimized for LOW LATENCY recognition
PROCESS_EVERY_N_FRAMES = 2  # Queue faces for recognition every 2nd frame (was 6)
GUI_PROCESS_EVERY_N_FRAMES = 2  # GUI triggers process_frame every 2nd frame (was 4)
GUI_RECOGNITION_INTERVAL_SEC = 0.10  # 100ms recognition refresh (was 250ms)
GUI_UI_TARGET_FPS = 30  # Stable UI target to reduce event-loop pressure
GUI_PREVIEW_WIDTH = 640  # Smaller preview improves rendering FPS
DETECT_EVERY_N_FRAMES = 1  # Detect every frame for stable person presence
ENHANCE_BEFORE_RECOGNITION = False  # Disable expensive enhancement for faster RTSP performance
DETECTION_SCALE = 0.6  # Used by Haar fallback; lower is faster
NO_FACE_HOLD_FRAMES = 8  # Keep last detected boxes briefly when a few frames miss detection
BBOX_MATCH_DISTANCE_FACTOR = 0.30  # Tighter bbox association to avoid cross-person name swaps
RECOGNITION_LABEL_TTL_SEC = 3.0  # Keep labels visible longer to avoid flicker (was 1.5)
PENDING_LABEL_TTL_SEC = 1.5  # Pending candidate labels expire quickly when face leaves
KNOWN_CONFIRM_FRAMES = 1  # Show name on first match — Facenet512+gallery is accurate enough (was 2)
UNKNOWN_CONFIRM_FRAMES = 3  # Require a bit more consistency before switching to Unknown
RECOGNITION_QUEUE_SIZE = 6  # Allow multiple faces per cycle instead of single-face bottleneck
RESULT_QUEUE_SIZE = 12  # Keep several recognition results to avoid drops during bursts
USE_GPU = True  # Enable GPU acceleration
USE_THREADED_RECOGNITION = True  # Run recognition in background thread
USE_DNN_DETECTOR = True  # Use GPU-accelerated DNN detector when CUDA backend is available
ALLOW_CPU_DNN = False  # Keep False for FPS; CPU DNN is usually slower than Haar fallback
DNN_CONFIDENCE_THRESHOLD = 0.45  # Slightly lower to reduce intermittent no-face misses

# UI / logging controls
SHOW_WINDOW = False  # Names-only detection display (no live video feed)
LOG_RECOGNITIONS = True  # Print detected faces to console
VERBOSE_RTSP_LOGS = False  # Set True to see dropped-frame warnings

# Frame buffer settings to avoid stale/overlapping frames
FRAME_BUFFER_SIZE = 1  # Keep exactly one frame in memory (latest only)
FRAME_FETCH_TIMEOUT = 0.08  # Short wait prevents stale blocking and keeps loop real-time

# Real-time frame policy
REALTIME_ONLY_MODE = True  # Drop queued/old frames aggressively and prefer latest frame only

# Camera settings
CAMERA_INDEX = 0  # Default camera (change if you have multiple cameras)
FRAME_WIDTH = 640  # Lower resolution = faster processing
FRAME_HEIGHT = 480
FPS = 60

# Multi-camera source settings
# If USE_MULTI_CAMERA is True and CAMERA_SOURCES has 2+ items,
# the app opens all sources and runs recognition across all streams.
# Each source can be:
# - int camera index (e.g., 0, 1)
# - RTSP URL (recommended for Android IP Webcam app)
# - HTTP MJPEG URL (Android IP Webcam often provides /video endpoint)
USE_MULTI_CAMERA = False
CAMERA_SOURCES = [
	"rtsp://192.168.1.3:8080/h264.sdp",
	# "rtsp://192.168.1.51:8080/h264_pcm.sdp",
	# "http://192.168.1.52:8080/video",
]

# RTSP Stream settings
USE_RTSP = True  # Use RTSP stream instead of laptop webcam
# go2rtc relay settings (recommended for low-latency development)
GO2RTC_STREAM_NAME = "cam"
GO2RTC_SOURCE_URL = "rtsp://192.168.1.3:8080/h264.sdp"  # Original camera stream ingested by go2rtc
RTSP_URL = f"rtsp://127.0.0.1:8554/{GO2RTC_STREAM_NAME}"  # App reads local low-latency go2rtc relay
# Alternative RTSP URL examples:
# RTSP_URL = f"rtsp://127.0.0.1:8554/{GO2RTC_STREAM_NAME}"  # App reads local low-latency go2rtc relay
# RTSP_URL = "rtsp://username:password@192.168.1.100:554/stream"
RTSP_RECONNECT_ATTEMPTS = 5  # Number of reconnection attempts
RTSP_RECONNECT_DELAY = 2  # Seconds between reconnection attempts
RTSP_BUFFER_SIZE = 1  # Keep freshest frame for low-latency RTSP
RTSP_TRANSPORT = "tcp"  # TCP is more stable and avoids RTP packet reorder/loss artifacts
RTSP_TARGET_FPS = 20  # Slightly lower target reduces encoder stress and macroblock artifacts
RTSP_RESTART_INTERVAL = 0  # Seconds; restart capture periodically to clear mosaic artifacts (set 0 to disable)

# OpenCV FFMPEG capture options to reduce latency and drop stale frames
# Format: key:value will be converted to "key;value" joined by "|"
RTSP_OPENCV_OPTIONS = {
	"rtsp_transport": RTSP_TRANSPORT,
	"fflags": "nobuffer+discardcorrupt+flush_packets",
	"flags": "low_delay",
	"avioflags": "direct",
	"probesize": "32",
	"analyzeduration": "0",
	"max_delay": "0",             # zero delay — always prefer latest packet
	"buffer_size": "65536",       # smaller buffer = less latency
	"stimeout": "2000000",        # 2s timeout for faster frozen-frame detection
	"reorder_queue_size": "0",    # disable frame reordering to reduce lag
	"framedrop": "1",             # drop frames when behind
}

# Number of grab() calls before each read() to skip queued frames when processing lags
RTSP_PREGRAB_COUNT = 2  # Grab 2 stale frames before reading fresh one (was 1)
RTSP_USE_HW_ACCEL = True  # Ask OpenCV/FFmpeg to use hardware decode when available
RTSP_FROZEN_THRESHOLD_SEC = 2.0  # Auto-reconnect if no frame arrives for this duration
RTSP_AUTO_RECONNECT = True  # Enable automatic RTSP reconnection on frozen stream

# Registration-mode low-latency controls
REGISTRATION_PREGRAB_COUNT = 4  # Moderate drain during registration for current-frame capture
REGISTRATION_DRAIN_READS = 1  # Keep 1 to avoid decode overhead; pre-grab already flushes backlog
REGISTRATION_DIRECT_PREGRAB_COUNT = 3  # Direct pose capture flush without starving decoder
REGISTRATION_MAX_FRAME_AGE_SEC = 0.45  # Allow brief jitter without dropping preview to blank
REGISTRATION_PREVIEW_REFRESH_MS = 20  # ~50 FPS preview refresh with lower CPU pressure
REGISTRATION_PREVIEW_WIDTH = 480  # Lower preview resolution for faster rendering
REGISTRATION_PREVIEW_HEIGHT = 270
FORCE_REOPEN_ON_REGISTRATION_START = True  # Reconnect stream when registration starts to drop old buffered data
FORCE_REOPEN_ON_REGISTRATION_END = True  # Reconnect back after registration mode
MIN_VALID_REGISTRATION_SAMPLES = 12  # Do not save a profile unless enough embeddings are usable
REGISTRATION_MIN_FACE_SIZE = 70  # Ignore tiny face crops during registration sample extraction

# UI settings
WINDOW_TITLE = "College Canteen Face Detection System"

# Time settings
MIN_TIME_BETWEEN_LOGS = 30  # Minimum seconds between logging same person





















