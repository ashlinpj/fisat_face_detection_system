import os

# --- PATH SETTINGS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FOLDER = os.path.join(BASE_DIR, "database")

# Create folders if they don't exist
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

DATABASE_PATH = os.path.join(DB_FOLDER, "canteen.db")
FACES_DIR = os.path.join(DB_FOLDER, "faces")
SCREENSHOTS_DIR = os.path.join(DB_FOLDER, "screenshots")

# Create subfolders
for folder in [FACES_DIR, SCREENSHOTS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- APPLICATION SETTINGS ---
WINDOW_TITLE = "College Canteen Face Detection System"
CAMERA_INDEX = 0  # 0 for webcam, 1 or 2 for external USB cameras
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- AI & DETECTION SETTINGS ---
CONFIDENCE_THRESHOLD = 0.5    # YOLO Detection confidence (0.0 to 1.0)
FACE_RECOGNITION_THRESHOLD = 0.40  # Slightly looser (0.40) to ensure detection happens easily
FACE_EMBEDDING_MODEL = "Facenet512" 
DETECTION_SCALE = 0.5  # Scale down frame for faster detection (0.5 = 50% size)

# --- SYSTEM SETTINGS ---
# CRITICAL FIX: Increased to 120 seconds (2 minutes) to prevent log spam.
# The system will only write to the DB once every 2 minutes per person.
LOG_COOLDOWN = 120             

USE_GPU = True                # Tries to use GPU if available
USE_THREADED_RECOGNITION = True