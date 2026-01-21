# 🍽️ College Canteen Face Detection System

## B.Tech Mini Project

A real-time face detection and recognition system designed to track student visits to the college canteen. This system helps prevent false complaints by maintaining a record of who visited the canteen with timestamps.

## 🆕 NEW: Multi-Angle Face Registration (v2.0)

**Now features smartphone-style face registration!** Similar to iPhone Face ID or Android Face Unlock, the system captures your face from 5 different angles for improved recognition accuracy.

- 📸 **5-Angle Capture**: Center, Left, Right, Up, Down
- 🎯 **Higher Accuracy**: Better recognition from any viewing angle
- 🌟 **Guided Process**: Step-by-step instructions with emoji indicators
- 💡 **Lighting Tolerance**: Works in various lighting conditions

👉 **[See Multi-Angle Setup Guide](MULTI_ANGLE_SETUP.md)** for detailed instructions

---

## 📋 Features

- **Real-time Face Detection** using YOLOv8 and DeepFace
- **Face Recognition** to identify registered students
- **🆕 Multi-Angle Registration** for enhanced accuracy (smartphone-like)
- **Automatic Visit Logging** with date, time, and duration tracking
- **Student Registration** with face capture
- **Unknown Face Detection** for identifying unregistered visitors
- **GUI Application** with modern Tkinter interface
- **Statistics & Reports** for daily/weekly analysis
- **CSV Export** for visit logs
- **SQLite Database** for persistent storage

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Face Detection | YOLOv8 (Ultralytics) |
| Face Recognition | DeepFace (Facenet512) |
| Database | SQLite |
| GUI | Tkinter |
| Image Processing | OpenCV |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
face/
├── config.py                  # Configuration settings
├── database.py                # Database operations
├── face_recognition_module.py # Face detection & recognition
├── main.py                    # Command-line application
├── gui_app.py                 # GUI application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── database/                  # Auto-created
│   ├── canteen.db            # SQLite database
│   └── faces/                # Stored face images
└── screenshots/              # Captured screenshots
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- Webcam/USB Camera
- Windows/Linux/MacOS

### Step 1: Clone or Download

```bash
https://github.com/ashlinpj/fisat_face_detection_system.git
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: First Run (Downloads YOLO model automatically)

```bash
python main.py
```

---

## 🚀 Usage

### Option 1: GUI Application (Recommended)

```bash
python gui_app.py
```

**Features:**
- 📹 Live Detection tab - Real-time face detection
- 👥 Students tab - Manage registered students
- 📋 Visit Logs tab - View and export visit history
- 📊 Statistics tab - Daily/weekly statistics

### Option 2: Command Line Application

```bash
python main.py
```

**Menu Options:**
1. Start Real-time Detection
2. Register New Student
3. View Statistics
4. View Today's Logs
5. List All Students
6. Exit

### Keyboard Controls (During Detection)

| Key | Action |
|-----|--------|
| Q | Quit detection |
| R | Register new student |
| S | Show statistics |
| L | Show today's logs |
| Space | Capture screenshot |

---

## 📸 How to Register Students

### Method 1: During Live Detection

1. Start the detection (`python gui_app.py` or `python main.py`)
2. Press `R` or click "Register Face"
3. Enter student details:
   - Student ID (e.g., 21CS001)
   - Full Name
   - Department (CSE, ECE, etc.)
   - Year (1-4)
4. Look at the camera
5. System captures and registers the face

### Method 2: Batch Registration (Advanced)

You can also add faces programmatically:

```python
from face_recognition_module import FaceRecognitionSystem
import cv2

system = FaceRecognitionSystem()

# Capture frame from camera or load image
frame = cv2.imread("student_photo.jpg")

# Register student
system.register_new_student(
    frame=frame,
    student_id="21CS001",
    name="John Doe",
    department="CSE",
    year=3
)
```

---

## 📊 Database Schema

### Students Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary Key |
| student_id | TEXT | Unique Student ID |
| name | TEXT | Student Name |
| department | TEXT | Department |
| year | INTEGER | Year of Study |
| face_embedding | TEXT | Face vector (JSON) |
| face_image_path | TEXT | Path to face image |
| created_at | TIMESTAMP | Registration time |

### Visit Logs Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary Key |
| student_id | TEXT | Student ID |
| student_name | TEXT | Student Name |
| entry_time | TIMESTAMP | Entry time |
| exit_time | TIMESTAMP | Exit time |
| duration_minutes | INTEGER | Visit duration |
| date | DATE | Visit date |
| is_known | INTEGER | 1=Known, 0=Unknown |

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Camera settings
CAMERA_INDEX = 0              # Change if multiple cameras
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Recognition settings
CONFIDENCE_THRESHOLD = 0.5    # YOLO detection confidence
FACE_RECOGNITION_THRESHOLD = 0.6  # Face matching threshold

# Time settings
MIN_TIME_BETWEEN_LOGS = 30    # Seconds between same person logs
```

---

## 🔧 Troubleshooting

### Camera not detected
- Check if camera is connected
- Try different `CAMERA_INDEX` values (0, 1, 2)
- Ensure no other application is using the camera

### Low detection accuracy
- Ensure good lighting
- Face should be clearly visible
- Increase `FACE_RECOGNITION_THRESHOLD` for stricter matching

### Slow performance
- Reduce `FRAME_WIDTH` and `FRAME_HEIGHT`
- Use GPU if available (CUDA-enabled TensorFlow)

### Model download issues
- Ensure internet connection for first run
- Models are cached after first download

---

## 📈 Use Cases

1. **Prevent False Complaints**: Students claiming they didn't eat at canteen
2. **Attendance Tracking**: Monitor canteen usage patterns
3. **Security**: Identify unknown individuals
4. **Analytics**: Peak hours, popular times, visitor frequency

---

## 🎓 Project Details

**Project Type:** B.Tech Mini Project  
**Domain:** Computer Vision, Machine Learning  
**Academic Year:** 2025-26

### Technologies Learned
- Deep Learning (YOLO, CNN)
- Face Recognition Systems
- Real-time Image Processing
- Database Management
- GUI Development
- Python Programming

---

## 📝 Future Enhancements

- [ ] Web-based dashboard
- [ ] Mobile app for registration
- [ ] Multiple camera support
- [ ] Cloud database integration
- [ ] Email/SMS notifications
- [ ] Face mask detection
- [ ] Anti-spoofing measures

---

## 👥 Contributors

- Add your team members here

---

## 📄 License

This project is developed for educational purposes as part of B.Tech curriculum.

---

## 🆘 Support

For issues or questions, contact your project guide or create an issue in the repository.
