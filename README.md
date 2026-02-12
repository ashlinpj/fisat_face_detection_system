# 🍽️ College Canteen Face Detection System

## B.Tech Mini Project

A real-time face detection and recognition system designed to track student visits to the college canteen. This system helps prevent false complaints by maintaining a record of who visited the canteen with timestamps.

---

## 📋 Features

- **Real-time Face Detection** using YOLOv8 and DeepFace
- **Face Recognition** to identify registered students
- **RTSP Stream Support** - Use network cameras or OBS streams
- **YouTube Stream Support** - Use live YouTube streams as camera feed
- **Video File Processing** - Automated queue system for processing pre-recorded videos
- **Automatic Reconnection** for RTSP streams
- **Automatic Visit Logging** with date, time, and duration tracking
- **Student Registration** with face capture
- **Unknown Face Detection** for identifying unregistered visitors
- **GUI Application** with modern Tkinter interface
- **Statistics & Reports** for daily/weekly analysis
- **CSV Export** for visit logs
- **SQLite Database** for persistent storage

---

## 🛠️ Technology Stack

| Component        | Technology             |
| ---------------- | ---------------------- |
| Face Detection   | YOLOv8 (Ultralytics)   |
| Face Recognition | DeepFace (Facenet512)  |
| Database         | SQLite                 |
| GUI              | Tkinter                |
| Image Processing | OpenCV                 |
| Stream Support   | RTSP, YouTube (yt-dlp) |
| Language         | Python 3.10+           |

---

## 📁 Project Structure

```
fisat_face_detection_system/
├── config.py                  # Configuration settings
├── database.py                # Database operations
├── face_recognition_module.py # Face detection & recognition
├── video_processor.py         # Video processing queue system
├── gui_app.py                 # GUI application (Tkinter)
├── main.py                    # Command-line application
├── requirements.txt           # Python dependencies
├── setup.py                   # Setup script (optional)
├── start.bat                  # Windows batch starter
├── test_system.py             # System test script
├── test_rtsp.py               # RTSP stream test utility
├── test_youtube_stream.py     # YouTube stream test utility
├── utils.py                   # Utility functions
├── README.md                  # Project documentation
├── RTSP_SETUP_GUIDE.md        # RTSP configuration guide
├── RTSP_QUICK_START.md        # RTSP quick start guide
├── YOUTUBE_STREAM_GUIDE.md    # YouTube stream guide
├── database/                  # Database folder
│   ├── canteen.db             # SQLite database file
│   └── faces/                 # Stored face images (per student)
├── videos/                    # Incoming videos for processing
├── processed_videos/          # Processed videos (with _done suffix)
├── logs/                      # Video processing logs
│   └── video_processing.log   # Processing log file
├── screenshots/               # Captured screenshots
├── __pycache__/               # Python bytecode cache
│   └── ...                    # Compiled .pyc files
└── venv/                      # (Optional) Virtual environment
    └── ...
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- Webcam/USB Camera, RTSP Stream (e.g., from OBS, IP Camera), or YouTube Stream
- Windows/Linux/MacOS
- Internet connection (for YouTube streams and initial model downloads)

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
- 🎥 Video Processing tab - Process pre-recorded video files

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

| Key   | Action               |
| ----- | -------------------- |
| Q     | Quit detection       |
| R     | Register new student |
| S     | Show statistics      |
| L     | Show today's logs    |
| Space | Capture screenshot   |

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

### Method 2: Register from Image File (Admin)

1. Go to the **Students** tab in the GUI
2. Click **🖼️ Register from Image**
3. Select a high-resolution student photo (JPG/PNG)
4. Enter student details when prompted
5. The system will detect, enhance, and register the student from the image

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

## 📡 RTSP Stream Support

The system now supports RTSP streams from network cameras, OBS, and other RTSP sources!

### Quick Setup with OBS

1. **Install OBS Studio** and the **RTSP Server Plugin**
2. **Configure RTSP** in OBS (Tools → RTSP Server Settings)
3. **Update config.py:**
   ```python
   USE_RTSP = True
   RTSP_URL = "rtsp://127.0.0.1:8554/live"
   ```
4. **Run the application** - It will connect to the RTSP stream automatically!

### Test Your RTSP Connection

```bash
python test_rtsp.py
```

### Full Setup Guides

- 📖 **Quick Start:** See [RTSP_QUICK_START.md](RTSP_QUICK_START.md) - 5-minute setup
- 📚 **Complete Guide:** See [RTSP_SETUP_GUIDE.md](RTSP_SETUP_GUIDE.md) - Full documentation

### RTSP Features

- ✅ Automatic reconnection on connection loss
- ✅ Support for IP cameras with authentication
- ✅ Low-latency streaming with TCP/UDP
- ✅ Configurable buffer size and reconnection attempts
- ✅ Easy switching between webcam and RTSP

---

## 📺 YouTube Stream Support

The system now supports YouTube live streams as a camera feed source!

### Quick Setup

1. **Install yt-dlp dependency** (already in requirements.txt):

   ```bash
   pip install yt-dlp
   ```

2. **Update config.py:**

   ```python
   USE_YOUTUBE = True
   YOUTUBE_URL = "https://www.youtube.com/watch?v=jfKfPfyJRdk"  # Default: Lofi Hip Hop Radio
   ```

3. **Run the application** - It will connect to the YouTube stream automatically!

### Test Your YouTube Stream

```bash
python test_youtube_stream.py
```

### Full Guide

- 📖 **Complete Guide:** See [YOUTUBE_STREAM_GUIDE.md](YOUTUBE_STREAM_GUIDE.md) - Full documentation

### YouTube Stream Features

- ✅ Support for YouTube live streams and videos
- ✅ Automatic stream URL extraction with yt-dlp
- ✅ Quality optimization (720p by default)
- ✅ Works with 24/7 live streams (e.g., Lofi Hip Hop Radio)
- ✅ Easy URL configuration
- ✅ Test utility to verify stream compatibility

### Use Cases for YouTube Streams

- 🎬 Demo presentations without physical camera
- 🧪 Testing and development
- 📹 Using pre-recorded footage for training
- 🌐 Remote demos over the internet

---

## 🎥 Video File Processing System

The system includes an **automated video processing queue** that can process pre-recorded video files to detect and recognize faces!

### 🌟 Key Features

- ✅ **Automated Queue System** - Drop videos in a folder and let the system process them
- ✅ **Batch Processing** - Process multiple videos sequentially
- ✅ **Smart Frame Sampling** - Process 1 frame every 10 frames for efficiency
- ✅ **Video Timestamp Tracking** - Each recognition includes timestamp (HH:MM:SS) in video
- ✅ **Auto Wait & Rescan** - Automatically waits for new videos when folder is empty
- ✅ **Background Processing** - Non-blocking GUI operation
- ✅ **Comprehensive Logging** - Console, file, and database logs
- ✅ **Duplicate Filtering** - Avoids logging same person within 30 seconds

### 🚀 How to Use

#### Option 1: Using GUI (Recommended)

1. **Launch the application:**

   ```bash
   python gui_app.py
   ```

2. **Go to the 🎥 Video Processing tab**

3. **Upload videos:**
   - Click **📁 Upload Video** to select video files
   - Or manually place videos in the `videos/` folder

4. **Start processing:**
   - Click **▶ Start Processing**
   - Watch real-time recognition results
   - System automatically processes all videos in queue

5. **Monitor progress:**
   - View current processing status
   - See pending videos count
   - Track recognized faces in real-time
   - Check session statistics

#### Option 2: Manual Folder Method

1. **Place videos in the `videos/` folder:**

   ```
   videos/
   ├── lecture_recording.mp4
   ├── cafeteria_footage.avi
   └── event_video.mov
   ```

2. **Open GUI and start processing** as described above

3. **Processed videos** are automatically moved to `processed_videos/` with `_done` suffix:
   ```
   processed_videos/
   ├── lecture_recording_done.mp4
   ├── cafeteria_footage_done.avi
   └── event_video_done.mov
   ```

### ⚙️ Configuration

Edit [config.py](config.py) to customize video processing:

```python
# Video Processing Queue settings
VIDEO_FRAME_SKIP_INTERVAL = 10        # Process 1 frame every N frames
VIDEO_MAX_DURATION = 10 * 60          # Max video duration (seconds)
VIDEO_SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
VIDEO_AUTO_WAIT_ENABLED = True        # Wait and rescan when folder empty
VIDEO_WAIT_INTERVAL = 60              # Seconds to wait before rescan
```

### 📋 Video Requirements

| Requirement      | Details                                |
| ---------------- | -------------------------------------- |
| **Formats**      | `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv` |
| **Max Duration** | 10 minutes (configurable)              |
| **Resolution**   | Any (automatically processed)          |
| **File Naming**  | Must not contain `_done` suffix        |

### 📊 Processing Workflow

```
┌─────────────────────┐
│  Place video in     │
│  videos/ folder     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  System scans       │
│  for new videos     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Validate video:    │
│  • Duration check   │
│  • Format check     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Process frames:    │
│  • Sample frames    │
│  • Detect faces     │
│  • Recognize        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Log results:       │
│  • Database entry   │
│  • Log file         │
│  • Console output   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Mark as processed: │
│  • Rename _done     │
│  • Move to folder   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Process next video │
│  or wait for new    │
└─────────────────────┘
```

### 📝 Logging & Results

#### Console Output

```
Processing video: event_recording.mp4
Duration: 00:05:23
FPS: 30.00
Total frames: 9690
Recognized: John Doe at 00:01:15
Recognized: Jane Smith at 00:02:30
Recognized: John Doe at 00:03:45
Completed: event_recording.mp4
```

#### Log File: `logs/video_processing.log`

```
[2026-02-12 14:30:15] Processing video: event_recording.mp4
[2026-02-12 14:30:16] Recognized: John Doe | Timestamp: 00:01:15
[2026-02-12 14:30:18] Recognized: Jane Smith | Timestamp: 00:02:30
[2026-02-12 14:30:20] Completed: event_recording.mp4
```

#### Database

All recognitions are stored in the `visit_logs` table with:

- `source_type`: "video"
- `video_name`: Original filename
- `video_timestamp`: HH:MM:SS in video
- All standard visit log fields

### 🎯 Use Cases

- 📹 **Process Event Recordings** - Analyze who attended past events
- 🏫 **Classroom Attendance** - Process lecture recordings for attendance
- 🔍 **Security Review** - Review security footage for specific individuals
- 📊 **Traffic Analysis** - Analyze foot traffic patterns over time
- 🧪 **Testing & Development** - Test recognition without live camera

### 🔧 Troubleshooting Video Processing

#### Video Not Processing

- ✔️ Check video format is supported
- ✔️ Ensure video duration < 10 minutes
- ✔️ Verify filename doesn't contain `_done`
- ✔️ Check `logs/video_processing.log` for errors

#### No Faces Detected

- ✔️ Ensure video has clear face visibility
- ✔️ Check lighting conditions in video
- ✔️ Verify students are registered in database
- ✔️ Try reducing `FACE_RECOGNITION_THRESHOLD` in config

#### Slow Processing

- ✔️ Increase `VIDEO_FRAME_SKIP_INTERVAL` (process fewer frames)
- ✔️ Use GPU acceleration if available
- ✔️ Process shorter videos
- ✔️ Reduce video resolution before upload

### 💡 Pro Tips

- **Optimize Frame Sampling**: Balance between speed and accuracy by adjusting `VIDEO_FRAME_SKIP_INTERVAL`
- **Batch Upload**: Place multiple videos in folder for overnight processing
- **Quality Matters**: Higher quality videos = better face recognition
- **Organize Results**: Check `processed_videos/` folder regularly
- **Monitor Logs**: Review `logs/video_processing.log` for insights

---

## �📊 Database Schema

### Students Table

| Column          | Type      | Description        |
| --------------- | --------- | ------------------ |
| id              | INTEGER   | Primary Key        |
| student_id      | TEXT      | Unique Student ID  |
| name            | TEXT      | Student Name       |
| department      | TEXT      | Department         |
| year            | INTEGER   | Year of Study      |
| face_embedding  | TEXT      | Face vector (JSON) |
| face_image_path | TEXT      | Path to face image |
| created_at      | TIMESTAMP | Registration time  |

### Visit Logs Table

| Column           | Type      | Description                      |
| ---------------- | --------- | -------------------------------- |
| id               | INTEGER   | Primary Key                      |
| student_id       | TEXT      | Student ID                       |
| student_name     | TEXT      | Student Name                     |
| entry_time       | TIMESTAMP | Entry time                       |
| exit_time        | TIMESTAMP | Exit time                        |
| duration_minutes | INTEGER   | Visit duration                   |
| date             | DATE      | Visit date                       |
| is_known         | INTEGER   | 1=Known, 0=Unknown               |
| screenshot_path  | TEXT      | Path to visit screenshot         |
| source_type      | TEXT      | Source: 'live' or 'video'        |
| video_name       | TEXT      | Video filename (if source=video) |
| video_timestamp  | TEXT      | Timestamp in video (HH:MM:SS)    |

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

# RTSP Stream settings
USE_RTSP = False              # Enable RTSP stream
RTSP_URL = "rtsp://127.0.0.1:8554/live"  # RTSP stream URL

# YouTube Stream settings
USE_YOUTUBE = False           # Enable YouTube stream
YOUTUBE_URL = "https://www.youtube.com/watch?v=jfKfPfyJRdk"  # YouTube URL
```

---

## 🔧 Troubleshooting

### Camera not detected

- Check if camera is connected
- Try different `CAMERA_INDEX` values (0, 1, 2)
- Ensure no other application is using the camera

### RTSP stream issues

- Verify RTSP URL is correct
- Check network connectivity
- Ensure RTSP server is running (for OBS streams)
- Try changing `RTSP_TRANSPORT` between 'tcp' and 'udp' in config.py

### YouTube stream issues

- Ensure `yt-dlp` is installed: `pip install yt-dlp`
- Verify YouTube URL is a valid live stream or video
- Check internet connection
- Run `python test_youtube_stream.py` to diagnose issues
- Try updating yt-dlp: `pip install --upgrade yt-dlp`

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
