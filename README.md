# College Face Detection and Recognition System

Primarily a college face detection and recognition system for student monitoring and safety.

This project is implemented mainly for college use. It tracks student presence across camera feeds, identifies known individuals, flags unknown visitors, and provides admin tools for registration, reporting, and bulk onboarding.

## B.Tech Mini Project

Academic Year: 2025-26

## Key Features

- Real-time face detection and recognition
- Designed primarily for college campus/canteen deployment
- Crowd presence monitoring across live camera streams
- Multi-camera support (USB/RTSP/network sources)
- Gallery-based matching from multiple registration photos
- Automatic visit logging with timestamps and duration
- Unknown face detection and storage
- Admin GUI (Tkinter) for day-to-day operations
- Bulk student import from CSV + photo folders
- Export and statistics for daily/weekly analysis
- SQLite-based local persistent storage

## Technology Stack

| Component | Technology |
| --- | --- |
| Detection | YOLOv8 + OpenCV detectors |
| Recognition | DeepFace embeddings (default: SFace) |
| Database | SQLite |
| GUI | Tkinter |
| Video ingest | OpenCV + FFmpeg (RTSP support) |
| Language | Python 3.10+ |

## Project Structure

```text
fisat_face_detection_system/
|- camera_pipeline.py
|- config.py
|- database.py
|- face_matcher.py
|- face_recognition_module.py
|- gui_app.py
|- main.py
|- requirements.txt
|- test_system.py
|- test_rtsp.py
|- bulk_registration_template.csv
|- RTSP_QUICK_START.md
|- RTSP_SETUP_GUIDE.md
|- go2rtc.yaml
`- database/
   `- faces/
```

## Installation

### Prerequisites

- Python 3.10 or later
- Camera source: USB webcam or RTSP stream
- Windows/Linux/macOS

### 1) Clone

```bash
git clone https://github.com/ashlinpj/fisat_face_detection_system.git
cd fisat_face_detection_system
```

### 2) Create and activate virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the app

```bash
python gui_app.py
```

For CLI mode:

```bash
python main.py
```

## Registration Methods

### Method 1: Live capture from camera

1. Open the app (`python gui_app.py` or `python main.py`)
2. Press `R` or click `Register Face`
3. Enter student details (ID, name, department, year)
4. Follow pose instructions
5. System registers a multi-sample profile

### Method 2: Single image registration

1. Open GUI -> `Students` tab
2. Click `Register from Image`
3. Select one photo and fill student details

### Method 3: Multi-photo registration (recommended)

1. Open GUI -> `Students` tab
2. Click `Register from Photos`
3. Select 20-30+ photos from different angles/lighting
4. Submit details to create a stronger gallery profile

### Method 4: Bulk CSV import

1. Open GUI -> `Students` tab
2. Click `Bulk Import CSV`
3. Choose your CSV file and a base photos folder
4. Start import and monitor progress/logs

Required CSV columns:

- `student_id`
- `name`
- `department`
- `year`

Optional CSV columns:

- `photos`: semicolon-separated image paths
- `photos_folder`: folder path containing a student's images

Fallback behavior:

- If both optional columns are empty, importer tries `<base_folder>/<student_id>/`.

Reference template:

- `bulk_registration_template.csv`

## RTSP and go2rtc

The app supports direct RTSP sources and optional go2rtc relay.

In `config.py`:

- Set `USE_RTSP = True`
- Configure `RTSP_URL`

Direct camera (current default style):

```python
RTSP_URL = "rtsp://192.168.x.x:8080/h264.sdp"
```

go2rtc relay (recommended when stream stability is an issue):

```python
GO2RTC_STREAM_NAME = "cam"
RTSP_URL = f"rtsp://127.0.0.1:8554/{GO2RTC_STREAM_NAME}"
```

Useful files:

- `go2rtc.yaml`
- `start_go2rtc.bat`
- `update_go2rtc_config.ps1`
- `RTSP_QUICK_START.md`
- `RTSP_SETUP_GUIDE.md`
- `RTSP_IMPLEMENTATION.md`

## Configuration Quick Guide

Edit `config.py` for your setup.

Important options:

- `FACE_RECOGNITION_THRESHOLD`: Higher = stricter match acceptance
- `FACE_MATCH_MARGIN`: Extra separation needed between top candidates
- `USE_MULTI_CAMERA`: Enable recognition across multiple configured sources
- `CAMERA_SOURCES`: List of USB indexes and/or RTSP URLs
- `RTSP_USE_HW_ACCEL`: Hardware decode toggle
- `RTSP_TARGET_FPS`: Lower this if stream is unstable

## Troubleshooting

### 1) A known student appears as Unknown on one camera

- Register more diverse photos (angles, distance, lighting)
- Prefer `Register from Photos` with 20-30+ images
- Slightly lower `FACE_RECOGNITION_THRESHOLD` if too strict
- Check that the problematic camera is not low-light or blurry

### 2) FFmpeg/H.264 decode errors (macroblock/async lock)

- Usually caused by unstable RTSP transport or damaged packets
- Switch to go2rtc relay URL and test again
- Reduce source bitrate/FPS from camera app or OBS
- Try `RTSP_USE_HW_ACCEL = False` if decoder instability appears
- Keep `RTSP_TRANSPORT = "tcp"` for better reliability

### 3) Camera not opening

- Verify source URL/index in `config.py`
- Ensure no other app is locking the camera
- Test stream with `python test_rtsp.py`

### 4) Slow performance

- Lower frame size in config
- Reduce stream FPS
- Limit processing load (frame-skip settings)

## Validation Scripts

- `python test_system.py` - general system checks
- `python test_rtsp.py` - RTSP connectivity checks
- `python diagnose_embeddings.py` - embedding/gallery diagnostics

## Database Overview

Main tables:

- `students`
- `visit_logs`

Data location:

- `database/canteen.db`
- face galleries under `database/faces/`

## Future Enhancements

- Web dashboard
- Mobile registration companion
- Cloud sync and backup
- Anti-spoofing/liveness checks

## License

Educational project for B.Tech curriculum use.

## Support

For issues, open a GitHub issue in this repository.
