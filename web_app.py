"""
Web GUI Application - College Canteen Face Detection System
Flask-based web interface mirroring the Tkinter GUI
"""

import cv2
import os
import sys
import time
import json
import csv
import io
import base64
import threading
import numpy as np
from datetime import datetime
from flask import (
    Flask, render_template, Response, request,
    jsonify, send_file, redirect, url_for
)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import database
from face_recognition_module import FaceRecognitionSystem

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state (mirrors the Tkinter app's instance variables)
# ---------------------------------------------------------------------------
face_system: FaceRecognitionSystem = None
cap = None
is_running = False
current_frame = None
video_lock = threading.Lock()
recent_detections = []          # list of dicts {time, name}
detection_info = {
    "faces": 0, "known": 0, "unknown": 0, "fps": 0.0
}
system_status = "initializing"  # initializing | ready | running | stopped | error
init_error = None
video_thread = None


def initialize_system():
    """Initialise DB + face recognition (called once on startup)."""
    global face_system, system_status, init_error
    try:
        database.init_database()
        face_system = FaceRecognitionSystem()
        system_status = "ready"
        print("[web] System initialised successfully.")
    except Exception as e:
        system_status = "error"
        init_error = str(e)
        print(f"[web] Initialisation error: {e}")


# ---------------------------------------------------------------------------
# Camera / detection helpers
# ---------------------------------------------------------------------------

def _open_camera():
    """Open webcam or RTSP stream and return VideoCapture object or None."""
    if config.USE_RTSP:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        for attempt in range(config.RTSP_RECONNECT_ATTEMPTS):
            c = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
            c.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            if c.isOpened():
                ret, _ = c.read()
                if ret:
                    return c
            if attempt < config.RTSP_RECONNECT_ATTEMPTS - 1:
                time.sleep(config.RTSP_RECONNECT_DELAY)
        return None
    else:
        c = cv2.VideoCapture(config.CAMERA_INDEX)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        return c if c.isOpened() else None


def _video_loop():
    """Background thread that reads frames, runs detection, and updates globals."""
    global cap, is_running, current_frame, detection_info, recent_detections

    frame_count = 0
    start_time = time.time()
    fps = 0.0
    failed_reads = 0
    max_failed = 30

    while is_running:
        ret, frame = cap.read()
        if not ret:
            failed_reads += 1
            if config.USE_RTSP and failed_reads >= max_failed:
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
                failed_reads = 0
                if cap.isOpened():
                    continue
                else:
                    is_running = False
                    break
            elif failed_reads >= max_failed:
                is_running = False
                break
            time.sleep(0.1)
            continue

        failed_reads = 0

        annotated, recognized = face_system.process_frame(frame)

        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        known = sum(1 for p in recognized if p.get("is_known"))
        unknown = len(recognized) - known

        # ---- build the display frame (respects SHOW_WINDOW setting) ----
        if config.SHOW_WINDOW:
            display = annotated
        else:
            # Dark canvas with detected names in big text (same as Tkinter GUI)
            dw, dh = 640, 480
            display = np.zeros((dh, dw, 3), dtype=np.uint8)
            display[:] = (30, 30, 30)

            detected_names = []
            for p in recognized:
                if p.get("is_known") and p.get("name"):
                    detected_names.append(p["name"])
                elif not p.get("is_known"):
                    detected_names.append("Unknown Person")
            unique_names = list(dict.fromkeys(detected_names))

            cv2.putText(display, "People Detected", (dw // 2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.line(display, (40, 70), (dw - 40, 70), (100, 100, 100), 2)

            if unique_names:
                y_off = 120
                for i, nm in enumerate(unique_names):
                    color = (100, 255, 100) if "Unknown" not in nm else (100, 150, 255)
                    cv2.putText(display, f"{i+1}. {nm}", (80, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    y_off += 55
                    if y_off > dh - 100:
                        remaining = len(unique_names) - i - 1
                        if remaining > 0:
                            cv2.putText(display, f"... and {remaining} more", (80, y_off),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
                        break
            else:
                cv2.putText(display, "No people detected", (dw // 2 - 150, dh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)

            cv2.putText(display, f"FPS: {fps:.1f} | Total: {len(unique_names)}",
                        (dw // 2 - 100, dh - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        with video_lock:
            current_frame = display.copy()
            detection_info = {
                "faces": len(recognized),
                "known": known,
                "unknown": unknown,
                "fps": round(fps, 1),
            }

            for p in recognized:
                if p.get("is_known") and p.get("name"):
                    name = p["name"]
                elif not p.get("is_known"):
                    name = "Unknown"
                else:
                    continue
                recent_detections.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "name": name,
                })
            # Keep only the last 50 entries
            recent_detections = recent_detections[:50]

        time.sleep(0.03)


def _generate_mjpeg():
    """Generator that yields MJPEG frames for the <img> video feed."""
    while True:
        with video_lock:
            frame = current_frame
        if frame is None:
            # Send a blank frame while camera is off
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera not started", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buf = cv2.imencode(".jpg", blank)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.5)
            continue

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Routes – pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes – video stream
# ---------------------------------------------------------------------------

@app.route("/video_feed")
def video_feed():
    return Response(_generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# Routes – detection control
# ---------------------------------------------------------------------------

@app.route("/api/detection/start", methods=["POST"])
def start_detection():
    global cap, is_running, video_thread, system_status
    if face_system is None:
        return jsonify(ok=False, error="System not initialised yet"), 503

    if is_running:
        return jsonify(ok=True, msg="Already running")

    cap = _open_camera()
    if cap is None:
        return jsonify(ok=False, error="Could not open camera / RTSP stream"), 500

    is_running = True
    system_status = "running"
    video_thread = threading.Thread(target=_video_loop, daemon=True)
    video_thread.start()
    return jsonify(ok=True)


@app.route("/api/detection/stop", methods=["POST"])
def stop_detection():
    global is_running, cap, current_frame, system_status
    is_running = False
    time.sleep(0.2)
    if cap:
        cap.release()
        cap = None
    current_frame = None
    system_status = "stopped"
    return jsonify(ok=True)


@app.route("/api/detection/info")
def detection_status():
    with video_lock:
        info = detection_info.copy()
        recent = list(recent_detections[:20])
    today = database.get_daily_statistics()
    return jsonify(
        status=system_status,
        info=info,
        recent=recent,
        today=today,
    )


# ---------------------------------------------------------------------------
# Routes – screenshot
# ---------------------------------------------------------------------------

@app.route("/api/screenshot", methods=["POST"])
def take_screenshot():
    with video_lock:
        frame = current_frame
    if frame is None:
        return jsonify(ok=False, error="No frame available"), 400

    screenshots_dir = os.path.join(os.path.dirname(__file__), "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(screenshots_dir, filename)
    cv2.imwrite(filepath, frame)
    return jsonify(ok=True, filename=filename)


# ---------------------------------------------------------------------------
# Routes – students
# ---------------------------------------------------------------------------

@app.route("/api/students")
def list_students():
    search = request.args.get("search", "").lower()
    students = database.get_all_students()
    result = []
    for s in students:
        if search and not (
            search in s["student_id"].lower()
            or search in s["name"].lower()
            or search in (s.get("department") or "").lower()
        ):
            continue
        result.append({
            "id": s["id"],
            "student_id": s["student_id"],
            "name": s["name"],
            "department": s.get("department", "N/A"),
            "year": s.get("year", "N/A"),
            "created_at": (s.get("created_at") or "N/A")[:10],
        })
    return jsonify(result)


@app.route("/api/students/<student_id>", methods=["DELETE"])
def remove_student(student_id):
    database.delete_student(student_id)
    if face_system:
        face_system.reload_known_faces()
    return jsonify(ok=True)


@app.route("/api/students/register_image", methods=["POST"])
def register_from_image():
    """Register a student from an uploaded image file."""
    if face_system is None:
        return jsonify(ok=False, error="System not initialised"), 503

    file = request.files.get("photo")
    student_id = request.form.get("student_id", "").strip()
    name = request.form.get("name", "").strip()
    department = request.form.get("department", "").strip()
    year = request.form.get("year", "1")

    if not file or not student_id or not name:
        return jsonify(ok=False, error="Photo, student_id and name are required"), 400

    # Save temp file
    tmp_path = os.path.join(config.FACES_DIR, f"_tmp_{student_id}.jpg")
    os.makedirs(config.FACES_DIR, exist_ok=True)
    file.save(tmp_path)

    try:
        success = face_system.register_from_image(
            tmp_path, student_id, name, department, int(year)
        )
    except Exception as e:
        success = False

    # Clean temp only if registration created its own copy
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except:
            pass

    if success:
        return jsonify(ok=True)
    return jsonify(ok=False, error="Registration failed – face not detected or duplicate ID"), 400


# Temporary storage for multi-pose capture session
_capture_session_lock = threading.Lock()
_capture_session_frames = []  # list of numpy frames


def _get_pose_script(include_glasses: bool = True):
    """Return the guided capture script (same as Tkinter version)."""
    script = [
        {"title": "Front - neutral (no glasses)", "tip": "Look straight ahead with a relaxed face."},
        {"title": "Front - smile", "tip": "Smile naturally while facing the camera."},
        {"title": "Left turn ~30°", "tip": "Turn your head slightly left; keep eyes on camera."},
        {"title": "Right turn ~30°", "tip": "Turn your head slightly right; keep eyes on camera."},
        {"title": "Left profile ~60°", "tip": "Turn further left so only part of the face is visible."},
        {"title": "Right profile ~60°", "tip": "Turn further right so only part of the face is visible."},
        {"title": "Chin slightly down", "tip": "Tilt your chin down a bit (as if looking at chest)."},
        {"title": "Chin slightly up", "tip": "Tilt your chin up a bit (as if looking above camera)."},
        {"title": "Bright light", "tip": "Step into brighter light facing the camera."},
        {"title": "Softer light", "tip": "Step slightly aside to introduce mild shadows."},
    ]
    if include_glasses:
        script.append({"title": "With glasses - front", "tip": "Put on glasses (if any) and face the camera."})
        script.append({"title": "With glasses - slight angle", "tip": "Glasses on; turn 20-30° to either side."})
    return script


@app.route("/api/pose_script")
def pose_script():
    """Return the pose capture script."""
    include_glasses = request.args.get("glasses", "1") == "1"
    return jsonify(_get_pose_script(include_glasses))


@app.route("/api/capture_session/start", methods=["POST"])
def capture_session_start():
    """Clear frame buffer and start a new capture session."""
    with _capture_session_lock:
        _capture_session_frames.clear()
    return jsonify(ok=True)


@app.route("/api/capture_session/snap", methods=["POST"])
def capture_session_snap():
    """Capture the current camera frame and add it to the session buffer."""
    if not is_running:
        return jsonify(ok=False, error="Camera not running"), 400
    with video_lock:
        frame = current_frame
    if frame is None:
        return jsonify(ok=False, error="No frame available"), 400
    with _capture_session_lock:
        _capture_session_frames.append(frame.copy())
        count = len(_capture_session_frames)
    return jsonify(ok=True, captured=count)


@app.route("/api/students/register_capture", methods=["POST"])
def register_from_capture():
    """Register a student using the frames collected in the capture session."""
    if face_system is None:
        return jsonify(ok=False, error="System not initialised"), 503
    if not is_running:
        return jsonify(ok=False, error="Camera not running"), 400

    data = request.get_json(force=True)
    student_id = data.get("student_id", "").strip()
    name = data.get("name", "").strip()
    department = data.get("department", "").strip()
    year = int(data.get("year", 1))

    if not student_id or not name:
        return jsonify(ok=False, error="student_id and name are required"), 400

    # Grab collected frames
    with _capture_session_lock:
        frames = list(_capture_session_frames)
        _capture_session_frames.clear()

    # Fallback: if no multi-pose session ran, grab the live frame
    if not frames:
        with video_lock:
            f = current_frame
        if f is not None:
            frames = [f.copy()]

    if not frames:
        return jsonify(ok=False, error="No frames available"), 400

    if len(frames) < 6:
        return jsonify(ok=False, error=f"Only {len(frames)} samples captured – need at least 6."), 400

    success = face_system.register_student_from_frames(
        frames, student_id, name, department, year
    )
    if success:
        return jsonify(ok=True, samples=len(frames))
    return jsonify(ok=False, error="Registration failed – ensure face is visible"), 400


# ---------------------------------------------------------------------------
# Routes – visit logs
# ---------------------------------------------------------------------------

@app.route("/api/logs")
def visit_logs():
    date = request.args.get("date") or None
    student_id = request.args.get("student_id") or None
    logs = database.get_visit_logs(date=date, student_id=student_id)
    result = []
    for log in logs:
        entry_time = (log.get("entry_time") or "N/A")[:19]
        duration = log.get("duration_minutes")
        result.append({
            "id": log["id"],
            "date": log.get("date", "N/A"),
            "entry_time": entry_time,
            "student_id": log.get("student_id", "Unknown"),
            "student_name": log.get("student_name", "Unknown"),
            "status": "Known" if log.get("is_known") else "Unknown",
            "duration": f"{duration} min" if duration else "-",
        })
    return jsonify(result)


@app.route("/api/logs/export")
def export_logs():
    date = request.args.get("date") or None
    student_id = request.args.get("student_id") or None
    logs = database.get_visit_logs(date=date, student_id=student_id)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["ID", "Date", "Entry Time", "Student ID", "Name", "Status", "Duration"])
    for log in logs:
        status = "Known" if log.get("is_known") else "Unknown"
        writer.writerow([
            log["id"],
            log.get("date", ""),
            log.get("entry_time", ""),
            log.get("student_id", ""),
            log.get("student_name", ""),
            status,
            log.get("duration_minutes", ""),
        ])

    output = io.BytesIO(buf.getvalue().encode("utf-8"))
    filename = f"canteen_logs_{datetime.now().strftime('%Y%m%d')}.csv"
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name=filename)


# ---------------------------------------------------------------------------
# Routes – statistics
# ---------------------------------------------------------------------------

@app.route("/api/statistics")
def statistics():
    stats = database.get_daily_statistics()
    students = database.get_all_students()
    return jsonify(
        total_students=len(students),
        today=stats,
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    # Initialise in background so the server starts immediately
    threading.Thread(target=initialize_system, daemon=True).start()
    print("\n" + "=" * 60)
    print("  COLLEGE CANTEEN FACE DETECTION SYSTEM  –  Web GUI")
    print("  Open  http://127.0.0.1:5000  in your browser")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
