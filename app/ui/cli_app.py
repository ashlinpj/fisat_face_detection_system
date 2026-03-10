"""CLI Application - Command-line face detection interface.

Receives a Container for all dependencies rather than creating them directly.
All database access goes through injected repositories.
"""

import cv2
import os
import numpy as np
from datetime import datetime
from threading import Thread
import time
import queue

import config


class CanteenFaceDetectionApp:
    """CLI application. Receives a DI container for all dependencies."""

    def __init__(self, container):
        self.container = container
        self.student_repo = container.student_repo
        self.visit_repo = container.visit_repo

        print("=" * 60)
        print("  COLLEGE CANTEEN FACE DETECTION SYSTEM")
        print("  Mini Project - B.Tech")
        print("=" * 60)

        self.cap = None
        self.is_running = False
        self.frame_buffer = queue.Queue(maxsize=config.FRAME_BUFFER_SIZE)
        self.capture_thread = None
        self.capture_thread_running = False

        print("\nSystem initialized successfully!")
        print("-" * 60)

    def cleanup_camera(self, cap=None):
        """Properly release camera resources"""
        if cap is None:
            cap = self.cap
        if cap is not None:
            try:
                cap.release()
            except Exception as e:
                print(f"Error releasing camera: {e}")
        cv2.destroyAllWindows()

    def start_camera(self):
        """Start the camera capture (webcam or RTSP stream)"""
        if config.USE_RTSP:
            print(f"Connecting to RTSP stream: {config.RTSP_URL}")

            rtsp_opts = [f"{k};{v}" for k, v in config.RTSP_OPENCV_OPTIONS.items()]
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(rtsp_opts)

            for attempt in range(config.RTSP_RECONNECT_ATTEMPTS):
                self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

                if config.RTSP_TARGET_FPS:
                    self.cap.set(cv2.CAP_PROP_FPS, config.RTSP_TARGET_FPS)

                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret:
                        print(f"✓ RTSP stream connected successfully!")
                        print(f"  Stream: {config.RTSP_URL}")
                        print(f"  Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                        self._clear_frame_buffer()
                        self._start_capture_thread()
                        return True

                print(f"  Attempt {attempt + 1}/{config.RTSP_RECONNECT_ATTEMPTS} failed...")
                if attempt < config.RTSP_RECONNECT_ATTEMPTS - 1:
                    time.sleep(config.RTSP_RECONNECT_DELAY)

            self.cleanup_camera(self.cap)
            print("\nERROR: Could not connect to RTSP stream!")
            print("Please check:")
            print("  1. RTSP server is running (e.g., OBS with RTSP output)")
            print("  2. RTSP URL is correct")
            print(f"  3. Current URL: {config.RTSP_URL}")
            print("  4. Firewall is not blocking the connection")
            return False
        else:
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.FPS)

            if not self.cap.isOpened():
                self.cleanup_camera(self.cap)
                print("ERROR: Could not open camera!")
                print("Please check:")
                print("  1. Camera is connected")
                print("  2. Camera is not being used by another application")
                print(f"  3. Camera index is correct (current: {config.CAMERA_INDEX})")
                return False

            print(f"✓ Camera opened successfully (Index: {config.CAMERA_INDEX})")
            self._clear_frame_buffer()
            self._start_capture_thread()
            return True

    def stop_camera(self):
        """Stop the camera capture"""
        self._stop_capture_thread()
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def _start_capture_thread(self):
        """Continuously read frames into buffer"""
        self._stop_capture_thread()
        self.capture_thread_running = True

        def capture_loop():
            while self.capture_thread_running and self.cap:
                try:
                    if config.USE_RTSP and config.RTSP_PREGRAB_COUNT > 0:
                        for _ in range(config.RTSP_PREGRAB_COUNT):
                            self.cap.grab()

                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()
                        except queue.Empty:
                            pass

                    try:
                        self.frame_buffer.put_nowait(frame)
                    except queue.Full:
                        pass
                except Exception:
                    time.sleep(0.01)

        self.capture_thread = Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()

    def _stop_capture_thread(self):
        self.capture_thread_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        self.capture_thread = None

    def _clear_frame_buffer(self):
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break

    def _get_latest_frame(self):
        """Fetch the most recent frame"""
        try:
            frame = self.frame_buffer.get(timeout=config.FRAME_FETCH_TIMEOUT)
        except queue.Empty:
            return None

        while not self.frame_buffer.empty():
            try:
                frame = self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        return frame

    def run_detection(self):
        """Main detection loop"""
        if not self.start_camera():
            return

        self.is_running = True
        frame_count = 0
        start_time = time.time()
        fps = 0
        last_restart = time.time()
        failed_reads = 0
        max_failed_reads = 30

        print("\n" + "=" * 60)
        print("DETECTION STARTED")
        print("=" * 60)
        if config.SHOW_WINDOW:
            print("\nControls: [Q] Quit | [R] Register | [S] Stats | [L] Logs | [SPACE] Screenshot")
            print("-" * 60)

        while self.is_running:
            if config.USE_RTSP and getattr(config, 'RTSP_RESTART_INTERVAL', 0) > 0:
                if time.time() - last_restart >= config.RTSP_RESTART_INTERVAL:
                    print("\nPeriodic RTSP restart to clear artifacts...")
                    self.stop_camera()
                    time.sleep(0.2)
                    if not self.start_camera():
                        print("Failed to restart RTSP stream. Exiting...")
                        break
                    last_restart = time.time()
                    failed_reads = 0
                    continue

            frame = self._get_latest_frame()
            if frame is None:
                failed_reads += 1
                if getattr(config, 'VERBOSE_RTSP_LOGS', False):
                    print(f"Warning: Could not read frame (attempt {failed_reads})")

                if config.USE_RTSP and failed_reads >= max_failed_reads:
                    print("\nRTSP stream lost. Attempting to reconnect...")
                    self.stop_camera()
                    time.sleep(2)
                    if not self.start_camera():
                        print("Failed to reconnect. Exiting...")
                        break
                    failed_reads = 0
                    continue
                elif failed_reads >= max_failed_reads:
                    print("Too many failed reads. Exiting...")
                    break

                time.sleep(0.1)
                continue

            failed_reads = 0
            annotated_frame, recognized_people = self.container.frame_processor.process_frame(frame)

            if getattr(config, 'LOG_RECOGNITIONS', True) and recognized_people:
                names = [f"{p['name']} ({p['confidence']:.2f})" for p in recognized_people if p.get('name')]
                if names:
                    print(f"Detected: {', '.join(names)}")

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            display_frame = self._create_display_frame(annotated_frame, recognized_people, fps)
            cv2.imshow(config.WINDOW_TITLE, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                self.is_running = False
            elif key == ord('r') or key == ord('R'):
                self.register_student_interactive(frame)
            elif key == ord('s') or key == ord('S'):
                self.show_statistics()
            elif key == ord('l') or key == ord('L'):
                self.show_todays_logs()
            elif key == ord(' '):
                self.capture_screenshot(annotated_frame)

        self.cleanup_camera()
        print("Detection stopped.")

    def _create_display_frame(self, annotated_frame, recognized_people, fps):
        """Create display frame based on SHOW_WINDOW setting"""
        if config.SHOW_WINDOW:
            display_height = 600
            h, w = annotated_frame.shape[:2]
            if h > display_height:
                scale = display_height / h
                display_width = int(w * scale)
                display_frame = cv2.resize(annotated_frame, (display_width, display_height),
                                          interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = annotated_frame

            cv2.putText(display_frame, f"FPS: {fps:.1f}",
                       (display_frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display_frame, "Press Q:Quit | R:Register | S:Stats | L:Logs",
                       (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            return display_frame
        else:
            return self._create_names_only_display(recognized_people, fps)

    def _create_names_only_display(self, recognized_people, fps):
        """Create a display showing only detected names"""
        display_width, display_height = 800, 600
        display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        display_frame[:] = (30, 30, 30)

        detected_names = [
            p['name'] if (p.get('is_known') and p.get('name')) else 'Unknown Person'
            for p in recognized_people
        ]
        unique_names = list(dict.fromkeys(detected_names))

        cv2.putText(display_frame, "People Detected",
                   (display_width // 2 - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.line(display_frame, (50, 90), (display_width - 50, 90), (100, 100, 100), 2)

        if unique_names:
            y_offset = 150
            for i, name in enumerate(unique_names):
                color = (100, 255, 100) if 'Unknown' not in name else (100, 150, 255)
                cv2.putText(display_frame, f"{i+1}. {name}",
                           (100, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                y_offset += 70

                if y_offset > display_height - 120:
                    remaining = len(unique_names) - i - 1
                    if remaining > 0:
                        cv2.putText(display_frame, f"... and {remaining} more",
                                   (100, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
                    break
        else:
            cv2.putText(display_frame, "No people detected",
                       (display_width // 2 - 200, display_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)

        cv2.putText(display_frame, f"FPS: {fps:.1f} | Total: {len(unique_names)}",
                   (display_width // 2 - 120, display_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        return display_frame

    def register_student_interactive(self, frame):
        """Interactive student registration"""
        print("\n" + "=" * 40)
        print("  STUDENT REGISTRATION")
        print("=" * 40)

        cv2.imshow(config.WINDOW_TITLE, frame)

        try:
            student_id = input("Enter Student ID (e.g., 21CS001): ").strip()
            if not student_id:
                print("Registration cancelled - No ID provided")
                return

            existing = self.student_repo.get_by_id(student_id)
            if existing:
                print(f"Student {student_id} already exists!")
                update = input("Do you want to update their face data? (y/n): ").lower()
                if update != 'y':
                    return

            name = input("Enter Student Name: ").strip()
            if not name:
                print("Registration cancelled - No name provided")
                return

            department = input("Enter Department (e.g., CSE, ECE, ME): ").strip()
            year = input("Enter Year (1-4): ").strip()

            try:
                year = int(year)
                if year < 1 or year > 4:
                    year = 1
            except (ValueError, TypeError):
                year = 1

            print("\n>>> Look at the camera...")
            print(">>> Capturing in 3 seconds...")

            for i in range(3, 0, -1):
                ret, preview_frame = self.cap.read()
                if ret:
                    cv2.putText(
                        preview_frame,
                        f"Capturing in {i}...",
                        (preview_frame.shape[1]//2 - 100, preview_frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3
                    )
                    cv2.imshow(config.WINDOW_TITLE, preview_frame)
                cv2.waitKey(1000)

            ret, capture_frame = self.cap.read()
            if ret:
                success = self.container.registration_service.register_new_student(
                    capture_frame, student_id, name, department, year
                )

                if success:
                    self.container.reload_known_faces()
                    print("\n✓ Registration successful!")
                    print(f"  Student ID: {student_id}")
                    print(f"  Name: {name}")
                    print(f"  Department: {department}")
                    print(f"  Year: {year}")
                else:
                    print("\n✗ Registration failed!")
                    print("  Please ensure:")
                    print("  1. Only one person is visible")
                    print("  2. Face is clearly visible")
                    print("  3. Good lighting conditions")
            else:
                print("Error: Could not capture frame")

        except Exception as e:
            print(f"Error during registration: {e}")

        print("-" * 40)

    def show_statistics(self):
        """Show daily statistics"""
        print("\n" + "=" * 40)
        print("  TODAY'S STATISTICS")
        print("=" * 40)

        stats = self.visit_repo.get_daily_statistics()

        print(f"  Date: {stats['date']}")
        print(f"  Total Visits: {stats['total_visits']}")
        print(f"  Unique Visitors: {stats['unique_visitors']}")
        print(f"  Unknown Visitors: {stats['unknown_visitors']}")
        print(f"  Avg Duration: {stats['average_duration_minutes']} mins")

        students = self.student_repo.get_all()
        print(f"\n  Total Registered Students: {len(students)}")

        print("-" * 40)

    def show_todays_logs(self):
        """Show today's visit logs"""
        print("\n" + "=" * 60)
        print("  TODAY'S VISIT LOGS")
        print("=" * 60)

        today = datetime.now().strftime('%Y-%m-%d')
        logs = self.visit_repo.get_visit_logs(date=today)

        if not logs:
            print("  No visits recorded today.")
        else:
            print(f"  {'Time':<10} {'Student ID':<12} {'Name':<20} {'Status':<10}")
            print("  " + "-" * 54)

            for log in logs[:20]:
                entry_time = log['entry_time']
                if entry_time:
                    try:
                        time_str = datetime.fromisoformat(entry_time).strftime('%H:%M:%S')
                    except (ValueError, AttributeError):
                        time_str = entry_time[:8] if len(entry_time) >= 8 else entry_time
                else:
                    time_str = "N/A"

                student_id = log['student_id'] or 'Unknown'
                name = log['student_name'] or 'Unknown'
                status = "Known" if log['is_known'] else "Unknown"

                print(f"  {time_str:<10} {student_id:<12} {name[:18]:<20} {status:<10}")

            if len(logs) > 20:
                print(f"\n  ... and {len(logs) - 20} more entries")

        print("-" * 60)

    def capture_screenshot(self, frame):
        """Capture and save screenshot"""
        screenshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)

        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(screenshots_dir, filename)

        cv2.imwrite(filepath, frame)
        print(f"\n✓ Screenshot saved: {filename}")

    def list_students(self):
        """List all registered students"""
        print("\n" + "=" * 60)
        print("  REGISTERED STUDENTS")
        print("=" * 60)

        students = self.student_repo.get_all()

        if not students:
            print("  No students registered yet.")
        else:
            print(f"  {'ID':<12} {'Name':<25} {'Dept':<8} {'Year':<6}")
            print("  " + "-" * 53)

            for student in students:
                print(f"  {student['student_id']:<12} {student['name'][:23]:<25} "
                      f"{student['department'] or 'N/A':<8} {student['year'] or 'N/A':<6}")

        print(f"\n  Total: {len(students)} students")
        print("-" * 60)
