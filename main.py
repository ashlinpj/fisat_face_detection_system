"""
Main Application - College Canteen Face Detection System
Real-time face detection and recognition for tracking canteen visits
"""

import cv2
import sys
import os
import math
from datetime import datetime
import numpy as np
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import database
from camera_pipeline import CameraPipeline
from face_recognition_module import FaceRecognitionSystem

class CanteenFaceDetectionApp:
    def __init__(self):
        print("=" * 60)
        print("  COLLEGE CANTEEN FACE DETECTION SYSTEM")
        print("  Mini Project - B.Tech")
        print("=" * 60)
        
        # Initialize database
        print("\n[1/3] Initializing database...")
        database.init_database()
        
        # Initialize face recognition system
        print("[2/3] Loading face recognition system...")
        self.face_system = FaceRecognitionSystem()
        
        # Initialize camera
        print("[3/3] Setting up camera...")
        self.camera = CameraPipeline()
        self.is_running = False
        
        print("\nSystem initialized successfully!")
        print("-" * 60)

    @property
    def cap(self):
        return self.camera.cap

    @property
    def active_source(self):
        return self.camera.active_source

    @property
    def multi_cameras(self):
        return self.camera.multi_cameras

    def _is_rtsp_source(self, source):
        return self.camera.is_rtsp_source(source)

    def _is_network_source(self, source):
        return self.camera.is_network_source(source)

    def _normalize_source(self, source):
        return self.camera.normalize_source(source)

    def _get_camera_sources(self):
        return self.camera.get_camera_sources()

    def _is_multi_camera_enabled(self):
        return self.camera.is_multi_camera_enabled()

    def _source_label(self, source, index):
        return self.camera.source_label(source, index)

    def _open_capture_for_source(self, source):
        return self.camera.open_capture_for_source(source)

    def _new_camera_state(self, source, index):
        return self.camera.new_camera_state(source, index)

    def _start_multi_capture_thread(self, state):
        self.camera.start_multi_capture_thread(state)

    def _stop_multi_capture_thread(self, state):
        self.camera.stop_multi_capture_thread(state)

    def _get_latest_frame_from_state(self, state):
        return self.camera.get_latest_frame_from_state(state)

    def _reconnect_camera_state(self, state):
        return self.camera.reconnect_camera_state(state)

    def _build_multi_display(self, processed_items, fps):
        if config.SHOW_WINDOW:
            tile_w = 640
            tile_h = 360
            total = len(processed_items)
            cols = 1 if total == 1 else 2
            rows = int(math.ceil(total / cols))

            canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
            for idx, item in enumerate(processed_items):
                row = idx // cols
                col = idx % cols
                x0 = col * tile_w
                y0 = row * tile_h

                resized = cv2.resize(item["frame"], (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                known = sum(1 for p in item["recognized"] if p.get("is_known"))
                unknown = len(item["recognized"]) - known
                label_text = f"{item['label']} | Faces:{len(item['recognized'])} K:{known} U:{unknown}"

                cv2.putText(
                    resized,
                    label_text,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                canvas[y0:y0 + tile_h, x0:x0 + tile_w] = resized

            cv2.putText(
                canvas,
                f"Overall FPS: {fps:.1f}",
                (20, canvas.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            return canvas

        display_width = 1000
        display_height = 700
        canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        cv2.putText(
            canvas,
            "People Detected Across Cameras",
            (display_width // 2 - 300, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2,
        )
        cv2.line(canvas, (40, 70), (display_width - 40, 70), (100, 100, 100), 2)

        y = 120
        for item in processed_items:
            names = []
            for p in item["recognized"]:
                if p.get("is_known") and p.get("name"):
                    names.append(p["name"])
                elif not p.get("is_known"):
                    names.append("Unknown Person")

            unique_names = list(dict.fromkeys(names))
            text = ", ".join(unique_names) if unique_names else "No detections"

            cv2.putText(
                canvas,
                f"{item['label']}: {text}",
                (60, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (120, 255, 120),
                2,
            )
            y += 48
            if y > display_height - 80:
                break

        cv2.putText(
            canvas,
            f"FPS: {fps:.1f} | Cameras: {len(processed_items)}",
            (display_width // 2 - 170, display_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            2,
        )
        return canvas

    def _start_multi_camera(self):
        return self.camera.start_multi_camera()

    def _stop_multi_camera(self):
        self.camera.stop_multi_camera()

    def _get_registration_frame_from_multi(self):
        return self.camera.get_registration_frame_from_multi()
    
    def start_camera(self):
        return self.camera.start_camera()
    
    def stop_camera(self):
        self.camera.stop_camera()

    def _start_capture_thread(self):
        self.camera.start_capture_thread()

    def _stop_capture_thread(self):
        self.camera.stop_capture_thread()

    def _clear_frame_buffer(self):
        self.camera.clear_frame_buffer()

    def _get_latest_frame(self):
        return self.camera.get_latest_frame()
    
    def run_detection(self):
        """Main detection loop"""
        if self._is_multi_camera_enabled():
            return self.run_multi_detection()

        if not self.start_camera():
            return
        
        self.is_running = True
        frame_count = 0
        start_time = time.time()
        fps = 0
        last_restart = time.time()
        
        print("\n" + "=" * 60)
        print("DETECTION STARTED")
        print("=" * 60)
        if config.SHOW_WINDOW:
            print("\nControls:")
            print("  [Q] - Quit")
            print("  [R] - Register new student")
            print("  [S] - Show statistics")
            print("  [L] - Show today's logs")
            print("  [SPACE] - Capture screenshot")
            print("-" * 60)
        
        # Track consecutive failed reads for RTSP reconnection
        failed_reads = 0
        max_failed_reads = 30  # Try to reconnect after 30 consecutive failures
        active_is_network = self._is_network_source(self.active_source)
        
        while self.is_running:
            if active_is_network and getattr(config, 'RTSP_RESTART_INTERVAL', 0) > 0:
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
                    print(f"Warning: Could not read frame from camera (attempt {failed_reads})")
                
                # If using RTSP and multiple failures, try to reconnect
                if active_is_network and failed_reads >= max_failed_reads:
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
            
            # Reset failed reads counter on successful frame
            failed_reads = 0
            
            # Process frame
            annotated_frame, recognized_people = self.face_system.process_frame(frame)

            if getattr(config, 'LOG_RECOGNITIONS', True) and recognized_people:
                names = [f"{p['name']} ({p['confidence']:.2f})" for p in recognized_people if p.get('name')]
                if names:
                    print(f"Detected: {', '.join(names)}")
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Create display frame based on SHOW_WINDOW setting
            if config.SHOW_WINDOW:
                # Show camera feed with annotations
                display_height = 600  # Maximum height for display
                h, w = annotated_frame.shape[:2]
                if h > display_height:
                    scale = display_height / h
                    display_width = int(w * scale)
                    display_frame = cv2.resize(annotated_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
                else:
                    display_frame = annotated_frame
                
                # Add FPS to frame
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (display_frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Add control hints
                cv2.putText(
                    display_frame,
                    "Press Q:Quit | R:Register | S:Stats | L:Logs",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1
                )
                
                # Display frame
                cv2.imshow(config.WINDOW_TITLE, display_frame)
            else:
                # When SHOW_WINDOW is False, show names only (no camera feed)
                # Create a dark display with detected names in big text
                display_width = 800
                display_height = 600
                display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                display_frame[:] = (30, 30, 30)  # Dark gray background
                
                # Get list of detected names
                detected_names = []
                for p in recognized_people:
                    if p.get('is_known') and p.get('name'):
                        detected_names.append(p['name'])
                    elif not p.get('is_known'):
                        detected_names.append('Unknown Person')
                
                # Remove duplicates while preserving order
                unique_names = list(dict.fromkeys(detected_names))
                
                # Title
                cv2.putText(
                    display_frame,
                    "People Detected",
                    (display_width // 2 - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    3
                )
                
                # Draw a line under title
                cv2.line(display_frame, (50, 90), (display_width - 50, 90), (100, 100, 100), 2)
                
                # Display names in big text
                if unique_names:
                    y_offset = 150
                    for i, name in enumerate(unique_names):
                        # Alternate colors for better visibility
                        color = (100, 255, 100) if 'Unknown' not in name else (100, 150, 255)
                        
                        cv2.putText(
                            display_frame,
                            f"{i+1}. {name}",
                            (100, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            color,
                            2
                        )
                        y_offset += 70
                        
                        # If too many names, show count
                        if y_offset > display_height - 120:
                            remaining = len(unique_names) - i - 1
                            if remaining > 0:
                                cv2.putText(
                                    display_frame,
                                    f"... and {remaining} more",
                                    (100, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0,
                                    (150, 150, 150),
                                    2
                                )
                            break
                else:
                    # No people detected
                    cv2.putText(
                        display_frame,
                        "No people detected",
                        (display_width // 2 - 200, display_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (150, 150, 150),
                        2
                    )
                
                # Add FPS and count at bottom
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f} | Total: {len(unique_names)}",
                    (display_width // 2 - 120, display_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2
                )
                
                # Display the names window
                cv2.imshow(config.WINDOW_TITLE, display_frame)
            
            # Handle key presses (works for both modes)
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
            
            elif key == ord(' '):  # Spacebar
                self.capture_screenshot(annotated_frame)
        
        self.stop_camera()
        print("Detection stopped.")

    def run_multi_detection(self):
        """Detection loop for multiple camera sources."""
        if not self._start_multi_camera():
            return

        self.is_running = True
        frame_count = 0
        start_time = time.time()
        fps = 0.0

        print("\n" + "=" * 60)
        print("MULTI-CAMERA DETECTION STARTED")
        print("=" * 60)
        print(f"Active cameras: {len(self.multi_cameras)}")
        print("Controls: [Q] Quit | [R] Register | [S] Stats | [L] Logs | [SPACE] Screenshot")
        print("-" * 60)

        max_failed_reads = 30
        last_display = None

        while self.is_running:
            processed_items = []
            multi_process_every = max(1, int(getattr(config, 'MULTI_STREAM_PROCESS_EVERY_N_FRAMES', 1)))

            for state in list(self.multi_cameras):
                frame = self._get_latest_frame_from_state(state)
                if frame is None:
                    state["failed_reads"] += 1
                    if state["failed_reads"] >= max_failed_reads:
                        print(f"\nStream stalled for {state['label']}. Reconnecting...")
                        self._reconnect_camera_state(state)
                    continue

                state["failed_reads"] = 0
                state["last_raw_frame"] = frame
                state["process_counter"] = state.get("process_counter", 0) + 1
                stream_key = str(state.get("label") or state.get("source") or "default")

                should_process = (
                    state["process_counter"] % multi_process_every == 0
                    or state.get("last_processed") is None
                )

                if should_process:
                    annotated_frame, recognized_people = self.face_system.process_frame(frame, stream_id=stream_key)
                    state["last_processed"] = (annotated_frame, recognized_people)
                else:
                    annotated_frame = frame
                    last_processed = state.get("last_processed")
                    recognized_people = last_processed[1] if last_processed else []

                if should_process and getattr(config, 'LOG_RECOGNITIONS', True) and recognized_people:
                    names = [
                        f"{p['name']} ({p['confidence']:.2f})"
                        for p in recognized_people
                        if p.get('name')
                    ]
                    if names:
                        print(f"[{state['label']}] Detected: {', '.join(names)}")

                processed_items.append({
                    "label": state["label"],
                    "frame": annotated_frame,
                    "recognized": recognized_people,
                })

            if not processed_items:
                time.sleep(0.05)
                continue

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            display_frame = self._build_multi_display(processed_items, fps)
            last_display = display_frame
            cv2.imshow(config.WINDOW_TITLE + " - Multi Camera", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                self.is_running = False

            elif key == ord('r') or key == ord('R'):
                reg_frame = self._get_registration_frame_from_multi()
                if reg_frame is not None:
                    self.register_student_interactive(reg_frame, self._get_registration_frame_from_multi)
                else:
                    print("No camera frame available for registration.")

            elif key == ord('s') or key == ord('S'):
                self.show_statistics()

            elif key == ord('l') or key == ord('L'):
                self.show_todays_logs()

            elif key == ord(' '):
                if last_display is not None:
                    self.capture_screenshot(last_display)

        self._stop_multi_camera()
        cv2.destroyAllWindows()
        print("Detection stopped.")
    
    def register_student_interactive(self, frame, live_frame_getter=None):
        """Interactive student registration"""
        print("\n" + "=" * 40)
        print("  STUDENT REGISTRATION")
        print("=" * 40)
        
        # Pause the main window
        cv2.imshow(config.WINDOW_TITLE, frame)
        
        try:
            student_id = input("Enter Student ID (e.g., 21CS001): ").strip()
            if not student_id:
                print("Registration cancelled - No ID provided")
                return
            
            # Check if student already exists
            existing = database.get_student_by_id(student_id)
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
            except:
                year = 1
            
            print("\n>>> Look at the camera...")
            print(">>> Capturing in 3 seconds...")
            
            # Countdown
            for i in range(3, 0, -1):
                preview_frame = None
                if live_frame_getter is not None:
                    preview_frame = live_frame_getter()
                elif self.cap is not None:
                    ret, tmp_frame = self.cap.read()
                    if ret:
                        preview_frame = tmp_frame

                if preview_frame is not None:
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
            
            # Capture frame for registration
            capture_frame = None
            if live_frame_getter is not None:
                capture_frame = live_frame_getter()
            elif self.cap is not None:
                ret, tmp_frame = self.cap.read()
                if ret:
                    capture_frame = tmp_frame

            if capture_frame is not None:
                success = self.face_system.register_new_student(
                    capture_frame, student_id, name, department, year
                )
                
                if success:
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
        
        stats = database.get_daily_statistics()
        
        print(f"  Date: {stats['date']}")
        print(f"  Total Visits: {stats['total_visits']}")
        print(f"  Unique Visitors: {stats['unique_visitors']}")
        print(f"  Unknown Visitors: {stats['unknown_visitors']}")
        print(f"  Avg Duration: {stats['average_duration_minutes']} mins")
        
        # Show registered students count
        students = database.get_all_students()
        print(f"\n  Total Registered Students: {len(students)}")
        
        print("-" * 40)
    
    def show_todays_logs(self):
        """Show today's visit logs"""
        print("\n" + "=" * 60)
        print("  TODAY'S VISIT LOGS")
        print("=" * 60)
        
        today = datetime.now().strftime('%Y-%m-%d')
        logs = database.get_visit_logs(date=today)
        
        if not logs:
            print("  No visits recorded today.")
        else:
            print(f"  {'Time':<10} {'Student ID':<12} {'Name':<20} {'Status':<10}")
            print("  " + "-" * 54)
            
            for log in logs[:20]:  # Show last 20
                entry_time = log['entry_time']
                if entry_time:
                    try:
                        time_str = datetime.fromisoformat(entry_time).strftime('%H:%M:%S')
                    except:
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
        screenshots_dir = os.path.join(os.path.dirname(__file__), "screenshots")
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
        
        students = database.get_all_students()
        
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

def main():
    """Main entry point"""
    app = CanteenFaceDetectionApp()
    
    print("\nOptions:")
    print("  1. Start Real-time Detection")
    print("  2. Register New Student")
    print("  3. View Statistics")
    print("  4. View Today's Logs")
    print("  5. List All Students")
    print("  6. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                app.run_detection()
            elif choice == '2':
                # Open first configured camera source for registration
                if app.start_camera():
                    ret, frame = app.cap.read()
                    if ret:
                        app.register_student_interactive(frame)
                    app.stop_camera()
            elif choice == '3':
                app.show_statistics()
            elif choice == '4':
                app.show_todays_logs()
            elif choice == '5':
                app.list_students()
            elif choice == '6':
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-6.")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
