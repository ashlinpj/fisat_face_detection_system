"""
Main Application - College Canteen Face Detection System
Real-time face detection and recognition for tracking canteen visits
"""

import cv2
import sys
import os
from datetime import datetime
import numpy as np
from threading import Thread
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import database
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
        self.cap = None
        self.is_running = False
        
        print("\nSystem initialized successfully!")
        print("-" * 60)
    
    def start_camera(self):
        """Start the camera capture (webcam or RTSP stream)"""
        if config.USE_RTSP:
            # RTSP Stream mode
            print(f"Connecting to RTSP stream: {config.RTSP_URL}")
            
            # Set RTSP options for better performance
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            for attempt in range(config.RTSP_RECONNECT_ATTEMPTS):
                self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
                
                # Set buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
                # Set codec for better quality
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                
                if self.cap.isOpened():
                    # Test if we can read a frame
                    ret, test_frame = self.cap.read()
                    if ret:
                        print(f"✓ RTSP stream connected successfully!")
                        print(f"  Stream: {config.RTSP_URL}")
                        print(f"  Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                        return True
                
                print(f"  Attempt {attempt + 1}/{config.RTSP_RECONNECT_ATTEMPTS} failed...")
                if attempt < config.RTSP_RECONNECT_ATTEMPTS - 1:
                    time.sleep(config.RTSP_RECONNECT_DELAY)
            
            print("\nERROR: Could not connect to RTSP stream!")
            print("Please check:")
            print("  1. RTSP server is running (e.g., OBS with RTSP output)")
            print("  2. RTSP URL is correct")
            print(f"  3. Current URL: {config.RTSP_URL}")
            print("  4. Firewall is not blocking the connection")
            print("\nTip: To use webcam instead, set USE_RTSP = False in config.py")
            return False
        else:
            # Webcam mode
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
            
            if not self.cap.isOpened():
                print("ERROR: Could not open camera!")
                print("Please check:")
                print("  1. Camera is connected")
                print("  2. Camera is not being used by another application")
                print(f"  3. Camera index is correct (current: {config.CAMERA_INDEX})")
                return False
            
            print(f"✓ Camera opened successfully (Index: {config.CAMERA_INDEX})")
            return True
    
    def stop_camera(self):
        """Stop the camera capture"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def run_detection(self):
        """Main detection loop"""
        if not self.start_camera():
            return
        
        self.is_running = True
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        print("\n" + "=" * 60)
        print("DETECTION STARTED")
        print("=" * 60)
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
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                failed_reads += 1
                print(f"Warning: Could not read frame from camera (attempt {failed_reads})")
                
                # If using RTSP and multiple failures, try to reconnect
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
            
            # Reset failed reads counter on successful frame
            failed_reads = 0
            
            # Process frame
            annotated_frame, recognized_people = self.face_system.process_frame(frame)
            
            # Resize frame for display to fit on screen with high quality
            display_height = 600  # Maximum height for display
            h, w = annotated_frame.shape[:2]
            if h > display_height:
                scale = display_height / h
                display_width = int(w * scale)
                annotated_frame = cv2.resize(annotated_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Add FPS to frame
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (annotated_frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Add control hints
            cv2.putText(
                annotated_frame,
                "Press Q:Quit | R:Register | S:Stats | L:Logs",
                (10, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            
            # Display frame
            cv2.imshow(config.WINDOW_TITLE, annotated_frame)
            
            # Handle key presses
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
    
    def register_student_interactive(self, frame):
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
            
            # Capture frame for registration
            ret, capture_frame = self.cap.read()
            if ret:
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
                # Open camera for registration
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
