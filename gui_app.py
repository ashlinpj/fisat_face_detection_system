"""
GUI Application - College Canteen Face Detection System
Modern Tkinter-based graphical user interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import sys
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import database
from camera_pipeline import CameraPipeline
from face_recognition_module import FaceRecognitionSystem

class CanteenFaceDetectionGUI:
    def register_from_file(self):
        """Register student from image file"""
        from tkinter import simpledialog, filedialog
        filepath = filedialog.askopenfilename(
            title="Select Student Photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )
        if not filepath:
            return
        student_id = simpledialog.askstring("Student ID", "Enter Student ID:")
        if not student_id:
            return
        name = simpledialog.askstring("Name", "Enter Full Name:")
        department = simpledialog.askstring("Department", "Enter Department:")
        year = simpledialog.askinteger("Year", "Enter Year (1-4):")
        success = self.face_system.register_from_image(
            filepath, student_id, name, department, year
        )
        if success:
            messagebox.showinfo("Success", f"Registered {name}!")
        else:
            messagebox.showerror("Error", "Registration failed!")
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        
        # Set icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Initialize variables
        self.cap = None
        self.is_running = False
        self.face_system = None
        self.camera = CameraPipeline()
        self.multi_mode = False
        self.current_frame = None
        self.video_thread = None
        self.capture_thread = None
        self.capture_thread_running = False
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        self.cap_lock = threading.Lock()
        self.last_frame_ts = 0.0
        self.registration_mode = False
        self.ui_update_pending = False
        self.recent_ui_last_seen = {}
        self.recent_ui_update_ts = 0.0
        self.recent_ui_update_interval_sec = 0.5
        self.recent_ui_name_cooldown_sec = 2.0
        
        # Style configuration
        self.setup_styles()
        
        # Build UI
        self.build_ui()
        
        # Initialize system
        self.initialize_system()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom styles
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Status.TLabel', font=('Segoe UI', 10))
        style.configure('Big.TButton', font=('Segoe UI', 11), padding=10)
        style.configure('Success.TLabel', foreground='green', font=('Segoe UI', 10, 'bold'))
        style.configure('Warning.TLabel', foreground='orange', font=('Segoe UI', 10, 'bold'))
        style.configure('Danger.TLabel', foreground='red', font=('Segoe UI', 10, 'bold'))
    
    def build_ui(self):
        """Build the main user interface"""
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame, 
            text="🍽️ College Canteen Face Detection System",
            style='Title.TLabel'
        )
        title_label.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(
            header_frame,
            text="⚪ System Ready",
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Content area with notebook (tabs)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Live Detection
        self.detection_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.detection_tab, text="📹 Live Detection")
        self.build_detection_tab()
        
        # Tab 2: Student Management
        self.students_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.students_tab, text="👥 Students")
        self.build_students_tab()
        
        # Tab 3: Visit Logs
        self.logs_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_tab, text="📋 Visit Logs")
        self.build_logs_tab()
        
        # Tab 4: Statistics
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="📊 Statistics")
        self.build_stats_tab()
    
    def build_detection_tab(self):
        """Build the live detection tab"""
        # Split into left (video) and right (info) panels
        left_frame = ttk.Frame(self.detection_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(self.detection_tab, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # Video display
        video_frame = ttk.LabelFrame(left_frame, text="Camera Feed", padding="5")
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(video_frame, text="Camera not started")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(
            control_frame,
            text="▶ Start Detection",
            style='Big.TButton',
            command=self.toggle_detection
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.register_btn = ttk.Button(
            control_frame,
            text="➕ Register Face",
            style='Big.TButton',
            command=self.open_registration_dialog
        )
        self.register_btn.pack(side=tk.LEFT, padx=5)

        self.register_photos_btn = ttk.Button(
            control_frame,
            text="📁 Register from Photos",
            style='Big.TButton',
            command=self.open_photo_registration_dialog
        )
        self.register_photos_btn.pack(side=tk.LEFT, padx=5)
        
        screenshot_btn = ttk.Button(
            control_frame,
            text="📷 Screenshot",
            command=self.take_screenshot
        )
        screenshot_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Detection info
        info_frame = ttk.LabelFrame(right_frame, text="Detection Info", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.detected_count_label = ttk.Label(info_frame, text="Faces Detected: 0")
        self.detected_count_label.pack(anchor=tk.W)
        
        self.known_count_label = ttk.Label(info_frame, text="Known: 0")
        self.known_count_label.pack(anchor=tk.W)
        
        self.unknown_count_label = ttk.Label(info_frame, text="Unknown: 0")
        self.unknown_count_label.pack(anchor=tk.W)
        
        self.fps_label = ttk.Label(info_frame, text="FPS: 0")
        self.fps_label.pack(anchor=tk.W)
        
        # Recent detections
        recent_frame = ttk.LabelFrame(right_frame, text="Recent Detections", padding="10")
        recent_frame.pack(fill=tk.BOTH, expand=True)
        
        self.recent_listbox = tk.Listbox(recent_frame, font=('Consolas', 9))
        self.recent_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Today's stats
        today_frame = ttk.LabelFrame(right_frame, text="Today's Summary", padding="10")
        today_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.today_visits_label = ttk.Label(today_frame, text="Total Visits: 0")
        self.today_visits_label.pack(anchor=tk.W)
        
        self.today_unique_label = ttk.Label(today_frame, text="Unique Visitors: 0")
        self.today_unique_label.pack(anchor=tk.W)
    
    def build_students_tab(self):
        """Build the student management tab"""
        # Toolbar
        toolbar = ttk.Frame(self.students_tab)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        

        add_btn = ttk.Button(toolbar, text="➕ Add Student", command=self.add_student_dialog)
        add_btn.pack(side=tk.LEFT, padx=5)

        upload_btn = ttk.Button(toolbar, text="🖼️ Register from Image", command=self.register_from_file)
        upload_btn.pack(side=tk.LEFT, padx=5)

        bulk_btn = ttk.Button(toolbar, text="📦 Bulk Import CSV", command=self.open_bulk_registration_dialog)
        bulk_btn.pack(side=tk.LEFT, padx=5)

        refresh_btn = ttk.Button(toolbar, text="🔄 Refresh", command=self.refresh_students)
        refresh_btn.pack(side=tk.LEFT, padx=5)

        delete_btn = ttk.Button(toolbar, text="🗑️ Delete Selected", command=self.delete_student)
        delete_btn.pack(side=tk.LEFT, padx=5)
        
        # Search
        ttk.Label(toolbar, text="Search:").pack(side=tk.LEFT, padx=(20, 5))
        self.student_search_var = tk.StringVar()
        self.student_search_var.trace('w', lambda *args: self.filter_students())
        search_entry = ttk.Entry(toolbar, textvariable=self.student_search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=5)
        
        # Student table
        table_frame = ttk.Frame(self.students_tab)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('ID', 'Student ID', 'Name', 'Department', 'Year', 'Registered')
        self.students_tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        for col in columns:
            self.students_tree.heading(col, text=col)
            self.students_tree.column(col, width=100)
        
        self.students_tree.column('Name', width=200)
        self.students_tree.column('Registered', width=150)
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.students_tree.yview)
        self.students_tree.configure(yscrollcommand=scrollbar.set)
        
        self.students_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def build_logs_tab(self):
        """Build the visit logs tab"""
        # Filters
        filter_frame = ttk.Frame(self.logs_tab)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(filter_frame, text="Date:").pack(side=tk.LEFT, padx=5)
        self.log_date_var = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        date_entry = ttk.Entry(filter_frame, textvariable=self.log_date_var, width=15)
        date_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(filter_frame, text="Student ID:").pack(side=tk.LEFT, padx=(20, 5))
        self.log_student_var = tk.StringVar()
        student_entry = ttk.Entry(filter_frame, textvariable=self.log_student_var, width=15)
        student_entry.pack(side=tk.LEFT, padx=5)
        
        filter_btn = ttk.Button(filter_frame, text="🔍 Filter", command=self.refresh_logs)
        filter_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(filter_frame, text="Clear", command=self.clear_log_filters)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        export_btn = ttk.Button(filter_frame, text="📥 Export CSV", command=self.export_logs)
        export_btn.pack(side=tk.RIGHT, padx=5)
        
        # Logs table
        table_frame = ttk.Frame(self.logs_tab)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('ID', 'Date', 'Entry Time', 'Student ID', 'Name', 'Status', 'Duration')
        self.logs_tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        for col in columns:
            self.logs_tree.heading(col, text=col)
            self.logs_tree.column(col, width=100)
        
        self.logs_tree.column('Name', width=180)
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.logs_tree.yview)
        self.logs_tree.configure(yscrollcommand=scrollbar.set)
        
        self.logs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def build_stats_tab(self):
        """Build the statistics tab"""
        # Summary cards
        cards_frame = ttk.Frame(self.stats_tab)
        cards_frame.pack(fill=tk.X, pady=10)
        
        # Card 1: Total Students
        card1 = ttk.LabelFrame(cards_frame, text="Total Students", padding="20")
        card1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.stat_students = ttk.Label(card1, text="0", font=('Segoe UI', 24, 'bold'))
        self.stat_students.pack()
        
        # Card 2: Today's Visits
        card2 = ttk.LabelFrame(cards_frame, text="Today's Visits", padding="20")
        card2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.stat_today = ttk.Label(card2, text="0", font=('Segoe UI', 24, 'bold'))
        self.stat_today.pack()
        
        # Card 3: Unique Today
        card3 = ttk.LabelFrame(cards_frame, text="Unique Visitors Today", padding="20")
        card3.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.stat_unique = ttk.Label(card3, text="0", font=('Segoe UI', 24, 'bold'))
        self.stat_unique.pack()
        
        # Card 4: Unknown Today
        card4 = ttk.LabelFrame(cards_frame, text="Unknown Faces Today", padding="20")
        card4.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.stat_unknown = ttk.Label(card4, text="0", font=('Segoe UI', 24, 'bold'))
        self.stat_unknown.pack()
        
        # Refresh button
        refresh_btn = ttk.Button(self.stats_tab, text="🔄 Refresh Statistics", command=self.refresh_statistics)
        refresh_btn.pack(pady=10)
        
        # Detailed stats frame
        details_frame = ttk.LabelFrame(self.stats_tab, text="Detailed Statistics", padding="10")
        details_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.stats_text = tk.Text(details_frame, font=('Consolas', 10), height=15)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
    
    def initialize_system(self):
        """Initialize the face recognition system"""
        self.update_status("Initializing system...", "orange")
        
        def init_thread():
            try:
                database.init_database()
                self.face_system = FaceRecognitionSystem()
                self.root.after(0, lambda: self.update_status("🟢 System Ready", "green"))
                self.root.after(0, self.refresh_all_data)
            except Exception as e:
                self.root.after(0, lambda: self.update_status(f"🔴 Error: {str(e)}", "red"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to initialize system: {e}"))
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def update_status(self, text, color="black"):
        """Update status label"""
        self.status_label.config(text=text)
    
    def toggle_detection(self):
        """Start or stop detection"""
        if self.is_running:
            self.stop_detection()
        else:
            self.start_detection()

    def _open_camera_capture(self, show_errors=True):
        """Open webcam/RTSP capture without starting detection loop."""
        if config.USE_RTSP:
            rtsp_opts = [f"{k};{v}" for k, v in config.RTSP_OPENCV_OPTIONS.items()]
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(rtsp_opts)

            cap = None
            for attempt in range(config.RTSP_RECONNECT_ATTEMPTS):
                cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                if config.RTSP_TARGET_FPS:
                    cap.set(cv2.CAP_PROP_FPS, config.RTSP_TARGET_FPS)
                if getattr(config, 'RTSP_USE_HW_ACCEL', True):
                    try:
                        if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION') and hasattr(cv2, 'VIDEO_ACCELERATION_ANY'):
                            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                        if hasattr(cv2, 'CAP_PROP_HW_DEVICE'):
                            cap.set(cv2.CAP_PROP_HW_DEVICE, 0)
                    except Exception:
                        pass

                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        return cap

                if cap:
                    cap.release()
                    cap = None

                if attempt < config.RTSP_RECONNECT_ATTEMPTS - 1:
                    time.sleep(config.RTSP_RECONNECT_DELAY)

            if show_errors:
                messagebox.showerror(
                    "RTSP Error",
                    f"Could not connect to RTSP stream!\n\nURL: {config.RTSP_URL}\n\n"
                    f"Please check:\n"
                    f"1. RTSP server is running (e.g., OBS)\n"
                    f"2. RTSP URL is correct\n"
                    f"3. Firewall is not blocking connection\n\n"
                    f"To use webcam: Set USE_RTSP = False in config.py"
                )
            return None

        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not cap.isOpened():
            if show_errors:
                messagebox.showerror("Error", "Could not open camera!")
            return None

        return cap
    
    def start_detection(self):
        """Start live detection (webcam or RTSP)"""
        if self.face_system is None:
            messagebox.showerror("Error", "System not initialized yet!")
            return

        # Multi-camera mode uses independent capture threads per source via CameraPipeline.
        if self.camera.is_multi_camera_enabled():
            if not self.camera.start_multi_camera():
                messagebox.showerror("Error", "Could not open any configured camera source!")
                return

            self.multi_mode = True
            self.is_running = True
            with self.latest_frame_lock:
                self.latest_frame = None
                self.last_frame_ts = 0.0

            self.start_btn.config(text="⏹ Stop Detection")
            self.update_status(f"🟢 Multi-Camera Active ({len(self.camera.multi_cameras)} source(s))", "green")

            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            return

        self.multi_mode = False
        self.cap = self._open_camera_capture(show_errors=True)
        if self.cap is None:
            return
        
        self.is_running = True
        with self.latest_frame_lock:
            self.latest_frame = None
            self.last_frame_ts = 0.0
        self._start_capture_thread()
        self.start_btn.config(text="⏹ Stop Detection")
        if config.USE_RTSP:
            self.update_status(f"🟢 RTSP Stream Active", "green")
        else:
            self.update_status("🟢 Detection Running", "green")
        
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
    
    def stop_detection(self):
        """Stop live detection"""
        self.is_running = False
        self.ui_update_pending = False

        if self.multi_mode:
            self.camera.stop_multi_camera()
            self.multi_mode = False
        else:
            self._stop_capture_thread()
            if self.cap:
                self.cap.release()

        self.start_btn.config(text="▶ Start Detection")
        self.update_status("⚪ Detection Stopped", "gray")
        self.video_label.config(image='', text="Camera stopped")

    def _start_capture_thread(self):
        """Continuously grab frames and keep only the freshest one.
        
        Detects frozen/stuck frames and triggers auto-reconnect.
        Always drains the RTSP buffer aggressively to stay near real-time.
        """
        self._stop_capture_thread()
        self.capture_thread_running = True

        def capture_loop():
            frozen_threshold = float(getattr(config, 'RTSP_FROZEN_THRESHOLD_SEC', 2.0))
            last_good_frame_time = time.time()
            consecutive_failures = 0
            max_consecutive_failures = 15  # trigger reconnect after this many

            while self.capture_thread_running and self.cap:
                try:
                    frame = None

                    if config.USE_RTSP:
                        drain_count = int(getattr(config, 'RTSP_PREGRAB_COUNT', 2))
                        if getattr(config, 'REALTIME_ONLY_MODE', False):
                            drain_count = max(drain_count, 2)
                        if self.registration_mode:
                            drain_count = max(
                                drain_count,
                                int(getattr(config, 'REGISTRATION_PREGRAB_COUNT', 6))
                            )

                        read_burst = 1
                        if self.registration_mode:
                            read_burst = max(1, int(getattr(config, 'REGISTRATION_DRAIN_READS', 4)))

                        for _ in range(read_burst):
                            with self.cap_lock:
                                # Drain queued frames aggressively
                                for __ in range(max(0, drain_count)):
                                    self.cap.grab()
                                ret, fresh = self.cap.read()
                            if ret and fresh is not None:
                                frame = fresh
                    else:
                        with self.cap_lock:
                            ret, fresh = self.cap.read()
                        if ret:
                            frame = fresh

                    if frame is None:
                        consecutive_failures += 1
                        now = time.time()

                        # Auto-reconnect if stream appears frozen
                        if config.USE_RTSP and (
                            consecutive_failures >= max_consecutive_failures
                            or (now - last_good_frame_time) > frozen_threshold
                        ):
                            print(f"⚠ RTSP frozen for {now - last_good_frame_time:.1f}s — reconnecting...")
                            try:
                                with self.cap_lock:
                                    if self.cap:
                                        self.cap.release()
                                    self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
                                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
                                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                                    if config.RTSP_TARGET_FPS:
                                        self.cap.set(cv2.CAP_PROP_FPS, config.RTSP_TARGET_FPS)
                                consecutive_failures = 0
                                last_good_frame_time = time.time()
                            except Exception as e:
                                print(f"  Reconnect error: {e}")
                                time.sleep(1.0)
                            continue

                        time.sleep(0.005)  # Brief sleep, recover quickly
                        continue

                    # Got a good frame
                    consecutive_failures = 0
                    last_good_frame_time = time.time()

                    with self.latest_frame_lock:
                        self.latest_frame = frame
                        self.last_frame_ts = time.time()
                except Exception:
                    time.sleep(0.01)

        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()

    def _stop_capture_thread(self):
        self.capture_thread_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        self.capture_thread = None

    def _get_latest_frame(self):
        if self.multi_mode:
            for state in self.camera.multi_cameras:
                frame = state.get("last_raw_frame")
                if frame is not None:
                    return frame.copy()
            return None

        with self.latest_frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def _get_fresh_latest_frame(self, max_age_sec=0.35, include_timestamp=False):
        with self.latest_frame_lock:
            if self.latest_frame is None:
                return (None, 0.0) if include_timestamp else None
            age = time.time() - self.last_frame_ts
            if age > max_age_sec:
                return (None, self.last_frame_ts) if include_timestamp else None
            frame_copy = self.latest_frame.copy()
            if include_timestamp:
                return frame_copy, self.last_frame_ts
            return frame_copy

    def _reopen_rtsp_capture(self):
        """Reconnect RTSP capture with low-latency options."""
        if self.multi_mode:
            return False

        self._stop_capture_thread()

        rtsp_opts = [f"{k};{v}" for k, v in config.RTSP_OPENCV_OPTIONS.items()]
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(rtsp_opts)

        if self.cap:
            try:
                with self.cap_lock:
                    self.cap.release()
            except Exception:
                pass

        with self.latest_frame_lock:
            self.latest_frame = None
            self.last_frame_ts = 0.0

        with self.cap_lock:
            self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            if config.RTSP_TARGET_FPS:
                self.cap.set(cv2.CAP_PROP_FPS, config.RTSP_TARGET_FPS)
            if getattr(config, 'RTSP_USE_HW_ACCEL', True):
                try:
                    if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION') and hasattr(cv2, 'VIDEO_ACCELERATION_ANY'):
                        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                    if hasattr(cv2, 'CAP_PROP_HW_DEVICE'):
                        self.cap.set(cv2.CAP_PROP_HW_DEVICE, 0)
                except Exception:
                    pass

        opened = self.cap.isOpened()
        if opened and (self.is_running or self.registration_mode):
            self._start_capture_thread()
        return opened

    def _read_current_frame_direct(self):
        """Read a near-real-time frame directly from stream by flushing queued RTSP packets."""
        if self.multi_mode:
            # In multi-camera mode, reuse the latest rendered frame path for registration capture.
            return self._get_latest_frame()

        if not self.cap:
            return None

        drain_count = int(getattr(config, 'REGISTRATION_DIRECT_PREGRAB_COUNT', 3))
        with self.cap_lock:
            if not self.cap:
                return None
            if config.USE_RTSP:
                for _ in range(max(0, drain_count)):
                    self.cap.grab()
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def _build_multi_display_frame(self, processed_items, fps):
        """Create a single GUI frame from multiple camera outputs."""
        if config.SHOW_WINDOW:
            tile_w = max(320, int(getattr(config, 'GUI_PREVIEW_WIDTH', 640)))
            tile_h = int(tile_w * 9 / 16)
            total = len(processed_items)
            cols = 1 if total == 1 else 2
            rows = (total + cols - 1) // cols

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
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                canvas[y0:y0 + tile_h, x0:x0 + tile_w] = resized

            cv2.putText(
                canvas,
                f"FPS: {fps:.1f} | Cameras: {len(processed_items)}",
                (12, max(28, canvas.shape[0] - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        display_width = 700
        display_height = 520
        display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        display_frame[:] = (30, 30, 30)

        cv2.putText(
            display_frame,
            "People Detected Across Cameras",
            (display_width // 2 - 210, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.line(display_frame, (30, 62), (display_width - 30, 62), (100, 100, 100), 2)

        y = 104
        for item in processed_items:
            names = []
            for p in item["recognized"]:
                if p.get("is_known") and p.get("name"):
                    names.append(p["name"])
                elif not p.get("is_known"):
                    names.append("Unknown Person")
            unique_names = list(dict.fromkeys(names))
            text = ", ".join(unique_names) if unique_names else "No detections"
            text = text[:72] + "..." if len(text) > 75 else text

            cv2.putText(
                display_frame,
                f"{item['label']}: {text}",
                (38, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (120, 255, 120),
                2,
            )
            y += 44
            if y > display_height - 50:
                break

        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f} | Cameras: {len(processed_items)}",
            (display_width // 2 - 130, display_height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )
        return display_frame

    def _video_loop_multi(self):
        """Multi-camera detection loop with per-source reconnect handling."""
        frame_count = 0
        start_time = time.time()
        fps = 0.0
        max_failed_reads = 30
        multi_process_every = max(1, int(getattr(config, 'MULTI_STREAM_PROCESS_EVERY_N_FRAMES', 1)))

        while self.is_running and self.multi_mode:
            processed_items = []

            for state in list(self.camera.multi_cameras):
                frame = self.camera.get_latest_frame_from_state(state)
                if frame is None:
                    state["failed_reads"] = state.get("failed_reads", 0) + 1
                    if state["failed_reads"] >= max_failed_reads:
                        self.camera.reconnect_camera_state(state)
                        state["failed_reads"] = 0
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

                processed_items.append({
                    "label": state["label"],
                    "frame": annotated_frame,
                    "recognized": recognized_people,
                })

            if not processed_items:
                time.sleep(0.03)
                continue

            # Keep a recent frame for screenshot/registration compatibility.
            first_raw = self.camera.multi_cameras[0].get("last_raw_frame") if self.camera.multi_cameras else None
            if first_raw is not None:
                self.current_frame = first_raw.copy()
                with self.latest_frame_lock:
                    self.latest_frame = first_raw.copy()
                    self.last_frame_ts = time.time()

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            display_frame = self._build_multi_display_frame(processed_items, fps)
            aggregate_people = []
            for item in processed_items:
                aggregate_people.extend(item["recognized"])

            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            def update_ui():
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

                known = sum(1 for p in aggregate_people if p.get('is_known', False))
                unknown = len(aggregate_people) - known

                self.detected_count_label.config(text=f"Faces Detected: {len(aggregate_people)}")
                self.known_count_label.config(text=f"Known: {known}")
                self.unknown_count_label.config(text=f"Unknown: {unknown}")
                self.fps_label.config(text=f"FPS: {fps:.1f}")

                now_epoch = time.time()
                if now_epoch - self.recent_ui_update_ts >= self.recent_ui_update_interval_sec:
                    self.recent_ui_update_ts = now_epoch
                    for person in aggregate_people:
                        if not person.get('is_known', False):
                            continue
                        name = person.get('name')
                        if not name or name == 'Unknown':
                            continue

                        last_seen = self.recent_ui_last_seen.get(name, 0.0)
                        if now_epoch - last_seen < self.recent_ui_name_cooldown_sec:
                            continue

                        self.recent_ui_last_seen[name] = now_epoch
                        time_str = datetime.now().strftime('%H:%M:%S')
                        self.recent_listbox.insert(0, f"{time_str} - {name}")
                        if self.recent_listbox.size() > 20:
                            self.recent_listbox.delete(20, tk.END)

            if not self.ui_update_pending:
                self.ui_update_pending = True

                def wrapped_update_ui():
                    try:
                        update_ui()
                    except Exception as e:
                        print(f"UI update warning: {e}")
                    finally:
                        self.ui_update_pending = False

                self.root.after(0, wrapped_update_ui)

            time.sleep(0.005)
    
    def video_loop(self):
        """Main video processing loop"""
        if self.multi_mode:
            self._video_loop_multi()
            return

        frame_count = 0
        start_time = time.time()
        fps = 0
        failed_reads = 0
        max_failed_reads = 30
        process_interval = max(1, int(getattr(config, 'GUI_PROCESS_EVERY_N_FRAMES', 4)))
        recognition_interval_sec = max(0.05, float(getattr(config, 'GUI_RECOGNITION_INTERVAL_SEC', 0.25)))
        ui_target_fps = max(5, int(getattr(config, 'GUI_UI_TARGET_FPS', 50)))
        ui_period = 1.0 / ui_target_fps
        next_recognition_at = 0.0
        last_ui_update_time = 0.0
        last_annotated_frame = None
        last_recognized_people = []
        skipped_recognition_frames = 0
        last_processed_frame_ts = 0.0
        
        while self.is_running:
            # Always get the latest frame for display
            frame = self._get_latest_frame()
            
            # Track whether this is a genuinely new frame for recognition
            is_new_frame = False
            with self.latest_frame_lock:
                current_ts = self.last_frame_ts
            if current_ts != last_processed_frame_ts and current_ts > 0:
                is_new_frame = True
                last_processed_frame_ts = current_ts

            if frame is None:
                if self.registration_mode:
                    time.sleep(0.01)
                    continue

                now_check = time.time()
                # Detect frozen stream and auto-reconnect
                if config.USE_RTSP and last_processed_frame_ts > 0 and (now_check - last_processed_frame_ts) > 3.0:
                    self.root.after(0, lambda: self.update_status("🟡 Stream stalled — reconnecting...", "orange"))
                    opened = self._reopen_rtsp_capture()
                    last_processed_frame_ts = 0.0
                    failed_reads = 0
                    if opened:
                        self.root.after(0, lambda: self.update_status("🟢 RTSP Reconnected", "green"))
                    continue

                failed_reads += 1
                
                # Try to reconnect RTSP stream
                if config.USE_RTSP and failed_reads >= max_failed_reads:
                    self.root.after(0, lambda: self.update_status("🟡 Reconnecting RTSP...", "orange"))
                    time.sleep(2)
                    opened = self._reopen_rtsp_capture()
                    failed_reads = 0
                    
                    if opened:
                        self.root.after(0, lambda: self.update_status("🟢 RTSP Reconnected", "green"))
                        continue
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Error", "RTSP stream lost and reconnection failed!"))
                        break
                elif failed_reads >= max_failed_reads:
                    break
                
                time.sleep(0.005)
                continue
            
            failed_reads = 0
            
            self.current_frame = frame.copy()
            now = time.time()

            # During guided registration, prioritize smooth live preview over recognition.
            if self.registration_mode:
                frame_count += 1
                elapsed = now - start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = now
                time.sleep(0.001)
                continue
            else:
                skipped_recognition_frames += 1
                should_recognize = (
                    is_new_frame
                    and (
                        last_annotated_frame is None
                        or skipped_recognition_frames >= process_interval
                        or now >= next_recognition_at
                    )
                )

                if should_recognize:
                    # Run heavy face processing periodically to keep preview smooth.
                    annotated_frame, recognized_people = self.face_system.process_frame(frame)
                    last_annotated_frame = annotated_frame
                    last_recognized_people = recognized_people
                    skipped_recognition_frames = 0
                    next_recognition_at = now + recognition_interval_sec
                else:
                    annotated_frame = frame
                    recognized_people = last_recognized_people
            
            # Calculate FPS
            frame_count += 1
            elapsed = now - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = now

            if now - last_ui_update_time < ui_period:
                time.sleep(0.001)
                continue
            last_ui_update_time = now
            
            # Create display based on SHOW_WINDOW setting
            if config.SHOW_WINDOW:
                # Show camera feed with annotations
                # Convert for display
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display - smaller size for better screen fit
                display_width = max(320, int(getattr(config, 'GUI_PREVIEW_WIDTH', 640)))
                h, w = display_frame.shape[:2]
                scale = display_width / w
                display_height = int(h * scale)
                display_frame = cv2.resize(display_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
            else:
                # When SHOW_WINDOW is False, show names only (no camera feed)
                # Create a display with detected names in big text
                display_width = 640
                display_height = 480
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
                    (display_width // 2 - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2
                )
                
                # Draw a line under title
                cv2.line(display_frame, (40, 70), (display_width - 40, 70), (100, 100, 100), 2)
                
                # Display names in big text
                if unique_names:
                    y_offset = 120
                    for i, name in enumerate(unique_names):
                        # Alternate colors for better visibility
                        color = (100, 255, 100) if 'Unknown' not in name else (100, 150, 255)
                        
                        cv2.putText(
                            display_frame,
                            f"{i+1}. {name}",
                            (80, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            color,
                            2
                        )
                        y_offset += 55
                        
                        # If too many names, show count
                        if y_offset > display_height - 100:
                            remaining = len(unique_names) - i - 1
                            if remaining > 0:
                                cv2.putText(
                                    display_frame,
                                    f"... and {remaining} more",
                                    (80, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (150, 150, 150),
                                    2
                                )
                            break
                else:
                    # No people detected
                    cv2.putText(
                        display_frame,
                        "No people detected",
                        (display_width // 2 - 150, display_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (150, 150, 150),
                        2
                    )
                
                # Add FPS and count at bottom
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f} | Total: {len(unique_names)}",
                    (display_width // 2 - 100, display_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1
                )
            
            # Convert to PhotoImage with high quality
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update UI
            def update_ui():
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

                known = sum(1 for p in recognized_people if p.get('is_known', False))
                unknown = len(recognized_people) - known

                self.detected_count_label.config(text=f"Faces Detected: {len(recognized_people)}")
                self.known_count_label.config(text=f"Known: {known}")
                self.unknown_count_label.config(text=f"Unknown: {unknown}")
                self.fps_label.config(text=f"FPS: {fps:.1f}")

                # Throttle listbox writes; updating this every frame causes Tk lag.
                now_epoch = time.time()
                if now_epoch - self.recent_ui_update_ts >= self.recent_ui_update_interval_sec:
                    self.recent_ui_update_ts = now_epoch

                    for person in recognized_people:
                        if not person.get('is_known', False):
                            continue

                        name = person.get('name')
                        if not name or name == 'Unknown':
                            continue

                        last_seen = self.recent_ui_last_seen.get(name, 0.0)
                        if now_epoch - last_seen < self.recent_ui_name_cooldown_sec:
                            continue

                        self.recent_ui_last_seen[name] = now_epoch
                        time_str = datetime.now().strftime('%H:%M:%S')
                        self.recent_listbox.insert(0, f"{time_str} - {name}")
                        if self.recent_listbox.size() > 20:
                            self.recent_listbox.delete(20, tk.END)
            
            if not self.ui_update_pending:
                self.ui_update_pending = True

                def wrapped_update_ui():
                    try:
                        update_ui()
                    except Exception as e:
                        # Prevent Tk callback crashes from killing the live loop.
                        print(f"UI update warning: {e}")
                    finally:
                        self.ui_update_pending = False

                self.root.after(0, wrapped_update_ui)
            
            # Tiny delay keeps Tkinter responsive without heavily throttling FPS.
            # In registration mode, avoid sleep to keep preview as smooth as possible.
            if self.registration_mode:
                time.sleep(0.0)
            else:
                time.sleep(0.005)
    
    def open_registration_dialog(self):
        """Open dialog to register new student"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Register New Student")
        dialog.geometry("400x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Form fields
        ttk.Label(dialog, text="Student ID:").pack(pady=5)
        id_entry = ttk.Entry(dialog, width=30)
        id_entry.pack()
        
        ttk.Label(dialog, text="Name:").pack(pady=5)
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack()
        
        ttk.Label(dialog, text="Department:").pack(pady=5)
        dept_entry = ttk.Entry(dialog, width=30)
        dept_entry.pack()
        
        ttk.Label(dialog, text="Year (1-4):").pack(pady=5)
        year_var = tk.StringVar(value="1")
        year_combo = ttk.Combobox(dialog, textvariable=year_var, values=['1', '2', '3', '4'], width=27)
        year_combo.pack()

        ttk.Label(
            dialog,
            text="We will capture 25 guided poses (all directions) for stronger accuracy.",
            wraplength=340
        ).pack(pady=(8, 4))

        ttk.Label(
            dialog,
            text="A live preview window will open during capture so you can align your face.",
            wraplength=340
        ).pack(pady=(2, 6))

        include_glasses = tk.BooleanVar(value=True)
        ttk.Checkbutton(dialog, text="Include glasses captures (if applicable)", variable=include_glasses).pack()
        
        def do_register():
            student_id = id_entry.get().strip()
            name = name_entry.get().strip()
            department = dept_entry.get().strip()
            year = int(year_var.get())
            
            if not student_id or not name:
                messagebox.showerror("Error", "Student ID and Name are required!")
                return

            started_for_registration = False
            if self.multi_mode and self.is_running:
                # Reuse active multi-camera frames; no single-camera capture open needed.
                started_for_registration = False
            elif (not self.cap) or (not self.cap.isOpened()):
                self.cap = self._open_camera_capture(show_errors=True)
                if self.cap is None:
                    return
                started_for_registration = not self.is_running
                if started_for_registration:
                    with self.latest_frame_lock:
                        self.latest_frame = None
                        self.last_frame_ts = 0.0
                    self._start_capture_thread()

            pose_script = self.get_pose_script(include_glasses=include_glasses.get())
            frames = self.capture_pose_sequence(pose_script)

            try:
                if len(frames) < 15:
                    messagebox.showerror(
                        "Error",
                        "Could not capture enough samples. Need at least 15 clear captures."
                    )
                    return

                success = self.face_system.register_student_from_frames(
                    frames, student_id, name, department, year
                )

                if success:
                    messagebox.showinfo("Success", f"Student {name} registered with {len(frames)} samples!")
                    dialog.destroy()
                    self.refresh_students()
                else:
                    messagebox.showerror("Error", "Failed to register student. Please ensure face is visible.")
            finally:
                if started_for_registration and not self.is_running:
                    self._stop_capture_thread()
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    with self.latest_frame_lock:
                        self.latest_frame = None
                        self.last_frame_ts = 0.0
                    self.video_label.config(image='', text="Camera not started")
                    self.update_status("🟢 System Ready", "green")
        
        ttk.Button(dialog, text="📷 Capture & Register", command=do_register).pack(pady=20)

    def open_photo_registration_dialog(self):
        """Open dialog to register a student from multiple photo files."""
        if self.face_system is None:
            messagebox.showerror("Error", "System not initialized yet!")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Register from Photos")
        dialog.geometry("500x550")
        dialog.transient(self.root)
        dialog.grab_set()

        # Form fields
        ttk.Label(dialog, text="Student ID:", font=('Segoe UI', 10)).pack(pady=(15, 2))
        id_entry = ttk.Entry(dialog, width=35)
        id_entry.pack()

        ttk.Label(dialog, text="Name:", font=('Segoe UI', 10)).pack(pady=(8, 2))
        name_entry = ttk.Entry(dialog, width=35)
        name_entry.pack()

        ttk.Label(dialog, text="Department:", font=('Segoe UI', 10)).pack(pady=(8, 2))
        dept_entry = ttk.Entry(dialog, width=35)
        dept_entry.pack()

        ttk.Label(dialog, text="Year (1-4):", font=('Segoe UI', 10)).pack(pady=(8, 2))
        year_var = tk.StringVar(value="1")
        year_combo = ttk.Combobox(dialog, textvariable=year_var, values=['1', '2', '3', '4'], width=32)
        year_combo.pack()

        # Photo selection
        ttk.Separator(dialog, orient='horizontal').pack(fill='x', pady=15)

        selected_files = {'paths': []}
        file_count_var = tk.StringVar(value="No photos selected")

        def select_photos():
            paths = filedialog.askopenfilenames(
                title="Select face photos (different angles & poses)",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                    ("JPEG", "*.jpg *.jpeg"),
                    ("PNG", "*.png"),
                    ("All files", "*.*")
                ]
            )
            if paths:
                selected_files['paths'] = list(paths)
                file_count_var.set(f"✓ {len(paths)} photos selected")
                register_btn.config(state='normal')

        ttk.Button(
            dialog, text="📁 Select Photos...", command=select_photos
        ).pack(pady=(0, 5))

        ttk.Label(dialog, textvariable=file_count_var, font=('Segoe UI', 10, 'bold')).pack()

        ttk.Label(
            dialog,
            text="Tip: Select 20-30+ photos from different angles, poses,\n"
                 "and lighting conditions for best accuracy.",
            wraplength=420,
            justify='center',
            foreground='gray'
        ).pack(pady=(5, 10))

        # Progress
        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(dialog, variable=progress_var, maximum=100, length=400)
        progress_bar.pack(pady=(5, 2))
        progress_text_var = tk.StringVar(value="")
        ttk.Label(dialog, textvariable=progress_text_var).pack()

        def do_photo_register():
            student_id = id_entry.get().strip()
            name = name_entry.get().strip()
            department = dept_entry.get().strip()
            year = int(year_var.get())

            if not student_id or not name:
                messagebox.showerror("Error", "Student ID and Name are required!")
                return

            if not selected_files['paths']:
                messagebox.showerror("Error", "Please select photos first!")
                return

            register_btn.config(state='disabled')
            select_btn_ref.config(state='disabled')

            def process_in_thread():
                def progress_cb(current, total, text):
                    pct = (current / total) * 100
                    dialog.after(0, lambda: progress_var.set(pct))
                    dialog.after(0, lambda: progress_text_var.set(text))

                success = self.face_system.register_from_images(
                    selected_files['paths'],
                    student_id, name, department, year,
                    progress_callback=progress_cb
                )

                def on_complete():
                    if success:
                        messagebox.showinfo(
                            "Success",
                            f"Student {name} registered with {len(selected_files['paths'])} photos!\n\n"
                            f"Gallery-based matching is now active for this student."
                        )
                        dialog.destroy()
                        self.refresh_students()
                    else:
                        messagebox.showerror(
                            "Error",
                            "Registration failed. Please check:\n"
                            "• Photos contain clearly visible faces\n"
                            "• Only one person per photo\n"
                            "• Images are not corrupted"
                        )
                        register_btn.config(state='normal')
                        select_btn_ref.config(state='normal')

                dialog.after(0, on_complete)

            threading.Thread(target=process_in_thread, daemon=True).start()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=15)
        select_btn_ref = ttk.Button(btn_frame, text="📁 Select Photos...", command=select_photos)
        register_btn = ttk.Button(
            btn_frame, text="✓ Register from Selected Photos",
            command=do_photo_register, state='disabled'
        )
        select_btn_ref.pack(side=tk.LEFT, padx=5)
        register_btn.pack(side=tk.LEFT, padx=5)

    def open_bulk_registration_dialog(self):
        """Bulk import students from CSV with per-student photos."""
        if self.face_system is None:
            messagebox.showerror("Error", "System not initialized yet!")
            return

        if self.is_running:
            messagebox.showwarning(
                "Stop Detection",
                "Please stop live detection before bulk import to avoid performance issues."
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Bulk Register Students (CSV)")
        dialog.geometry("760x700")
        dialog.transient(self.root)
        dialog.grab_set()

        csv_path_var = tk.StringVar(value="")
        base_dir_var = tk.StringVar(value=os.getcwd())
        status_var = tk.StringVar(value="Ready")
        progress_var = tk.DoubleVar(value=0)
        importing_flag = {'running': False}

        ttk.Label(
            dialog,
            text="Import many students using one CSV + photos",
            font=('Segoe UI', 12, 'bold')
        ).pack(pady=(12, 6))

        ttk.Label(
            dialog,
            text=(
                "Required CSV columns: student_id, name, department, year\n"
                "Optional columns: photos (semicolon-separated file paths), photos_folder (folder path)\n"
                "If both optional columns are missing, importer tries <base_folder>/<student_id>/"
            ),
            justify='left',
            wraplength=710
        ).pack(padx=16, pady=(0, 10), anchor='w')

        csv_frame = ttk.LabelFrame(dialog, text="CSV File", padding=10)
        csv_frame.pack(fill='x', padx=12, pady=(0, 8))
        ttk.Entry(csv_frame, textvariable=csv_path_var, width=82).pack(side=tk.LEFT, padx=(0, 6))

        def choose_csv():
            path = filedialog.askopenfilename(
                title="Select student CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not path:
                return
            csv_path_var.set(path)
            csv_dir = os.path.dirname(path)
            if csv_dir:
                base_dir_var.set(csv_dir)

        ttk.Button(csv_frame, text="Browse...", command=choose_csv).pack(side=tk.LEFT)

        base_frame = ttk.LabelFrame(dialog, text="Base Photos Folder", padding=10)
        base_frame.pack(fill='x', padx=12, pady=(0, 8))
        ttk.Entry(base_frame, textvariable=base_dir_var, width=82).pack(side=tk.LEFT, padx=(0, 6))

        def choose_base_dir():
            path = filedialog.askdirectory(title="Select base photos folder")
            if path:
                base_dir_var.set(path)

        ttk.Button(base_frame, text="Browse...", command=choose_base_dir).pack(side=tk.LEFT)

        progress_frame = ttk.LabelFrame(dialog, text="Progress", padding=10)
        progress_frame.pack(fill='x', padx=12, pady=(0, 8))
        progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
        progress_bar.pack(fill='x')
        ttk.Label(progress_frame, textvariable=status_var).pack(anchor='w', pady=(6, 0))

        log_frame = ttk.LabelFrame(dialog, text="Import Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=12, pady=(0, 12))

        log_text = tk.Text(log_frame, height=16, wrap='word', font=('Consolas', 9), state='disabled')
        log_text.pack(side=tk.LEFT, fill='both', expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient='vertical', command=log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill='y')
        log_text.configure(yscrollcommand=log_scroll.set)

        def append_log(message: str):
            def _append():
                if not dialog.winfo_exists():
                    return
                log_text.configure(state='normal')
                log_text.insert(tk.END, message + "\n")
                log_text.see(tk.END)
                log_text.configure(state='disabled')
            dialog.after(0, _append)

        def get_row_value(row: dict, aliases):
            for alias in aliases:
                if alias in row and row[alias] is not None and str(row[alias]).strip() != "":
                    return str(row[alias]).strip()
            return ""

        def split_path_list(raw_paths: str):
            if not raw_paths:
                return []
            normalized = raw_paths.replace('|', ';').replace(',', ';')
            return [p.strip().strip('"').strip("'") for p in normalized.split(';') if p.strip()]

        def resolve_path(path_value: str, csv_dir: str, base_dir: str):
            if not path_value:
                return None
            candidate = os.path.expanduser(path_value)
            if os.path.isabs(candidate):
                return os.path.normpath(candidate)

            if base_dir:
                from_base = os.path.normpath(os.path.join(base_dir, candidate))
                if os.path.exists(from_base):
                    return from_base

            if csv_dir:
                from_csv = os.path.normpath(os.path.join(csv_dir, candidate))
                if os.path.exists(from_csv):
                    return from_csv

            # Return base-joined path as default resolution target for clearer logging.
            root = base_dir or csv_dir or os.getcwd()
            return os.path.normpath(os.path.join(root, candidate))

        def collect_images_from_folder(folder_path: str):
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            paths = []
            if not folder_path or not os.path.isdir(folder_path):
                return paths
            for root_dir, _, files in os.walk(folder_path):
                for fname in sorted(files):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in image_exts:
                        paths.append(os.path.join(root_dir, fname))
            return paths

        def do_import():
            csv_path = csv_path_var.get().strip()
            base_dir = base_dir_var.get().strip()
            if not csv_path:
                messagebox.showerror("Missing CSV", "Please select a CSV file.")
                return
            if not os.path.isfile(csv_path):
                messagebox.showerror("Invalid CSV", "Selected CSV file does not exist.")
                return

            import_btn.config(state='disabled')
            close_btn.config(state='disabled')
            importing_flag['running'] = True
            progress_var.set(0)
            status_var.set("Reading CSV...")

            def worker():
                csv_dir = os.path.dirname(csv_path)
                success_count = 0
                fail_count = 0
                skipped_count = 0

                try:
                    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                        reader = csv.DictReader(f)
                        if not reader.fieldnames:
                            raise ValueError("CSV has no header row")

                        normalized_field_map = {}
                        for field_name in reader.fieldnames:
                            if field_name is None:
                                continue
                            normalized_field_map[field_name.strip().lower()] = field_name

                        def row_key(*aliases):
                            for alias in aliases:
                                actual = normalized_field_map.get(alias)
                                if actual:
                                    return actual
                            return None

                        field_student_id = row_key('student_id', 'id', 'roll_no', 'rollno')
                        field_name = row_key('name', 'student_name', 'full_name')
                        field_department = row_key('department', 'dept')
                        field_year = row_key('year', 'class_year')
                        field_photos = row_key('photos', 'photo_paths', 'image_paths', 'images')
                        field_photos_folder = row_key('photos_folder', 'photo_folder', 'image_folder', 'folder')

                        required_missing = []
                        if not field_student_id:
                            required_missing.append('student_id')
                        if not field_name:
                            required_missing.append('name')
                        if not field_department:
                            required_missing.append('department')
                        if not field_year:
                            required_missing.append('year')

                        if required_missing:
                            raise ValueError(
                                "CSV missing required column(s): " + ", ".join(required_missing)
                            )

                        rows = list(reader)

                    if not rows:
                        raise ValueError("CSV contains no data rows")

                    total = len(rows)
                    dialog.after(0, lambda: progress_bar.configure(maximum=total))

                    for idx, row in enumerate(rows, start=1):
                        if not importing_flag['running']:
                            break

                        student_id = get_row_value(row, [field_student_id])
                        student_name = get_row_value(row, [field_name])
                        department = get_row_value(row, [field_department])
                        year_raw = get_row_value(row, [field_year])

                        try:
                            year_val = int(year_raw)
                            if year_val < 1 or year_val > 4:
                                year_val = 1
                        except Exception:
                            year_val = 1

                        if not student_id or not student_name or not department:
                            fail_count += 1
                            append_log(f"[{idx}/{total}] SKIP: Missing required values in row {idx}")
                            dialog.after(0, lambda v=idx: progress_var.set(v))
                            continue

                        photo_paths = []

                        if field_photos:
                            raw_photos = get_row_value(row, [field_photos])
                            for p in split_path_list(raw_photos):
                                resolved = resolve_path(p, csv_dir, base_dir)
                                if resolved and os.path.isfile(resolved):
                                    photo_paths.append(resolved)

                        if field_photos_folder:
                            folder_raw = get_row_value(row, [field_photos_folder])
                            folder_path = resolve_path(folder_raw, csv_dir, base_dir)
                            photo_paths.extend(collect_images_from_folder(folder_path))

                        if not photo_paths:
                            fallback_folder = os.path.join(base_dir or csv_dir or os.getcwd(), student_id)
                            photo_paths.extend(collect_images_from_folder(fallback_folder))

                        # Deduplicate while preserving order
                        seen = set()
                        deduped = []
                        for p in photo_paths:
                            norm = os.path.normcase(os.path.normpath(p))
                            if norm in seen:
                                continue
                            seen.add(norm)
                            deduped.append(p)
                        photo_paths = deduped

                        if not photo_paths:
                            fail_count += 1
                            append_log(
                                f"[{idx}/{total}] FAIL {student_id} - no valid photos found"
                            )
                            dialog.after(0, lambda v=idx: progress_var.set(v))
                            continue

                        status_msg = (
                            f"Processing {idx}/{total}: {student_id} - {student_name} "
                            f"({len(photo_paths)} photos)"
                        )
                        dialog.after(0, lambda m=status_msg: status_var.set(m))

                        try:
                            success = self.face_system.register_from_images(
                                photo_paths,
                                student_id,
                                student_name,
                                department,
                                year_val,
                                progress_callback=None,
                            )
                        except Exception as e:
                            success = False
                            append_log(
                                f"[{idx}/{total}] ERROR {student_id} - {e}"
                            )

                        if success:
                            success_count += 1
                            append_log(
                                f"[{idx}/{total}] OK   {student_id} - {student_name} ({len(photo_paths)} photos)"
                            )
                        else:
                            fail_count += 1
                            append_log(
                                f"[{idx}/{total}] FAIL {student_id} - registration failed"
                            )

                        dialog.after(0, lambda v=idx: progress_var.set(v))

                    if not importing_flag['running']:
                        skipped_count = max(0, total - int(progress_var.get()))

                except Exception as e:
                    append_log(f"ERROR: {e}")
                    status_var.set("Import failed")
                finally:
                    importing_flag['running'] = False

                    def finish_ui():
                        if not dialog.winfo_exists():
                            return
                        import_btn.config(state='normal')
                        close_btn.config(state='normal')
                        if success_count > 0:
                            self.refresh_students()

                        total_done = success_count + fail_count
                        summary = (
                            f"Import complete. Success: {success_count}, "
                            f"Failed: {fail_count}, Skipped: {skipped_count}"
                        )
                        status_var.set(summary)

                        messagebox.showinfo(
                            "Bulk Import Summary",
                            f"Processed: {total_done}\n"
                            f"Successful: {success_count}\n"
                            f"Failed: {fail_count}\n"
                            f"Skipped: {skipped_count}"
                        )

                    dialog.after(0, finish_ui)

            threading.Thread(target=worker, daemon=True).start()

        def on_close():
            if importing_flag['running']:
                if not messagebox.askyesno(
                    "Import Running",
                    "Import is still running. Stop and close this dialog?"
                ):
                    return
                importing_flag['running'] = False
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill='x', padx=12, pady=(0, 12))

        import_btn = ttk.Button(btn_frame, text="▶ Start Bulk Import", command=do_import)
        import_btn.pack(side=tk.LEFT)

        close_btn = ttk.Button(btn_frame, text="Close", command=on_close)
        close_btn.pack(side=tk.RIGHT)

        dialog.protocol("WM_DELETE_WINDOW", on_close)

    def get_pose_script(self, include_glasses: bool = True):
        """Return a guided capture script covering angles and lighting."""
        script = [
            ("Front - neutral (no glasses)", "Look straight ahead with a relaxed face."),
            ("Front - smile", "Smile naturally while facing the camera."),
            ("Front - eyes closed", "Close eyes briefly while keeping head centered."),
            ("Front - slight left tilt", "Tilt your head slightly left."),
            ("Front - slight right tilt", "Tilt your head slightly right."),
            ("Left turn ~30°", "Turn your head slightly left; keep eyes on camera."),
            ("Right turn ~30°", "Turn your head slightly right; keep eyes on camera."),
            ("Left turn ~45°", "Turn your head further left around 45 degrees."),
            ("Right turn ~45°", "Turn your head further right around 45 degrees."),
            ("Left profile ~60°", "Turn further left so only part of the face is visible."),
            ("Right profile ~60°", "Turn further right so only part of the face is visible."),
            ("Left near profile ~75°", "Turn left close to side profile."),
            ("Right near profile ~75°", "Turn right close to side profile."),
            ("Chin slightly down", "Tilt your chin down a bit (as if looking at chest)."),
            ("Chin slightly up", "Tilt your chin up a bit (as if looking above camera)."),
            ("Chin down + left", "Chin slightly down and look left."),
            ("Chin down + right", "Chin slightly down and look right."),
            ("Chin up + left", "Chin slightly up and look left."),
            ("Chin up + right", "Chin slightly up and look right."),
            ("Bright light", "Step into brighter light facing the camera."),
            ("Softer light", "Step slightly aside to introduce mild shadows."),
            ("Backlight mild", "Stand with light behind but keep face visible."),
            ("Half-face shadow left", "Keep left side slightly shadowed."),
            ("Half-face shadow right", "Keep right side slightly shadowed."),
            ("Normal blink/smile", "Blink and smile naturally once."),
        ]

        if include_glasses:
            script.append(("With glasses - front", "Put on glasses (if any) and face the camera."))
            script.append(("With glasses - slight angle", "Glasses on; turn 20-30° to either side."))
            script.append(("With glasses - left 45°", "Glasses on; turn left around 45 degrees."))
            script.append(("With glasses - right 45°", "Glasses on; turn right around 45 degrees."))

        return script

    def capture_pose_sequence(self, pose_script):
        """Guide the user through the pose script with a live preview for alignment."""
        captured_frames = []
        total = len(pose_script)
        self.registration_mode = True
        started_capture_for_registration = False

        if (not self.multi_mode) and config.USE_RTSP and getattr(config, 'FORCE_REOPEN_ON_REGISTRATION_START', True):
            self._reopen_rtsp_capture()

        # Keep a live background capture thread so preview stays smooth.
        if (not self.multi_mode) and self.cap and not self.capture_thread_running:
            self._start_capture_thread()
            started_capture_for_registration = True

        guide = tk.Toplevel(self.root)
        guide.title("Guided Pose Capture")
        guide.geometry("760x620")
        guide.transient(self.root)
        guide.grab_set()

        title_var = tk.StringVar(value="")
        tip_var = tk.StringVar(value="")
        progress_var = tk.StringVar(value="Step 1/1")
        status_var = tk.StringVar(value="Align your face and click 'Capture This Pose'.")

        ttk.Label(guide, textvariable=progress_var, font=('Segoe UI', 11, 'bold')).pack(pady=(10, 4))
        ttk.Label(guide, textvariable=title_var, font=('Segoe UI', 12, 'bold')).pack(pady=(0, 4))
        ttk.Label(guide, textvariable=tip_var, wraplength=700).pack(pady=(0, 8))

        preview_label = ttk.Label(guide, text="Live preview loading...")
        preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        controls = ttk.Frame(guide)
        controls.pack(pady=8)

        ttk.Label(guide, textvariable=status_var).pack(pady=(0, 10))

        action_var = tk.StringVar(value="")
        closed = {'value': False}
        last_preview_ts = {'value': 0.0}
        has_preview_frame = {'value': False}
        preview_w = max(320, int(getattr(config, 'REGISTRATION_PREVIEW_WIDTH', 480)))
        preview_h = max(180, int(getattr(config, 'REGISTRATION_PREVIEW_HEIGHT', 270)))

        def capture_live_frame(use_direct=False):
            frame = None
            frame_ts = time.time()

            # For preview, prefer freshest buffered frame for smoother FPS.
            if not use_direct:
                frame, frame_ts = self._get_fresh_latest_frame(
                    max_age_sec=float(getattr(config, 'REGISTRATION_MAX_FRAME_AGE_SEC', 0.35)),
                    include_timestamp=True,
                )

            # For actual capture, try a direct stream read first to avoid stale samples.
            if frame is None and use_direct:
                frame = self._read_current_frame_direct()
                frame_ts = time.time()

            if frame is None:
                frame, frame_ts = self._get_fresh_latest_frame(
                    max_age_sec=float(getattr(config, 'REGISTRATION_MAX_FRAME_AGE_SEC', 0.35)),
                    include_timestamp=True,
                )

            # Preview should stay stable even when one cycle misses freshness window.
            if frame is None:
                frame = self._get_latest_frame()
                frame_ts = time.time()

            if frame is None and self.current_frame is not None:
                frame = self.current_frame.copy()
                frame_ts = time.time()
            return frame, frame_ts

        def update_preview():
            if closed['value']:
                return

            frame, frame_ts = capture_live_frame(use_direct=False)
            if frame is not None:
                # Skip costly conversion if we already rendered this exact frame timestamp.
                if frame_ts > last_preview_ts['value']:
                    h, w = frame.shape[:2]
                    scale = min(preview_w / max(1, w), preview_h / max(1, h))
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    resized_bgr = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                    imgtk = ImageTk.PhotoImage(image=Image.fromarray(resized_rgb))
                    preview_label.imgtk = imgtk
                    preview_label.config(image=imgtk, text="")
                    last_preview_ts['value'] = frame_ts
                    has_preview_frame['value'] = True
            else:
                if not has_preview_frame['value']:
                    preview_label.config(image='', text="Waiting for live frame...")

            guide.after(int(getattr(config, 'REGISTRATION_PREVIEW_REFRESH_MS', 33)), update_preview)

        def on_capture():
            action_var.set("capture")

        def on_skip():
            action_var.set("skip")

        def on_cancel():
            action_var.set("cancel")

        ttk.Button(controls, text="📷 Capture This Pose", command=on_capture).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Skip", command=on_skip).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=6)

        guide.protocol("WM_DELETE_WINDOW", on_cancel)
        update_preview()

        for idx, (title, tip) in enumerate(pose_script, start=1):
            progress_var.set(f"Step {idx}/{total}")
            title_var.set(title)
            tip_var.set(tip)
            status_var.set("Align your face and click 'Capture This Pose'.")
            action_var.set("")

            guide.wait_variable(action_var)
            action = action_var.get()

            if action == "cancel":
                status_var.set("Capture cancelled.")
                break

            if action == "skip":
                status_var.set(f"Skipped step {idx}.")
                continue

            frame, _ = capture_live_frame(use_direct=True)
            if frame is None:
                status_var.set(f"Step {idx}: no frame available. Skipped.")
                continue

            captured_frames.append(frame)
            status_var.set(f"Captured step {idx}. Total samples: {len(captured_frames)}")

        closed['value'] = True
        if guide.winfo_exists():
            guide.destroy()

        self.registration_mode = False

        # Resume background latest-frame capture after registration.
        if (not self.multi_mode) and self.cap and (self.is_running or started_capture_for_registration):
            self._start_capture_thread()

        if (not self.multi_mode) and config.USE_RTSP and getattr(config, 'FORCE_REOPEN_ON_REGISTRATION_END', True):
            self._reopen_rtsp_capture()

        return captured_frames
    
    def add_student_dialog(self):
        """Add student from file"""
        messagebox.showinfo("Info", "Please use 'Register Face' button in Live Detection tab to add new students with face capture.")
    
    def refresh_students(self):
        """Refresh students table"""
        # Clear existing
        for item in self.students_tree.get_children():
            self.students_tree.delete(item)
        
        # Load students
        students = database.get_all_students()
        
        for student in students:
            created = student.get('created_at', 'N/A')
            if created and len(created) > 10:
                created = created[:10]
            
            self.students_tree.insert('', tk.END, values=(
                student['id'],
                student['student_id'],
                student['name'],
                student.get('department', 'N/A'),
                student.get('year', 'N/A'),
                created
            ))
    
    def filter_students(self):
        """Filter students by search term"""
        search = self.student_search_var.get().lower()
        
        for item in self.students_tree.get_children():
            self.students_tree.delete(item)
        
        students = database.get_all_students()
        
        for student in students:
            if (search in student['student_id'].lower() or 
                search in student['name'].lower() or
                search in (student.get('department', '') or '').lower()):
                
                created = student.get('created_at', 'N/A')
                if created and len(created) > 10:
                    created = created[:10]
                
                self.students_tree.insert('', tk.END, values=(
                    student['id'],
                    student['student_id'],
                    student['name'],
                    student.get('department', 'N/A'),
                    student.get('year', 'N/A'),
                    created
                ))
    
    def delete_student(self):
        """Delete selected student"""
        selected = self.students_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a student to delete!")
            return
        
        item = self.students_tree.item(selected[0])
        student_id = item['values'][1]
        name = item['values'][2]
        
        if messagebox.askyesno("Confirm", f"Delete student {name} ({student_id})?"):
            database.delete_student(student_id)
            if self.face_system:
                self.face_system.reload_known_faces()
            self.refresh_students()
            messagebox.showinfo("Success", "Student deleted successfully!")
    
    def refresh_logs(self):
        """Refresh logs table"""
        # Clear existing
        for item in self.logs_tree.get_children():
            self.logs_tree.delete(item)
        
        # Get filters
        date = self.log_date_var.get().strip() or None
        student_id = self.log_student_var.get().strip() or None
        
        # Load logs
        logs = database.get_visit_logs(date=date, student_id=student_id)
        
        for log in logs:
            entry_time = log.get('entry_time', 'N/A')
            if entry_time and len(entry_time) > 19:
                entry_time = entry_time[:19]
            
            status = "Known" if log.get('is_known') else "Unknown"
            duration = log.get('duration_minutes', '-')
            if duration and duration != '-':
                duration = f"{duration} min"
            
            self.logs_tree.insert('', tk.END, values=(
                log['id'],
                log.get('date', 'N/A'),
                entry_time,
                log.get('student_id', 'Unknown'),
                log.get('student_name', 'Unknown'),
                status,
                duration
            ))
    
    def clear_log_filters(self):
        """Clear log filters"""
        self.log_date_var.set(datetime.now().strftime('%Y-%m-%d'))
        self.log_student_var.set('')
        self.refresh_logs()
    
    def export_logs(self):
        """Export logs to CSV"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"canteen_logs_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        
        if filepath:
            date = self.log_date_var.get().strip() or None
            student_id = self.log_student_var.get().strip() or None
            logs = database.get_visit_logs(date=date, student_id=student_id)
            
            with open(filepath, 'w') as f:
                f.write("ID,Date,Entry Time,Student ID,Name,Status,Duration\n")
                for log in logs:
                    status = "Known" if log.get('is_known') else "Unknown"
                    f.write(f"{log['id']},{log.get('date', '')},{log.get('entry_time', '')},"
                           f"{log.get('student_id', '')},{log.get('student_name', '')},"
                           f"{status},{log.get('duration_minutes', '')}\n")
            
            messagebox.showinfo("Success", f"Logs exported to {filepath}")
    
    def refresh_statistics(self):
        """Refresh statistics"""
        stats = database.get_daily_statistics()
        students = database.get_all_students()
        
        self.stat_students.config(text=str(len(students)))
        self.stat_today.config(text=str(stats['total_visits']))
        self.stat_unique.config(text=str(stats['unique_visitors']))
        self.stat_unknown.config(text=str(stats['unknown_visitors']))
        
        # Update detailed stats
        self.stats_text.delete(1.0, tk.END)
        
        stats_report = f"""
╔════════════════════════════════════════════════════════════╗
║           CANTEEN FACE DETECTION - STATISTICS REPORT       ║
╠════════════════════════════════════════════════════════════╣
║  Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<39}║
╠════════════════════════════════════════════════════════════╣
║  TODAY'S SUMMARY ({stats['date']})                          ║
╠════════════════════════════════════════════════════════════╣
║  • Total Visits:        {stats['total_visits']:<35}║
║  • Unique Visitors:     {stats['unique_visitors']:<35}║
║  • Unknown Visitors:    {stats['unknown_visitors']:<35}║
║  • Avg Duration:        {stats['average_duration_minutes']:<35}min ║
╠════════════════════════════════════════════════════════════╣
║  REGISTERED STUDENTS                                       ║
╠════════════════════════════════════════════════════════════╣
║  • Total Registered:    {len(students):<35}║
╚════════════════════════════════════════════════════════════╝
"""
        self.stats_text.insert(tk.END, stats_report)
        
        # Update today's summary in detection tab
        self.today_visits_label.config(text=f"Total Visits: {stats['total_visits']}")
        self.today_unique_label.config(text=f"Unique Visitors: {stats['unique_visitors']}")
    
    def refresh_all_data(self):
        """Refresh all data in all tabs"""
        self.refresh_students()
        self.refresh_logs()
        self.refresh_statistics()
    
    def take_screenshot(self):
        """Take a screenshot of current frame"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame to capture!")
            return
        
        screenshots_dir = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(screenshots_dir, filename)
        
        cv2.imwrite(filepath, self.current_frame)
        messagebox.showinfo("Success", f"Screenshot saved: {filename}")
    
    def on_closing(self):
        """Handle window close"""
        if self.is_running:
            self.stop_detection()
        else:
            self._stop_capture_thread()
            if self.cap:
                self.cap.release()
            self.camera.stop_multi_camera()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    app = CanteenFaceDetectionGUI()
    app.run()

if __name__ == "__main__":
    main()
