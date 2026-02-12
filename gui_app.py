"""
GUI Application - College Canteen Face Detection System
Modern Tkinter-based graphical user interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import os
import sys
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import database
from face_recognition_module import FaceRecognitionSystem
from video_processor import get_video_processor

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
        self.current_frame = None
        self.video_thread = None
        
        # Video processor
        self.video_processor = get_video_processor()
        
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
        
        # Tab 5: Video Processing
        self.video_processing_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.video_processing_tab, text="🎥 Video Processing")
        self.build_video_processing_tab()
    
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
    
    def build_video_processing_tab(self):
        """Build the video processing tab"""
        # Control panel at top
        control_frame = ttk.Frame(self.video_processing_tab)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        self.video_upload_btn = ttk.Button(
            control_frame,
            text="📁 Upload Video",
            style='Big.TButton',
            command=self.upload_video_file
        )
        self.video_upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.video_start_btn = ttk.Button(
            control_frame,
            text="▶ Start Processing",
            style='Big.TButton',
            command=self.toggle_video_processing
        )
        self.video_start_btn.pack(side=tk.LEFT, padx=5)
        
        self.video_refresh_btn = ttk.Button(
            control_frame,
            text="🔄 Refresh Folder",
            command=self.refresh_video_queue
        )
        self.video_refresh_btn.pack(side=tk.LEFT, padx=5)
        
        self.video_open_folder_btn = ttk.Button(
            control_frame,
            text="📂 Open Videos Folder",
            command=self.open_videos_folder
        )
        self.video_open_folder_btn.pack(side=tk.LEFT, padx=5)
        
        # Status section
        status_frame = ttk.LabelFrame(self.video_processing_tab, text="Processing Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.video_status_label = ttk.Label(status_frame, text="⚪ Not Started", font=('Segoe UI', 10, 'bold'))
        self.video_status_label.pack(anchor=tk.W)
        
        self.video_current_label = ttk.Label(status_frame, text="Current Video: None")
        self.video_current_label.pack(anchor=tk.W)
        
        self.video_pending_label = ttk.Label(status_frame, text="Pending Videos: 0")
        self.video_pending_label.pack(anchor=tk.W)
        
        # Split into left (queue) and right (info) panels
        content_frame = ttk.Frame(self.video_processing_tab)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_panel = ttk.Frame(content_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Left: Video Queue List
        queue_frame = ttk.LabelFrame(left_panel, text="Video Queue", padding="5")
        queue_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_queue_listbox = tk.Listbox(queue_frame, font=('Consolas', 9))
        self.video_queue_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        queue_scrollbar = ttk.Scrollbar(queue_frame, orient=tk.VERTICAL, command=self.video_queue_listbox.yview)
        queue_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_queue_listbox.config(yscrollcommand=queue_scrollbar.set)
        
        # Right: Recognition Results
        results_frame = ttk.LabelFrame(right_panel, text="Recognition Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_results_listbox = tk.Listbox(results_frame, font=('Consolas', 9))
        self.video_results_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Statistics
        stats_frame = ttk.LabelFrame(self.video_processing_tab, text="Session Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        ttk.Label(stats_grid, text="Videos Processed:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.video_stat_processed = ttk.Label(stats_grid, text="0", font=('Segoe UI', 10, 'bold'))
        self.video_stat_processed.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(stats_grid, text="Total Faces Recognized:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.video_stat_faces = ttk.Label(stats_grid, text="0", font=('Segoe UI', 10, 'bold'))
        self.video_stat_faces.grid(row=1, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(stats_grid, text="Current Video Frames:").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        self.video_stat_frames = ttk.Label(stats_grid, text="0/0", font=('Segoe UI', 10, 'bold'))
        self.video_stat_frames.grid(row=0, column=3, sticky=tk.W, padx=10)
        
        ttk.Label(stats_grid, text="Current Video Faces:").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        self.video_stat_current_faces = ttk.Label(stats_grid, text="0 detected", font=('Segoe UI', 10, 'bold'))
        self.video_stat_current_faces.grid(row=1, column=3, sticky=tk.W, padx=10)
        
        # Setup callbacks for video processor
        self.video_processor.on_status_update = self.on_video_status_update
        self.video_processor.on_recognition_update = self.on_video_recognition_update
        self.video_processor.on_video_complete = self.on_video_complete
        
        # Initial refresh
        self.refresh_video_queue()
    
    def initialize_system(self):
        """Initialize the face recognition system"""
        self.update_status("Initializing system...", "orange")
        
        def init_thread():
            try:
                database.init_database()
                self.face_system = FaceRecognitionSystem()
                
                # Link face system to video processor
                if hasattr(self, 'video_processor') and self.video_processor:
                    self.video_processor.set_face_system(self.face_system)
                
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
    
    def start_detection(self):
        """Start live detection (webcam, RTSP, or YouTube stream)"""
        if self.face_system is None:
            messagebox.showerror("Error", "System not initialized yet!")
            return
        
        # YouTube Stream mode (checked first)
        if config.USE_YOUTUBE:
            import time
            for attempt in range(config.YOUTUBE_RECONNECT_ATTEMPTS):
                try:
                    import yt_dlp
                    
                    ydl_opts = {
                        'format': config.YOUTUBE_QUALITY,
                        'quiet': True,
                        'no_warnings': True,
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(config.YOUTUBE_URL, download=False)
                        stream_url = info['url']
                    
                    self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    
                    if self.cap.isOpened():
                        ret, test_frame = self.cap.read()
                        if ret:
                            break
                    
                    if attempt < config.YOUTUBE_RECONNECT_ATTEMPTS - 1:
                        time.sleep(config.YOUTUBE_RECONNECT_DELAY)
                        
                except Exception as e:
                    if attempt < config.YOUTUBE_RECONNECT_ATTEMPTS - 1:
                        time.sleep(config.YOUTUBE_RECONNECT_DELAY)
            
            if not self.cap or not self.cap.isOpened():
                messagebox.showerror(
                    "YouTube Error",
                    f"Could not connect to YouTube stream!\n\nURL: {config.YOUTUBE_URL}\n\n"
                    f"Please check:\n"
                    f"1. YouTube URL is correct and accessible\n"
                    f"2. Internet connection is stable\n"
                    f"3. yt-dlp is installed\n\n"
                    f"To use webcam: Set USE_YOUTUBE = False in config.py"
                )
                return
        
        elif config.USE_RTSP:
            # RTSP Stream mode
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            for attempt in range(config.RTSP_RECONNECT_ATTEMPTS):
                self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
                # Disable internal auto-exposure/gain adjustments that can degrade quality
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret:
                        break
                
                if attempt < config.RTSP_RECONNECT_ATTEMPTS - 1:
                    import time
                    time.sleep(config.RTSP_RECONNECT_DELAY)
            
            if not self.cap.isOpened():
                messagebox.showerror(
                    "RTSP Error",
                    f"Could not connect to RTSP stream!\n\nURL: {config.RTSP_URL}\n\n"
                    f"Please check:\n"
                    f"1. RTSP server is running (e.g., OBS)\n"
                    f"2. RTSP URL is correct\n"
                    f"3. Firewall is not blocking connection\n\n"
                    f"To use webcam: Set USE_RTSP = False in config.py"
                )
                return
        else:
            # Webcam mode
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera!")
                return
        
        self.is_running = True
        self.start_btn.config(text="⏹ Stop Detection")
        if config.USE_YOUTUBE:
            self.update_status(f"🟢 YouTube Stream Active", "green")
        elif config.USE_RTSP:
            self.update_status(f"🟢 RTSP Stream Active", "green")
        else:
            self.update_status("🟢 Detection Running", "green")
        
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
    
    def stop_detection(self):
        """Stop live detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(text="▶ Start Detection")
        self.update_status("⚪ Detection Stopped", "gray")
        self.video_label.config(image='', text="Camera stopped")
    
    def video_loop(self):
        """Main video processing loop"""
        import time
        frame_count = 0
        start_time = time.time()
        fps = 0
        failed_reads = 0
        max_failed_reads = 30
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                failed_reads += 1
                if config.USE_YOUTUBE and failed_reads >= max_failed_reads:
                    self.root.after(0, lambda: self.update_status("🟡 Reconnecting YouTube...", "orange"))
                    if self.cap:
                        self.cap.release()
                    time.sleep(2)
                    
                    try:
                        import yt_dlp
                        ydl_opts = {
                            'format': config.YOUTUBE_QUALITY,
                            'quiet': True,
                            'no_warnings': True,
                        }
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(config.YOUTUBE_URL, download=False)
                            stream_url = info['url']
                        
                        self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                        failed_reads = 0
                        
                        if self.cap.isOpened():
                            self.root.after(0, lambda: self.update_status("🟢 YouTube Reconnected", "green"))
                            continue
                        else:
                            self.root.after(0, lambda: messagebox.showerror("Error", "YouTube stream lost and reconnection failed!"))
                            break
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showerror("Error", f"YouTube stream error: {str(e)}"))
                        break
                
                # Try to reconnect RTSP stream
                el
                # Try to reconnect RTSP stream
                if config.USE_RTSP and failed_reads >= max_failed_reads:
                    self.root.after(0, lambda: self.update_status("🟡 Reconnecting RTSP...", "orange"))
                    if self.cap:
                        self.cap.release()
                    time.sleep(2)
                    
                    self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
                    failed_reads = 0
                    
                    if self.cap.isOpened():
                        self.root.after(0, lambda: self.update_status("🟢 RTSP Reconnected", "green"))
                        continue
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Error", "RTSP stream lost and reconnection failed!"))
                        break
                elif failed_reads >= max_failed_reads:
                    break
                
                time.sleep(0.1)
                continue
            
            failed_reads = 0
            
            self.current_frame = frame.copy()
            
            # Process frame
            annotated_frame, recognized_people = self.face_system.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Convert for display
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Resize for display - smaller size for better screen fit
            # Use INTER_LINEAR for better quality when downscaling
            display_width = 640
            h, w = annotated_frame.shape[:2]
            scale = display_width / w
            display_height = int(h * scale)
            annotated_frame = cv2.resize(annotated_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
            
            # Convert to PhotoImage with high quality
            img = Image.fromarray(annotated_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update UI
            def update_ui():
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
                
                known = sum(1 for p in recognized_people if p['is_known'])
                unknown = len(recognized_people) - known
                
                self.detected_count_label.config(text=f"Faces Detected: {len(recognized_people)}")
                self.known_count_label.config(text=f"Known: {known}")
                self.unknown_count_label.config(text=f"Unknown: {unknown}")
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                
                # Update recent detections
                for person in recognized_people:
                    if person['is_known']:
                        name = person['student']['name']
                        time_str = datetime.now().strftime('%H:%M:%S')
                        self.recent_listbox.insert(0, f"{time_str} - {name}")
                        if self.recent_listbox.size() > 20:
                            self.recent_listbox.delete(20, tk.END)
            
            self.root.after(0, update_ui)
            
            # Small delay to prevent overloading
            time.sleep(0.03)
    
    def open_registration_dialog(self):
        """Open dialog to register new student"""
        if not self.is_running:
            messagebox.showwarning("Warning", "Please start detection first!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Register New Student")
        dialog.geometry("400x300")
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
            text="We will capture ~10 guided poses (angles + glasses) for better training.",
            wraplength=340
        ).pack(pady=(8, 4))

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
            
            if not self.cap or not self.cap.isOpened():
                messagebox.showerror("Error", "Camera is not running. Start detection and try again.")
                return

            pose_script = self.get_pose_script(include_glasses=include_glasses.get())
            frames = self.capture_pose_sequence(pose_script)

            if len(frames) < 6:
                messagebox.showerror(
                    "Error",
                    "Could not capture enough samples. Please try again with clear face visibility."
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
        
        ttk.Button(dialog, text="📷 Capture & Register", command=do_register).pack(pady=20)

    def get_pose_script(self, include_glasses: bool = True):
        """Return a guided capture script covering angles and lighting."""
        script = [
            ("Front - neutral (no glasses)", "Look straight ahead with a relaxed face."),
            ("Front - smile", "Smile naturally while facing the camera."),
            ("Left turn ~30°", "Turn your head slightly left; keep eyes on camera."),
            ("Right turn ~30°", "Turn your head slightly right; keep eyes on camera."),
            ("Left profile ~60°", "Turn further left so only part of the face is visible."),
            ("Right profile ~60°", "Turn further right so only part of the face is visible."),
            ("Chin slightly down", "Tilt your chin down a bit (as if looking at chest)."),
            ("Chin slightly up", "Tilt your chin up a bit (as if looking above camera)."),
            ("Bright light", "Step into brighter light facing the camera."),
            ("Softer light", "Step slightly aside to introduce mild shadows."),
        ]

        if include_glasses:
            script.append(("With glasses - front", "Put on glasses (if any) and face the camera."))
            script.append(("With glasses - slight angle", "Glasses on; turn 20-30° to either side."))

        return script

    def capture_pose_sequence(self, pose_script):
        """Guide the user through the pose script and capture frames."""
        import time

        captured_frames = []
        total = len(pose_script)

        for idx, (title, tip) in enumerate(pose_script, start=1):
            prompt = f"Step {idx}/{total}: {title}\n\n{tip}\n\nClick OK when ready to capture."
            messagebox.showinfo("Capture Pose", prompt)

            # Small delay to let user settle after closing dialog
            time.sleep(0.4)

            frame = None
            if self.cap:
                ret, raw = self.cap.read()
                if ret:
                    frame = raw.copy()

            if frame is None and self.current_frame is not None:
                frame = self.current_frame.copy()

            if frame is None:
                messagebox.showwarning("Warning", f"Step {idx}: could not capture frame. Skipping.")
                continue

            captured_frames.append(frame)

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
    
    # Video Processing Methods
    
    def upload_video_file(self):
        """Upload a video file to the processing queue"""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        # Upload to queue
        success = self.video_processor.upload_video(filepath)
        
        if success:
            messagebox.showinfo("Success", f"Video uploaded successfully!\n\n{os.path.basename(filepath)}")
            self.refresh_video_queue()
        else:
            messagebox.showerror("Error", "Failed to upload video. Check if it's valid and not too long (max 10 minutes).")
    
    def toggle_video_processing(self):
        """Start or stop video processing"""
        if self.video_processor.is_running:
            # Stop processing
            self.video_processor.stop_processing()
            self.video_start_btn.config(text="▶ Start Processing")
            self.video_status_label.config(text="⚪ Stopped")
        else:
            # Start processing
            if self.face_system is None:
                messagebox.showerror("Error", "Face recognition system not initialized!")
                return
            
            # Link face system to video processor
            self.video_processor.set_face_system(self.face_system)
            
            # Start processing
            self.video_processor.start_processing()
            self.video_start_btn.config(text="⏹ Stop Processing")
            self.video_status_label.config(text="🟢 Processing...")
            
            # Start  status update loop
            self.update_video_status_loop()
    
    def refresh_video_queue(self):
        """Refresh the video queue list"""
        self.video_queue_listbox.delete(0, tk.END)
        
        videos = self.video_processor.scan_videos_folder()
        
        if not videos:
            self.video_queue_listbox.insert(tk.END, "  No videos in queue")
        else:
            for video_path in videos:
                filename = os.path.basename(video_path)
                self.video_queue_listbox.insert(tk.END, f"  📹 {filename}")
        
        # Update pending count
        status = self.video_processor.get_queue_status()
        self.video_pending_label.config(text=f"Pending Videos: {status['pending_videos']}")
    
    def open_videos_folder(self):
        """Open the videos folder in file explorer"""
        videos_dir = self.video_processor.videos_dir
        
        if os.path.exists(videos_dir):
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                os.startfile(videos_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", videos_dir])
            else:  # Linux
                subprocess.Popen(["xdg-open", videos_dir])
        else:
            messagebox.showwarning("Warning", f"Videos folder does not exist:\n{videos_dir}")
    
    def on_video_status_update(self, status_text):
        """Callback for video processor status updates"""
        def update():
            self.video_status_label.config(text=f"🟡 {status_text}")
            self.refresh_video_queue()
        
        self.root.after(0, update)
    
    def on_video_recognition_update(self, name, timestamp, confidence):
        """Callback for video recognition updates"""
        def update():
            result_text = f"{timestamp} - {name} ({confidence:.0%})"
            self.video_results_listbox.insert(0, result_text)
            
            # Keep only last 100 results
            if self.video_results_listbox.size() > 100:
                self.video_results_listbox.delete(tk.END)
        
        self.root.after(0, update)
    
    def on_video_complete(self, video_name, stats):
        """Callback when video processing completes"""
        def update():
            self.video_results_listbox.insert(0, "")
            self.video_results_listbox.insert(0, f"✅ Completed: {video_name}")
            self.video_results_listbox.insert(0, f"   Known: {stats['known_faces']}, Unknown: {stats['unknown_faces']}")
            self.video_results_listbox.insert(0, "")
            self.refresh_video_queue()
            self.refresh_logs()  # Refresh logs tab to show new entries
        
        self.root.after(0, update)
    
    def update_video_status_loop(self):
        """Periodically update video processing status"""
        if self.video_processor.is_running:
            status = self.video_processor.get_queue_status()
            
            # Update session stats
            self.video_stat_processed.config(text=str(status['total_processed']))
            self.video_stat_faces.config(text=str(status['total_faces_recognized']))
            
            # Update current video stats
            current_stats = status['current_stats']
            self.video_stat_frames.config(
                text=f"{current_stats['processed_frames']}/{current_stats['total_frames']}"
            )
            self.video_stat_current_faces.config(
                text=f"{current_stats['faces_detected']} detected"
            )
            
            # Update current video name
            if status['current_video']:
                self.video_current_label.config(text=f"Current Video: {status['current_video']}")
            else:
                self.video_current_label.config(text="Current Video: None")
            
            # Schedule next update
            self.root.after(1000, self.update_video_status_loop)
    
    def on_closing(self):
        """Handle window close"""
        self.is_running = False
        
        # Stop video processor
        if hasattr(self, 'video_processor') and self.video_processor:
            self.video_processor.stop_processing()
        
        if self.cap:
            self.cap.release()
        
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    app = CanteenFaceDetectionGUI()
    app.run()

if __name__ == "__main__":
    main()
