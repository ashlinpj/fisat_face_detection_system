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

class CanteenFaceDetectionGUI:
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
        """Build the visitor logging tab - Simple view without boxes"""
        # Main container
        main_frame = ttk.Frame(self.detection_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control section
        control_frame = ttk.LabelFrame(main_frame, text="System Control", padding="15")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(
            btn_frame,
            text="▶ Start Logging",
            style='Big.TButton',
            command=self.toggle_detection
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.register_btn = ttk.Button(
            btn_frame,
            text="➕ Register New Student",
            style='Big.TButton',
            command=self.open_registration_dialog
        )
        self.register_btn.pack(side=tk.LEFT, padx=5)
        
        # Status info
        status_info = ttk.Frame(control_frame)
        status_info.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(status_info, text="Status:", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.detection_status = ttk.Label(status_info, text="⚪ Stopped", font=('Segoe UI', 10))
        self.detection_status.pack(side=tk.LEFT)
        
        # Live visitor notifications
        notification_frame = ttk.LabelFrame(main_frame, text="Live Visitor Log", padding="15")
        notification_frame.pack(fill=tk.BOTH, expand=True)
        
        # Info text
        info_label = ttk.Label(
            notification_frame, 
            text="System will automatically log visitors when they enter the canteen.\nNo boxes or live feed - just clean logging.",
            font=('Segoe UI', 9),
            foreground='gray'
        )
        info_label.pack(pady=(0, 10))
        
        # Recent visitors list with scrollbar
        list_frame = ttk.Frame(notification_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.recent_listbox = tk.Listbox(
            list_frame, 
            font=('Consolas', 11),
            height=20,
            yscrollcommand=scrollbar.set
        )
        self.recent_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.recent_listbox.yview)
    
    def add_notification(self, message):
        """Add notification to the recent listbox"""
        self.recent_listbox.insert(0, message)  # Insert at top
        # Keep only last 100 entries
        if self.recent_listbox.size() > 100:
            self.recent_listbox.delete(100, tk.END)
    
    def build_students_tab(self):
        """Build the student management tab"""
        # Toolbar
        toolbar = ttk.Frame(self.students_tab)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        add_btn = ttk.Button(toolbar, text="➕ Add Student", command=self.add_student_dialog)
        add_btn.pack(side=tk.LEFT, padx=5)
        
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
    
    def start_detection(self):
        """Start visitor logging - non-blocking"""
        if self.face_system is None:
            messagebox.showerror("Error", "System not initialized yet!")
            return
        
        # Update UI immediately
        self.start_btn.config(text="⏹ Stop Logging", state='disabled')
        self.detection_status.config(text="🟡 Starting camera...", foreground="orange")
        
        # Initialize camera in background thread to prevent GUI lag
        def init_camera():
            try:
                self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                
                if not self.cap.isOpened():
                    self.root.after(0, lambda: messagebox.showerror("Error", "Could not open camera!"))
                    self.root.after(0, lambda: self.start_btn.config(state='normal'))
                    self.root.after(0, lambda: self.detection_status.config(text="🔴 Camera Error", foreground="red"))
                    return
                
                self.is_running = True
                self.root.after(0, lambda: self.start_btn.config(state='normal'))
                self.root.after(0, lambda: self.detection_status.config(text="🟢 Logging Active", foreground="green"))
                self.root.after(0, lambda: self.add_notification("System started. Monitoring for visitors..."))
                
                # Start video processing loop
                self.video_loop()
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Camera initialization failed: {e}"))
                self.root.after(0, lambda: self.start_btn.config(state='normal', text="▶ Start Logging"))
                self.root.after(0, lambda: self.detection_status.config(text="🔴 Error", foreground="red"))
        
        self.video_thread = threading.Thread(target=init_camera, daemon=True)
        self.video_thread.start()
    
    def stop_detection(self):
        """Stop visitor logging"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(text="▶ Start Logging")
        self.detection_status.config(text="⚪ Stopped", foreground="gray")
        self.add_notification("System stopped.")
    
    def video_loop(self):
        """Main video processing loop - background only, no display"""
        import time
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Store current frame for registration
                self.current_frame = frame.copy()
                
                # Process frame (recognition only - no visualization)
                try:
                    _, recognized_people = self.face_system.process_frame_silent(frame)
                    
                    # Check for new logs and notify
                    for person in recognized_people:
                        if person.get('newly_logged'):
                            student = person['student']
                            timestamp = datetime.now().strftime('%I:%M:%S %p')
                            date_str = datetime.now().strftime('%Y-%m-%d')
                            msg = f"✓ {timestamp} | {student['name']} ({student['student_id']}) | {date_str}"
                            self.root.after(0, lambda m=msg: self.add_notification(m))
                            
                except Exception as e:
                    print(f"Error processing frame: {e}")
                
                # Small delay
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Error in video loop: {e}")
                time.sleep(0.1)
                continue
    
    def open_registration_dialog(self):
        """Open dialog to register new student with multi-angle capture"""
        if not self.is_running:
            messagebox.showwarning("Warning", "Please start detection first!")
            return
        
        # First, get student details
        info_dialog = tk.Toplevel(self.root)
        info_dialog.title("Register New Student - Step 1")
        info_dialog.geometry("400x350")
        info_dialog.transient(self.root)
        info_dialog.grab_set()
        
        ttk.Label(info_dialog, text="Enter Student Details", font=('Segoe UI', 14, 'bold')).pack(pady=10)
        
        # Form fields
        form_frame = ttk.Frame(info_dialog, padding=20)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(form_frame, text="Student ID:").pack(anchor=tk.W, pady=(5, 0))
        id_entry = ttk.Entry(form_frame, width=40)
        id_entry.pack(pady=(0, 10), fill=tk.X)
        
        ttk.Label(form_frame, text="Name:").pack(anchor=tk.W, pady=(5, 0))
        name_entry = ttk.Entry(form_frame, width=40)
        name_entry.pack(pady=(0, 10), fill=tk.X)
        
        ttk.Label(form_frame, text="Department:").pack(anchor=tk.W, pady=(5, 0))
        dept_entry = ttk.Entry(form_frame, width=40)
        dept_entry.pack(pady=(0, 10), fill=tk.X)
        
        ttk.Label(form_frame, text="Year (1-4):").pack(anchor=tk.W, pady=(5, 0))
        year_var = tk.StringVar(value="1")
        year_combo = ttk.Combobox(form_frame, textvariable=year_var, values=['1', '2', '3', '4'], width=37)
        year_combo.pack(pady=(0, 10), fill=tk.X)
        
        def start_capture():
            student_id = id_entry.get().strip()
            name = name_entry.get().strip()
            department = dept_entry.get().strip()
            year = int(year_var.get())
            
            if not student_id or not name:
                messagebox.showerror("Error", "Student ID and Name are required!")
                return
            
            info_dialog.destroy()
            self.open_multi_angle_capture(student_id, name, department, year)
        
        ttk.Button(form_frame, text="Next: Capture Face ➡", command=start_capture, 
                  style='Big.TButton').pack(pady=20)
    
    def open_multi_angle_capture(self, student_id, name, department, year):
        """Multi-angle face capture dialog - similar to smartphone face recognition"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Face Capture - Multi-Angle Registration")
        dialog.geometry("800x650")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Get capture angles from config
        capture_angles = getattr(config, 'CAPTURE_ANGLES', [
            {"name": "Center", "description": "Look straight at the camera", "emoji": "😊"},
            {"name": "Turn Left", "description": "Turn your head slightly to the left", "emoji": "😏"},
            {"name": "Turn Right", "description": "Turn your head slightly to the right", "emoji": "😌"},
            {"name": "Look Up", "description": "Tilt your head slightly up", "emoji": "😄"},
            {"name": "Look Down", "description": "Tilt your head slightly down", "emoji": "🙂"}
        ])
        
        captured_data = []
        current_angle_idx = [0]  # Use list to allow modification in nested function
        is_capturing = [False]  # Prevent multiple clicks
        
        # Main container with scrollbar support
        main_container = ttk.Frame(dialog)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=5, padx=15)
        
        ttk.Label(header_frame, text=f"Registering: {name} ({student_id})", 
                 font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT)
        
        progress_label = ttk.Label(header_frame, text=f"0/{len(capture_angles)} angles", 
                                  font=('Segoe UI', 10))
        progress_label.pack(side=tk.RIGHT)
        
        # Video frame - more compact
        video_frame = ttk.LabelFrame(main_container, text="Camera", padding=5)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        video_label = ttk.Label(video_frame)
        video_label.pack()
        
        # Instruction frame - more compact
        instruction_frame = ttk.LabelFrame(main_container, text="Instructions", padding=8)
        instruction_frame.pack(fill=tk.X, padx=15, pady=5)
        
        # Horizontal layout for emoji and text
        instr_content = ttk.Frame(instruction_frame)
        instr_content.pack(fill=tk.X)
        
        emoji_label = ttk.Label(instr_content, text="", font=('Segoe UI', 32))
        emoji_label.pack(side=tk.LEFT, padx=10)
        
        text_frame = ttk.Frame(instr_content)
        text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        angle_label = ttk.Label(text_frame, text="", font=('Segoe UI', 11, 'bold'))
        angle_label.pack(anchor=tk.W)
        
        desc_label = ttk.Label(text_frame, text="", font=('Segoe UI', 9))
        desc_label.pack(anchor=tk.W)
        
        # Control buttons
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill=tk.X, padx=15, pady=10)
        
        capture_btn = ttk.Button(control_frame, text="📷 Capture This Angle", 
                                style='Big.TButton')
        capture_btn.pack(side=tk.LEFT, padx=5)
        
        finish_btn = ttk.Button(control_frame, text="✓ Finish Registration", 
                               state=tk.DISABLED, style='Big.TButton')
        finish_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(control_frame, text="Cancel", 
                               command=lambda: dialog.destroy())
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        # Update instruction display
        def update_instruction():
            if current_angle_idx[0] < len(capture_angles):
                angle = capture_angles[current_angle_idx[0]]
                emoji_label.config(text=angle['emoji'])
                angle_label.config(text=f"Angle {current_angle_idx[0] + 1}: {angle['name']}")
                desc_label.config(text=angle['description'])
            else:
                emoji_label.config(text="✅")
                angle_label.config(text="All angles captured!")
                desc_label.config(text="Click 'Finish Registration' to complete")
        
        # Video update loop
        def update_video():
            try:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                    
                    # Draw face detection rectangle
                    faces = self.face_system.detect_faces(frame)
                    if faces:
                        x1, y1, x2, y2 = faces[0]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, "Ready!", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Position Your Face", (30, 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Add angle progress
                    status_text = f"{current_angle_idx[0] + 1}/{len(capture_angles)}"
                    cv2.putText(frame, status_text, (frame.shape[1] - 80, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Convert and display - smaller size for compact window
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (560, 420))
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)
                else:
                    # Show "No camera" message
                    blank = np.zeros((420, 560, 3), dtype=np.uint8)
                    cv2.putText(blank, "Camera Not Available", (120, 210), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    img = Image.fromarray(blank)
                    imgtk = ImageTk.PhotoImage(image=img)
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)
            except Exception as e:
                print(f"Video update error: {e}")
            
            if dialog.winfo_exists():
                dialog.after(30, update_video)
        
        # Capture current angle
        def capture_angle():
            if is_capturing[0]:
                return  # Already capturing
            
            is_capturing[0] = True
            capture_btn.config(state=tk.DISABLED, text="Capturing...")
            
            # Run capture in background thread to avoid freezing UI
            def do_capture():
                try:
                    if self.current_frame is None:
                        dialog.after(0, lambda: messagebox.showerror("Error", "No camera frame available!"))
                        return
                    
                    # Copy the frame to work with
                    frame_copy = self.current_frame.copy()
                    angle = capture_angles[current_angle_idx[0]]
                    
                    # Detect and extract face
                    faces = self.face_system.detect_faces(frame_copy)
                    if not faces:
                        dialog.after(0, lambda: messagebox.showerror("Error", "No face detected! Ensure face is visible."))
                        return
                    
                    # Get face region
                    x1, y1, x2, y2 = faces[0]
                    h, w = frame_copy.shape[:2]
                    pad = int((x2 - x1) * 0.35)
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad)
                    y2 = min(h, y2 + pad)
                    
                    face_image = frame_copy[y1:y2, x1:x2]
                    
                    # Get embedding
                    print(f"Extracting embedding for angle {current_angle_idx[0] + 1}...")
                    embedding = self.face_system.get_face_embedding(face_image)
                    if embedding is None:
                        dialog.after(0, lambda: messagebox.showerror("Error", "Could not extract face features!"))
                        return
                    
                    # Save image
                    database.ensure_directories()
                    face_filename = f"{student_id}_angle{current_angle_idx[0] + 1}_{angle['name'].replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    face_path = os.path.join(config.FACES_DIR, face_filename)
                    cv2.imwrite(face_path, face_image)
                    
                    # Store capture data
                    captured_data.append({
                        'embedding': embedding,
                        'image_path': face_path,
                        'angle_description': angle['name'],
                        'order': current_angle_idx[0] + 1
                    })
                    
                    print(f"✓ Captured angle {current_angle_idx[0] + 1}/{len(capture_angles)}")
                    
                    # Update UI in main thread
                    def update_ui():
                        current_angle_idx[0] += 1
                        progress_label.config(text=f"{current_angle_idx[0]}/{len(capture_angles)} angles")
                        
                        if current_angle_idx[0] < len(capture_angles):
                            update_instruction()
                            capture_btn.config(text="📷 Capture This Angle", state=tk.NORMAL)
                        else:
                            # All angles captured
                            capture_btn.config(text="All Captured!", state=tk.DISABLED)
                            finish_btn.config(state=tk.NORMAL)
                            update_instruction()
                        
                        is_capturing[0] = False
                    
                    dialog.after(0, update_ui)
                    
                except Exception as e:
                    print(f"Capture error: {e}")
                    import traceback
                    traceback.print_exc()
                    dialog.after(0, lambda: messagebox.showerror("Error", f"Capture failed: {str(e)}"))
                    dialog.after(0, lambda: capture_btn.config(state=tk.NORMAL, text="📷 Capture This Angle"))
                    is_capturing[0] = False
            
            # Start capture in background thread
            threading.Thread(target=do_capture, daemon=True).start()
        
        # Finish registration
        def finish_registration():
            finish_btn.config(state=tk.DISABLED, text="Registering...")
            capture_btn.config(state=tk.DISABLED)
            
            def do_registration():
                try:
                    if len(captured_data) == 0:
                        dialog.after(0, lambda: messagebox.showerror("Error", "No face data captured!"))
                        dialog.after(0, lambda: finish_btn.config(state=tk.NORMAL, text="✓ Finish Registration"))
                        return
                    
                    print(f"\nRegistering {name} with {len(captured_data)} angles...")
                    
                    # Register student with multi-angle data
                    success = self.face_system.register_student_multi_angle(
                        student_id, name, department, year, captured_data
                    )
                    
                    if success:
                        dialog.after(0, lambda: messagebox.showinfo("Success", 
                                          f"Student {name} registered successfully with {len(captured_data)} face angles!"))
                        dialog.after(0, lambda: dialog.destroy())
                        self.root.after(100, self.refresh_students)
                    else:
                        dialog.after(0, lambda: messagebox.showerror("Error", "Failed to register student!"))
                        dialog.after(0, lambda: finish_btn.config(state=tk.NORMAL, text="✓ Finish Registration"))
                        dialog.after(0, lambda: capture_btn.config(state=tk.NORMAL))
                        
                except Exception as e:
                    print(f"Registration error: {e}")
                    import traceback
                    traceback.print_exc()
                    dialog.after(0, lambda: messagebox.showerror("Error", f"Registration failed: {str(e)}"))
                    dialog.after(0, lambda: finish_btn.config(state=tk.NORMAL, text="✓ Finish Registration"))
                    dialog.after(0, lambda: capture_btn.config(state=tk.NORMAL))
            
            # Run registration in background thread
            threading.Thread(target=do_registration, daemon=True).start()
        
        capture_btn.config(command=capture_angle)
        finish_btn.config(command=finish_registration)
        
        update_instruction()
        update_video()
    
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
            initialfilename=f"canteen_logs_{datetime.now().strftime('%Y%m%d')}.csv"
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
        
        stats_report = "=" * 64 + "\n"
        stats_report += "     CANTEEN FACE DETECTION - STATISTICS REPORT\n"
        stats_report += "=" * 64 + "\n"
        stats_report += f"  Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        stats_report += "=" * 64 + "\n"
        stats_report += f"  TODAY'S SUMMARY ({stats['date']})\n"
        stats_report += "=" * 64 + "\n"
        stats_report += f"  - Total Visits:        {stats['total_visits']}\n"
        stats_report += f"  - Unique Visitors:     {stats['unique_visitors']}\n"
        stats_report += f"  - Unknown Visitors:    {stats['unknown_visitors']}\n"
        stats_report += f"  - Avg Duration:        {stats['average_duration_minutes']} min\n"
        stats_report += "=" * 64 + "\n"
        stats_report += "  REGISTERED STUDENTS\n"
        stats_report += "=" * 64 + "\n"
        stats_report += f"  - Total Registered:    {len(students)}\n"
        stats_report += "=" * 64 + "\n"
        
        self.stats_text.insert(tk.END, stats_report)
    
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
        self.is_running = False
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
