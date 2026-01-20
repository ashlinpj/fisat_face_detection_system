"""
GUI Application - College Canteen Face Detection System
Modern Tkinter-based graphical user interface
Includes Multi-Image Upload & Training Integration
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import os
import sys
import shutil
from datetime import datetime
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Project Modules
import config
import database
import train_model
from face_recognition_module import FaceRecognitionSystem
# Try importing utils for export, fail gracefully if missing
try:
    from utils import export_logs_to_csv
except ImportError:
    export_logs_to_csv = None

class RegistrationDialog(tk.Toplevel):
    """
    Custom Dialog for Registering Students
    Supports both Webcam Capture and Multi-File Upload
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Register New Student")
        self.geometry("450x600")
        self.configure(bg="#f0f0f0")
        self.transient(parent)
        self.grab_set()
        
        self.result = None
        self.mode = None # 'webcam' or 'upload'
        self.uploaded_files = []
        
        # Style
        style = ttk.Style()
        style.configure("Reg.TLabel", font=("Segoe UI", 10))
        
        # --- Form Container ---
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # --- Fields ---
        ttk.Label(main_frame, text="Student ID (Unique):", style="Reg.TLabel").pack(fill="x", pady=2)
        self.id_entry = ttk.Entry(main_frame)
        self.id_entry.pack(fill="x", pady=5)
        
        ttk.Label(main_frame, text="Full Name:", style="Reg.TLabel").pack(fill="x", pady=2)
        self.name_entry = ttk.Entry(main_frame)
        self.name_entry.pack(fill="x", pady=5)
        
        ttk.Label(main_frame, text="Department:", style="Reg.TLabel").pack(fill="x", pady=2)
        self.dept_combo = ttk.Combobox(main_frame, values=["CSE", "ECE", "EEE", "ME", "CE", "AI&DS", "Robotics", "General"])
        self.dept_combo.pack(fill="x", pady=5)
        self.dept_combo.current(0)
        
        ttk.Label(main_frame, text="Year:", style="Reg.TLabel").pack(fill="x", pady=2)
        self.year_combo = ttk.Combobox(main_frame, values=["1", "2", "3", "4"])
        self.year_combo.pack(fill="x", pady=5)
        self.year_combo.current(0)
        
        # --- Divider ---
        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=20)
        
        # --- Action Buttons ---
        ttk.Label(main_frame, text="Choose Registration Method:", font=("Segoe UI", 10, "bold")).pack(pady=5)

        # 1. Webcam Button
        self.btn_capture = tk.Button(main_frame, text="📷 Capture from Webcam", 
                                   bg="#27ae60", fg="white", font=("Segoe UI", 11),
                                   command=self.use_webcam)
        self.btn_capture.pack(fill="x", pady=5)
        
        tk.Label(main_frame, text="- OR -", bg="#f0f0f0", fg="#7f8c8d").pack(pady=2)
        
        # 2. Upload Button
        self.btn_upload = tk.Button(main_frame, text="📂 Upload Photos (Multiple)", 
                                  bg="#2980b9", fg="white", font=("Segoe UI", 11),
                                  command=self.choose_files)
        self.btn_upload.pack(fill="x", pady=5)
        
        # Status Label
        self.lbl_status = tk.Label(main_frame, text="", bg="#f0f0f0", fg="#e67e22")
        self.lbl_status.pack(pady=10)

    def validate_inputs(self):
        if not self.id_entry.get().strip() or not self.name_entry.get().strip():
            messagebox.showerror("Error", "Student ID and Name are required!")
            return False
        return True

    def use_webcam(self):
        if self.validate_inputs():
            self.mode = 'webcam'
            self.save_data()

    def choose_files(self):
        if not self.validate_inputs():
            return
            
        files = filedialog.askopenfilenames(
            parent=self,
            title="Select Student Photos (Front, Side, etc.)",
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if files:
            self.uploaded_files = list(files)
            self.lbl_status.config(text=f"✅ Selected {len(files)} files")
            self.mode = 'upload'
            
            # Show Confirm Button if files selected
            if not hasattr(self, 'btn_confirm'):
                self.btn_confirm = tk.Button(self, text="✅ Confirm & Save", 
                                           bg="#2c3e50", fg="white", font=("Segoe UI", 11, "bold"),
                                           command=self.save_data)
                self.btn_confirm.pack(fill="x", side="bottom", padx=20, pady=20)

    def save_data(self):
        self.result = {
            "id": self.id_entry.get().strip(),
            "name": self.name_entry.get().strip(),
            "dept": self.dept_combo.get(),
            "year": int(self.year_combo.get())
        }
        self.destroy()


class CanteenFaceDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(getattr(config, 'WINDOW_TITLE', "College Canteen Face Detection"))
        self.root.geometry("1400x850")
        self.root.minsize(1200, 700)
        
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
        style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'))
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Status.TLabel', font=('Segoe UI', 10))
        style.configure('Big.TButton', font=('Segoe UI', 11, 'bold'), padding=10)
    
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
        """Build the live detection tab - LAYOUT FIXED"""
        # Split into left (video) and right (info) panels
        left_frame = ttk.Frame(self.detection_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(self.detection_tab, width=320)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # --- 1. CONTROL FRAME (Bottom) ---
        # We pack this FIRST so it reserves its space at the bottom immediately.
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, anchor="s")
        
        # Buttons
        self.start_btn = ttk.Button(control_frame, text="▶ Start Detection", style='Big.TButton', command=self.toggle_detection)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.register_btn = ttk.Button(control_frame, text="➕ Register / Upload", style='Big.TButton', command=self.open_registration_dialog)
        self.register_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = tk.Button(control_frame, text="🔄 Train System", font=('Segoe UI', 11, 'bold'), bg="#e67e22", fg="white", command=self.run_training)
        self.train_btn.pack(side=tk.LEFT, padx=5, fill="y")

        ttk.Button(control_frame, text="📷 Screenshot", command=self.take_screenshot).pack(side=tk.RIGHT, padx=5)
        
        # --- 2. VIDEO FRAME (Top) ---
        # We use pack_propagate(False) so the frame respects constraints
        video_frame = ttk.LabelFrame(left_frame, text="Camera Feed", padding="5")
        video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        video_frame.pack_propagate(False)
        
        self.video_label = ttk.Label(video_frame, text="Camera not started", background="black", foreground="white")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        # Bind resize event to adjust video size dynamically
        self.video_label.bind('<Configure>', lambda e: None) 
        
        # --- Right Panel (Stats) ---
        info_frame = ttk.LabelFrame(right_frame, text="Real-time Info", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.detected_count_label = ttk.Label(info_frame, text="Faces: 0", font=("Segoe UI", 12))
        self.detected_count_label.pack(anchor=tk.W)
        
        self.fps_label = ttk.Label(info_frame, text="FPS: 0")
        self.fps_label.pack(anchor=tk.W)
        
        recent_frame = ttk.LabelFrame(right_frame, text="Recent Recognitions", padding="10")
        recent_frame.pack(fill=tk.BOTH, expand=True)
        
        self.recent_listbox = tk.Listbox(recent_frame, font=('Consolas', 10), bg="#ecf0f1")
        self.recent_listbox.pack(fill=tk.BOTH, expand=True)
        
        today_frame = ttk.LabelFrame(right_frame, text="Today's Summary", padding="10")
        today_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.today_visits_label = ttk.Label(today_frame, text="Total Visits: 0", font=("Segoe UI", 10, "bold"))
        self.today_visits_label.pack(anchor=tk.W)
        
        self.today_unique_label = ttk.Label(today_frame, text="Unique Visitors: 0")
        self.today_unique_label.pack(anchor=tk.W)
    
    def build_students_tab(self):
        """Build the student management tab"""
        toolbar = ttk.Frame(self.students_tab)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        add_btn = ttk.Button(toolbar, text="➕ Add / Upload Student", command=self.open_registration_dialog)
        add_btn.pack(side=tk.LEFT, padx=5)

        refresh_btn = ttk.Button(toolbar, text="🔄 Refresh List", command=self.refresh_students)
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
        cards_frame = ttk.Frame(self.stats_tab)
        cards_frame.pack(fill=tk.X, pady=10)
        
        def make_card(parent, title):
            frame = ttk.LabelFrame(parent, text=title, padding="20")
            frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            lbl = ttk.Label(frame, text="0", font=('Segoe UI', 24, 'bold'))
            lbl.pack()
            return lbl

        self.stat_students = make_card(cards_frame, "Total Students")
        self.stat_today = make_card(cards_frame, "Today's Visits")
        self.stat_unique = make_card(cards_frame, "Unique Visitors")
        self.stat_unknown = make_card(cards_frame, "Unknown Faces")
        
        refresh_btn = ttk.Button(self.stats_tab, text="🔄 Refresh Statistics", command=self.refresh_statistics)
        refresh_btn.pack(pady=10)
        
        details_frame = ttk.LabelFrame(self.stats_tab, text="Detailed Report", padding="10")
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
        self.status_label.config(text=text, foreground=color)
    
    # --- ACTION HANDLERS ---

    def toggle_detection(self):
        if self.is_running:
            self.stop_detection()
        else:
            self.start_detection()
    
    def start_detection(self):
        if self.face_system is None:
            messagebox.showerror("Error", "System not initialized yet!")
            return
        
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return
        
        self.is_running = True
        self.start_btn.config(text="⏹ Stop Detection")
        self.update_status("🟢 Detection Running", "green")
        
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
    
    def stop_detection(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(text="▶ Start Detection")
        self.update_status("⚪ Detection Stopped", "gray")
        self.video_label.config(image='', text="Camera stopped")
    
    def video_loop(self):
        import time
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret: break
            
            self.current_frame = frame.copy()
            
            # AI Processing
            annotated_frame, recognized_people = self.face_system.process_frame(frame)
            
            # FPS Calculation
            frame_count += 1
            if time.time() - start_time >= 1.0:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()
            
            # UI Updates (Thread Safe)
            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            # Smart Resize logic for UI
            w_canvas = self.video_label.winfo_width()
            h_canvas = self.video_label.winfo_height()
            
            # Default if too small (start up)
            if w_canvas < 100: w_canvas = 800
            if h_canvas < 100: h_canvas = 600

            # Resize keeping aspect ratio roughly fit
            img = img.resize((w_canvas, h_canvas), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            def update_ui_elements():
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.detected_count_label.config(text=f"Faces: {len(recognized_people)}")
                
                # Update Listbox if someone known is found
                for person in recognized_people:
                    if person.get('student'):
                        name = person['student']['name']
                        ts = datetime.now().strftime('%H:%M:%S')
                        entry = f"{ts} - {name}"
                        
                        # Avoid duplicate spam in listbox (check top item)
                        if self.recent_listbox.size() == 0 or self.recent_listbox.get(0) != entry:
                            self.recent_listbox.insert(0, entry)
                            if self.recent_listbox.size() > 20:
                                self.recent_listbox.delete(20, tk.END)

            self.root.after(0, update_ui_elements)
            
            # Small sleep to prevent CPU hogging in loop
            time.sleep(0.01)

    def open_registration_dialog(self):
        """Open the unified registration dialog"""
        dialog = RegistrationDialog(self.root)
        self.root.wait_window(dialog)
        
        if dialog.result:
            data = dialog.result
            student_id = data['id']
            saved_paths = []
            
            # 1. Webcam Capture
            if dialog.mode == 'webcam':
                if not self.is_running or self.current_frame is None:
                     # Try to capture one frame if detection isn't running
                     temp_cap = cv2.VideoCapture(config.CAMERA_INDEX)
                     ret, frame = temp_cap.read()
                     temp_cap.release()
                     if ret:
                         self.current_frame = frame
                     else:
                         messagebox.showerror("Error", "Camera not available!")
                         return

                filename = f"{student_id}_cam_{int(time.time())}.jpg"
                path = os.path.join(config.FACES_DIR, filename)
                cv2.imwrite(path, self.current_frame)
                saved_paths.append(path)
                
            # 2. File Upload
            elif dialog.mode == 'upload':
                for idx, src_path in enumerate(dialog.uploaded_files):
                    try:
                        img = cv2.imread(src_path)
                        if img is not None:
                            filename = f"{student_id}_upload_{int(time.time())}_{idx}.jpg"
                            dest_path = os.path.join(config.FACES_DIR, filename)
                            cv2.imwrite(dest_path, img)
                            saved_paths.append(dest_path)
                    except Exception as e:
                        print(f"Error copying {src_path}: {e}")

            # 3. Finalize
            if saved_paths:
                # Add initial entry to DB
                if database.get_student_by_id(student_id):
                    messagebox.showinfo("Updated", f"Added new photos for existing student {data['name']}.")
                else:
                    database.add_student(
                        student_id=student_id,
                        name=data['name'],
                        department=data['dept'],
                        year=data['year'],
                        face_embedding=None, # Trainer will fill this
                        face_image_path=saved_paths[0]
                    )
                
                self.refresh_students()
                messagebox.showinfo("Success", 
                                  f"Saved {len(saved_paths)} photos.\n\n"
                                  "⚠️ CRITICAL: You must click 'Train System' now to update recognition!")
            else:
                messagebox.showerror("Error", "No valid images saved.")

    def run_training(self):
        """Run the training script in background"""
        def training_thread():
            self.train_btn.config(state="disabled", text="Training...", bg="gray")
            self.root.title("⚠️ Training System... (This may take a moment)")
            try:
                train_model.train_system()
                # Reload the recognition system's cache
                if self.face_system:
                    self.face_system.reload_known_faces()
                
                self.root.after(0, lambda: messagebox.showinfo("Success", "System Trained Successfully! Recognition updated."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Training Failed: {e}"))
            finally:
                self.root.after(0, lambda: self.train_btn.config(state="normal", text="🔄 Train System", bg="#e67e22"))
                self.root.after(0, lambda: self.root.title(getattr(config, 'WINDOW_TITLE', "College Canteen System")))
        
        threading.Thread(target=training_thread, daemon=True).start()

    # --- MANAGEMENT FUNCTIONS ---

    def refresh_students(self):
        for item in self.students_tree.get_children():
            self.students_tree.delete(item)
        
        students = database.get_all_students()
        for s in students:
            created = s.get('created_at', 'N/A')
            if created and len(created) > 10: created = created[:10]
            
            self.students_tree.insert('', tk.END, values=(
                s['id'], s['student_id'], s['name'], 
                s.get('department',''), s.get('year',''), created
            ))

    def filter_students(self):
        search = self.student_search_var.get().lower()
        for item in self.students_tree.get_children():
            self.students_tree.delete(item)
            
        all_s = database.get_all_students()
        for s in all_s:
            if search in s['name'].lower() or search in s['student_id'].lower():
                created = s.get('created_at', 'N/A')
                if created and len(created) > 10: created = created[:10]
                self.students_tree.insert('', tk.END, values=(
                    s['id'], s['student_id'], s['name'], 
                    s.get('department',''), s.get('year',''), created
                ))

    def delete_student(self):
        selected = self.students_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Select a student to delete")
            return
        
        item = self.students_tree.item(selected[0])
        student_id = item['values'][1]
        name = item['values'][2]
        
        if messagebox.askyesno("Confirm", f"Delete {name} ({student_id})?"):
            database.delete_student(student_id)
            self.refresh_students()
            if self.face_system: self.face_system.reload_known_faces()
            messagebox.showinfo("Deleted", "Student removed.")

    def refresh_logs(self):
        for item in self.logs_tree.get_children():
            self.logs_tree.delete(item)
            
        date = self.log_date_var.get().strip() or None
        sid = self.log_student_var.get().strip() or None
        
        logs = database.get_visit_logs(date=date, student_id=sid)
        for log in logs:
            status = "Known" if log['is_known'] else "Unknown"
            duration = f"{log.get('duration_minutes','-')} min"
            t = log['entry_time']
            if t and ' ' in t: t = t.split(' ')[1][:8]
            
            self.logs_tree.insert('', tk.END, values=(
                log['id'], log.get('date',''), t, 
                log.get('student_id',''), log.get('student_name',''), status, duration
            ))

    def export_logs(self):
        if export_logs_to_csv is None:
            messagebox.showerror("Error", "Utils module missing or incomplete.")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if filepath:
            try:
                export_logs_to_csv(filepath, self.log_date_var.get().strip(), self.log_student_var.get().strip())
                messagebox.showinfo("Exported", f"Saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def refresh_statistics(self):
        stats = database.get_daily_statistics()
        students = database.get_all_students()
        
        self.stat_students.config(text=str(len(students)))
        self.stat_today.config(text=str(stats['total_visits']))
        self.stat_unique.config(text=str(stats['unique_visitors']))
        self.stat_unknown.config(text=str(stats['unknown_visitors']))
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, f"Report Generated: {datetime.now()}\n\n")
        self.stats_text.insert(tk.END, f"Total Visits Today: {stats['total_visits']}\n")
        self.stats_text.insert(tk.END, f"Unique Visitors: {stats['unique_visitors']}\n")
        self.stats_text.insert(tk.END, f"Avg Duration: {stats['average_duration_minutes']} min\n")

    def refresh_all_data(self):
        self.refresh_students()
        self.refresh_logs()
        self.refresh_statistics()

    def take_screenshot(self):
        if self.current_frame is not None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(config.SCREENSHOTS_DIR, f"manual_{ts}.jpg")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, self.current_frame)
            messagebox.showinfo("Saved", f"Screenshot saved to {path}")

    def on_closing(self):
        self.is_running = False
        if self.cap: self.cap.release()
        if self.face_system: self.face_system.stop()
        self.root.destroy()
        sys.exit(0)

    def run(self):
        self.root.mainloop()

def main():
    app = CanteenFaceDetectionGUI()
    app.run()

if __name__ == "__main__":
    main()