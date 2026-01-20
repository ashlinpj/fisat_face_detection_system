"""
GUI Application - College Canteen Face Detection System
Modern Tkinter-based graphical user interface
Includes:
- Layout Fixed (Buttons visible)
- "Recent Recognitions" Removed
- System Console in Logs Tab
- Multi-Image Upload & Training
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

# --- CONSOLE REDIRECTOR ---
class TextRedirector:
    """Redirects print() statements to a Tkinter Text widget"""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        try:
            self.widget.configure(state="normal")
            self.widget.insert("end", str, (self.tag,))
            self.widget.see("end") # Auto-scroll to bottom
            self.widget.configure(state="disabled")
            # Keep printing to standard terminal too
            sys.__stdout__.write(str) 
        except:
            pass

    def flush(self):
        sys.__stdout__.flush()

# --- REGISTRATION DIALOG ---
class RegistrationDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Register New Student")
        self.geometry("450x600")
        self.configure(bg="#f0f0f0")
        self.transient(parent)
        self.grab_set()
        
        self.result = None
        self.mode = None 
        self.uploaded_files = []
        
        style = ttk.Style()
        style.configure("Reg.TLabel", font=("Segoe UI", 10))
        
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill="both", expand=True)
        
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
        
        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=20)
        
        ttk.Label(main_frame, text="Choose Method:", font=("Segoe UI", 10, "bold")).pack(pady=5)

        self.btn_capture = tk.Button(main_frame, text="📷 Capture from Webcam", 
                                   bg="#27ae60", fg="white", font=("Segoe UI", 11),
                                   command=self.use_webcam)
        self.btn_capture.pack(fill="x", pady=5)
        
        tk.Label(main_frame, text="- OR -", bg="#f0f0f0", fg="#7f8c8d").pack(pady=2)
        
        self.btn_upload = tk.Button(main_frame, text="📂 Upload Photos (Multiple)", 
                                  bg="#2980b9", fg="white", font=("Segoe UI", 11),
                                  command=self.choose_files)
        self.btn_upload.pack(fill="x", pady=5)
        
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
        if not self.validate_inputs(): return
        files = filedialog.askopenfilenames(parent=self, title="Select Photos", filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if files:
            self.uploaded_files = list(files)
            self.lbl_status.config(text=f"✅ Selected {len(files)} files")
            self.mode = 'upload'
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

# --- MAIN GUI ---
class CanteenFaceDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(getattr(config, 'WINDOW_TITLE', "College Canteen System"))
        self.root.geometry("1400x850")
        self.root.minsize(1200, 700)
        
        # Variables
        self.cap = None
        self.is_running = False
        self.face_system = None
        self.current_frame = None
        
        # UI & System
        self.setup_styles()
        self.build_ui()
        self.initialize_system()
        
        # Start Auto-Refresh Loop for Logs
        self.root.after(5000, self.auto_refresh_loop)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'))
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Status.TLabel', font=('Segoe UI', 10))
        style.configure('Big.TButton', font=('Segoe UI', 11, 'bold'), padding=10)
    
    def build_ui(self):
        # Main Layout
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Frame(main_container)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="🍽️ College Canteen Face Detection System", style='Title.TLabel').pack(side=tk.LEFT)
        self.status_label = ttk.Label(header, text="⚪ System Ready", style='Status.TLabel')
        self.status_label.pack(side=tk.RIGHT)
        
        # Tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create Tabs
        self.detection_tab = ttk.Frame(self.notebook); self.notebook.add(self.detection_tab, text="📹 Live Detection")
        self.students_tab = ttk.Frame(self.notebook); self.notebook.add(self.students_tab, text="👥 Students")
        self.logs_tab = ttk.Frame(self.notebook); self.notebook.add(self.logs_tab, text="📋 Visit Logs & Console")
        self.stats_tab = ttk.Frame(self.notebook); self.notebook.add(self.stats_tab, text="📊 Statistics")
        
        # Build Contents
        self.build_detection_tab()
        self.build_students_tab()
        self.build_logs_tab()
        self.build_stats_tab()
    
    def build_detection_tab(self):
        # Layout: Left (Video+Controls), Right (Info)
        left_frame = ttk.Frame(self.detection_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(self.detection_tab, width=320)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # 1. CONTROLS (Bottom) - Packed First to reserve space
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, anchor="s")
        
        self.start_btn = ttk.Button(control_frame, text="▶ Start Detection", style='Big.TButton', command=self.toggle_detection)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="➕ Register / Upload", style='Big.TButton', command=self.open_registration_dialog).pack(side=tk.LEFT, padx=5)
        
        self.train_btn = tk.Button(control_frame, text="🔄 Train System", font=('Segoe UI', 11, 'bold'), bg="#e67e22", fg="white", command=self.run_training)
        self.train_btn.pack(side=tk.LEFT, padx=5, fill="y")
        
        ttk.Button(control_frame, text="📷 Screenshot", command=self.take_screenshot).pack(side=tk.RIGHT, padx=5)
        
        # 2. VIDEO (Top - Resizable) - Packed Last
        video_frame = ttk.LabelFrame(left_frame, text="Camera Feed", padding="5")
        video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        video_frame.pack_propagate(False) # Prevent frame from growing too big
        
        self.video_label = ttk.Label(video_frame, text="Camera not started", background="black", foreground="white")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.bind('<Configure>', lambda e: None) 
        
        # 3. RIGHT PANEL (Stats)
        info_frame = ttk.LabelFrame(right_frame, text="Real-time Info", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        self.detected_count_label = ttk.Label(info_frame, text="Faces: 0", font=("Segoe UI", 12))
        self.detected_count_label.pack(anchor=tk.W)
        self.fps_label = ttk.Label(info_frame, text="FPS: 0")
        self.fps_label.pack(anchor=tk.W)
        
        # Removed "Recent Recognitions" Frame as requested
        
        today_frame = ttk.LabelFrame(right_frame, text="Today's Summary", padding="10")
        today_frame.pack(fill=tk.X, pady=(10, 0))
        self.today_visits_label = ttk.Label(today_frame, text="Total Visits: 0", font=("Segoe UI", 10, "bold"))
        self.today_visits_label.pack(anchor=tk.W)
        self.today_unique_label = ttk.Label(today_frame, text="Unique Visitors: 0")
        self.today_unique_label.pack(anchor=tk.W)
    
    def build_students_tab(self):
        toolbar = ttk.Frame(self.students_tab)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="➕ Add / Upload", command=self.open_registration_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="🔄 Refresh", command=self.refresh_students).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="🗑️ Delete", command=self.delete_student).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(toolbar, text="Search:").pack(side=tk.LEFT, padx=(20, 5))
        self.student_search_var = tk.StringVar()
        self.student_search_var.trace('w', lambda *args: self.filter_students())
        ttk.Entry(toolbar, textvariable=self.student_search_var, width=30).pack(side=tk.LEFT, padx=5)
        
        table_frame = ttk.Frame(self.students_tab)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        cols = ('ID', 'Student ID', 'Name', 'Department', 'Year', 'Registered')
        self.students_tree = ttk.Treeview(table_frame, columns=cols, show='headings')
        for col in cols: self.students_tree.heading(col, text=col)
        self.students_tree.column('ID', width=50); self.students_tree.column('Name', width=200)
        
        scrolly = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.students_tree.yview)
        self.students_tree.configure(yscrollcommand=scrolly.set)
        self.students_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrolly.pack(side=tk.RIGHT, fill=tk.Y)

    def build_logs_tab(self):
        """Split view: Top = Table, Bottom = System Console"""
        
        # Use PanedWindow to split top/bottom
        paned_window = ttk.PanedWindow(self.logs_tab, orient=tk.VERTICAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # --- TOP: VISIT LOGS TABLE ---
        top_frame = ttk.Frame(paned_window)
        paned_window.add(top_frame, weight=3)
        
        # Filters
        filter_frame = ttk.Frame(top_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(filter_frame, text="Date:").pack(side=tk.LEFT, padx=5)
        self.log_date_var = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        ttk.Entry(filter_frame, textvariable=self.log_date_var, width=12).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(filter_frame, text="ID:").pack(side=tk.LEFT, padx=5)
        self.log_student_var = tk.StringVar()
        ttk.Entry(filter_frame, textvariable=self.log_student_var, width=12).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(filter_frame, text="🔍 Filter", command=self.refresh_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="✖ Show All History", command=self.clear_filters).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="📥 Export CSV", command=self.export_logs).pack(side=tk.RIGHT, padx=5)
        
        # Table (Contains the requested fields: ID, Date, Entry Time, Student ID, Name)
        cols = ('ID', 'Date', 'Entry Time', 'Student ID', 'Name', 'Status', 'Duration')
        self.logs_tree = ttk.Treeview(top_frame, columns=cols, show='headings')
        for col in cols: self.logs_tree.heading(col, text=col)
        self.logs_tree.column('ID', width=50); self.logs_tree.column('Name', width=180)
        
        scrolly = ttk.Scrollbar(top_frame, orient=tk.VERTICAL, command=self.logs_tree.yview)
        self.logs_tree.configure(yscrollcommand=scrolly.set)
        self.logs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrolly.pack(side=tk.RIGHT, fill=tk.Y)
        
        # --- BOTTOM: SYSTEM CONSOLE ---
        bottom_frame = ttk.LabelFrame(paned_window, text="System Console (Live Output)")
        paned_window.add(bottom_frame, weight=1)
        
        self.console_text = tk.Text(bottom_frame, height=8, bg="#1e1e1e", fg="#00ff00", 
                                  font=("Consolas", 10), state="disabled")
        self.console_text.pack(fill=tk.BOTH, expand=True)
        
        sys.stdout = TextRedirector(self.console_text, "stdout")
        sys.stderr = TextRedirector(self.console_text, "stderr")
        
        print("System Console initialized. Training logs and errors will appear here.")

    def build_stats_tab(self):
        cards_frame = ttk.Frame(self.stats_tab)
        cards_frame.pack(fill=tk.X, pady=10)
        
        def make_card(parent, title):
            f = ttk.LabelFrame(parent, text=title, padding="20")
            f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            lbl = ttk.Label(f, text="0", font=('Segoe UI', 24, 'bold'))
            lbl.pack()
            return lbl

        self.stat_students = make_card(cards_frame, "Total Students")
        self.stat_today = make_card(cards_frame, "Today's Visits")
        self.stat_unique = make_card(cards_frame, "Unique Visitors")
        self.stat_unknown = make_card(cards_frame, "Unknown Faces")
        
        ttk.Button(self.stats_tab, text="🔄 Refresh Statistics", command=self.refresh_statistics).pack(pady=10)
        
        details_frame = ttk.LabelFrame(self.stats_tab, text="Detailed Report", padding="10")
        details_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.stats_text = tk.Text(details_frame, font=('Consolas', 10), height=15)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

    def initialize_system(self):
        self.update_status("Initializing...", "orange")
        def init_thread():
            try:
                database.init_database()
                self.face_system = FaceRecognitionSystem()
                self.root.after(0, lambda: self.update_status("🟢 System Ready", "green"))
                self.root.after(0, self.refresh_all_data)
            except Exception as e:
                self.root.after(0, lambda: self.update_status(f"🔴 Error: {str(e)}", "red"))
                print(f"Error initializing: {e}")
        threading.Thread(target=init_thread, daemon=True).start()

    def update_status(self, text, color="black"):
        self.status_label.config(text=text, foreground=color)

    # --- ACTIONS ---
    def toggle_detection(self):
        if self.is_running: self.stop_detection()
        else: self.start_detection()

    def start_detection(self):
        if not self.face_system: return
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Camera not found")
            return
        self.is_running = True
        self.start_btn.config(text="⏹ Stop Detection")
        self.update_status("🟢 Detection Running", "green")
        threading.Thread(target=self.video_loop, daemon=True).start()

    def stop_detection(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.start_btn.config(text="▶ Start Detection")
        self.update_status("⚪ Detection Stopped", "gray")
        self.video_label.config(image='')

    def video_loop(self):
        start_time = time.time()
        frame_count = 0
        fps = 0
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret: break
            self.current_frame = frame.copy()
            
            annotated_frame, recognized = self.face_system.process_frame(frame)
            
            # FPS
            frame_count += 1
            if time.time() - start_time >= 1.0:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0; start_time = time.time()
            
            # Display
            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            # Resize
            w = self.video_label.winfo_width()
            h = self.video_label.winfo_height()
            if w < 100: w = 800
            if h < 100: h = 600
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            def update_ui():
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.detected_count_label.config(text=f"Faces: {len(recognized)}")
                
                # Removed "Recent Recognitions" logic from here as requested
                            
            self.root.after(0, update_ui)
            time.sleep(0.01)

    def open_registration_dialog(self):
        dialog = RegistrationDialog(self.root)
        self.root.wait_window(dialog)
        
        if dialog.result:
            data = dialog.result
            sid = data['id']
            saved = []
            
            if dialog.mode == 'webcam':
                if self.is_running and self.current_frame is not None:
                    path = os.path.join(config.FACES_DIR, f"{sid}_cam_{int(time.time())}.jpg")
                    cv2.imwrite(path, self.current_frame)
                    saved.append(path)
                else:
                    messagebox.showerror("Error", "Camera not running!")
            elif dialog.mode == 'upload':
                for i, src in enumerate(dialog.uploaded_files):
                    img = cv2.imread(src)
                    if img is not None:
                        path = os.path.join(config.FACES_DIR, f"{sid}_upl_{int(time.time())}_{i}.jpg")
                        cv2.imwrite(path, img)
                        saved.append(path)
            
            if saved:
                if not database.get_student_by_id(sid):
                    database.add_student(sid, data['name'], data['dept'], data['year'], None, saved[0])
                self.refresh_students()
                print(f"Registered {data['name']}. Please Train System.")
                messagebox.showinfo("Success", "Photos saved. Please click 'Train System' now.")

    def run_training(self):
        def task():
            self.train_btn.config(state="disabled", text="Training...", bg="gray")
            print("\n--- STARTING TRAINING ---")
            try:
                train_model.train_system()
                if self.face_system: self.face_system.reload_known_faces()
                self.root.after(0, lambda: messagebox.showinfo("Done", "Training Complete!"))
            except Exception as e:
                print(f"Training Error: {e}")
            finally:
                self.root.after(0, lambda: self.train_btn.config(state="normal", text="🔄 Train System", bg="#e67e22"))
                print("--- TRAINING END ---")
        threading.Thread(target=task, daemon=True).start()

    def refresh_students(self):
        for i in self.students_tree.get_children(): self.students_tree.delete(i)
        for s in database.get_all_students():
            self.students_tree.insert('', 'end', values=(s['id'], s['student_id'], s['name'], s['department'], s['year'], s['created_at'][:10]))

    def filter_students(self):
        term = self.student_search_var.get().lower()
        for i in self.students_tree.get_children(): self.students_tree.delete(i)
        for s in database.get_all_students():
            if term in s['name'].lower() or term in s['student_id'].lower():
                self.students_tree.insert('', 'end', values=(s['id'], s['student_id'], s['name'], s['department'], s['year'], s['created_at'][:10]))

    def clear_filters(self):
        """Reset filters to show all history"""
        self.log_date_var.set("")
        self.log_student_var.set("")
        self.refresh_logs()

    def refresh_logs(self):
        for i in self.logs_tree.get_children(): self.logs_tree.delete(i)
        
        # FIX: Treat empty strings as None so DB fetches all records
        d = self.log_date_var.get().strip()
        if d == "": d = None
        s = self.log_student_var.get().strip()
        if s == "": s = None
        
        logs = database.get_visit_logs(date=d, student_id=s)
        for l in logs:
            t = l['entry_time'].split(' ')[1][:8] if ' ' in l['entry_time'] else l['entry_time']
            self.logs_tree.insert('', 'end', values=(l['id'], l.get('date'), t, l['student_id'], l['student_name'], "Known" if l['is_known'] else "Unknown", f"{l.get('duration_minutes','-')} m"))

    def auto_refresh_loop(self):
        """Automatically refresh logs every 5 seconds"""
        if self.is_running:
            self.refresh_logs()
            self.refresh_statistics()
        self.root.after(5000, self.auto_refresh_loop)

    def refresh_statistics(self):
        stats = database.get_daily_statistics()
        self.stat_students.config(text=str(len(database.get_all_students())))
        self.stat_today.config(text=str(stats['total_visits']))
        self.stat_unique.config(text=str(stats['unique_visitors']))
        self.stat_unknown.config(text=str(stats['unknown_visitors']))

    def refresh_all_data(self):
        self.refresh_students(); self.refresh_logs(); self.refresh_statistics()

    def delete_student(self):
        sel = self.students_tree.selection()
        if sel:
            sid = self.students_tree.item(sel[0])['values'][1]
            if messagebox.askyesno("Delete", f"Delete {sid}?"):
                database.delete_student(sid)
                self.refresh_all_data()
                if self.face_system: self.face_system.reload_known_faces()

    def export_logs(self):
        if not export_logs_to_csv: return
        f = filedialog.asksaveasfilename(defaultextension=".csv")
        if f: 
            # FIX: Pass current filter values correctly
            d = self.log_date_var.get().strip()
            if d == "": d = None
            s = self.log_student_var.get().strip()
            if s == "": s = None
            export_logs_to_csv(f, d, s)

    def take_screenshot(self):
        if self.current_frame is not None:
            p = os.path.join(config.SCREENSHOTS_DIR, f"shot_{int(time.time())}.jpg")
            cv2.imwrite(p, self.current_frame)
            print(f"Screenshot: {p}")

    def on_closing(self):
        self.is_running = False
        if self.cap: self.cap.release()
        if self.face_system: self.face_system.stop()
        self.root.destroy()
        sys.exit(0)

def main():
    CanteenFaceDetectionGUI().root.mainloop()

if __name__ == "__main__":
    main()