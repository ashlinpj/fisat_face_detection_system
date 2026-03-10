"""GUI Application - Tkinter-based face detection interface.

Receives a Container for all dependencies rather than creating them directly.
All database access goes through injected repositories.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from datetime import datetime
from collections import deque

import config


class CanteenFaceDetectionGUI:
    """Main GUI application. Receives a DI container for all dependencies."""

    def __init__(self, container):
        self.container = container
        self.student_repo = container.student_repo
        self.visit_repo = container.visit_repo
        self.report_service = container.report_service

        self.root = tk.Tk()
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)

        try:
            self.root.iconbitmap("icon.ico")
        except tk.TclError:
            pass

        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.video_thread = None

        self.recent_detections = deque(maxlen=20)

        self.setup_styles()
        self.build_ui()
        self._finish_init()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _finish_init(self):
        """Complete initialization using the container"""
        self.update_status("🟢 System Ready", "green")
        self.refresh_all_data()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Status.TLabel', font=('Segoe UI', 10))
        style.configure('Big.TButton', font=('Segoe UI', 11), padding=10)
        style.configure('Success.TLabel', foreground='green', font=('Segoe UI', 10, 'bold'))
        style.configure('Warning.TLabel', foreground='orange', font=('Segoe UI', 10, 'bold'))
        style.configure('Danger.TLabel', foreground='red', font=('Segoe UI', 10, 'bold'))

    def build_ui(self):
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="🍽️ College Canteen Face Detection System",
                  style='Title.TLabel').pack(side=tk.LEFT)

        self.status_label = ttk.Label(header_frame, text="⚪ System Ready",
                                      style='Status.TLabel')
        self.status_label.pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        tabs = [
            ("detection", "📹 Live Detection", self.build_detection_tab),
            ("students", "👥 Students", self.build_students_tab),
            ("logs", "📋 Visit Logs", self.build_logs_tab),
            ("stats", "📊 Statistics", self.build_stats_tab)
        ]

        for tab_name, tab_text, build_func in tabs:
            tab = ttk.Frame(self.notebook)
            setattr(self, f"{tab_name}_tab", tab)
            self.notebook.add(tab, text=tab_text)
            build_func()

    def build_detection_tab(self):
        left_frame = ttk.Frame(self.detection_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        right_frame = ttk.Frame(self.detection_tab, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)

        video_frame = ttk.LabelFrame(left_frame, text="Camera Feed", padding="5")
        video_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_frame, text="Camera not started")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=10)

        buttons = [
            ("start_btn", "▶ Start Detection", self.toggle_detection, True),
            ("register_btn", "➕ Register Face", self.open_registration_dialog, True),
            (None, "📷 Screenshot", self.take_screenshot, False)
        ]

        for attr, text, cmd, is_big in buttons:
            btn = ttk.Button(control_frame, text=text, command=cmd,
                           style='Big.TButton' if is_big else None)
            btn.pack(side=tk.LEFT, padx=5)
            if attr:
                setattr(self, attr, btn)

        info_frame = ttk.LabelFrame(right_frame, text="Detection Info", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        info_labels = [
            ("detected_count_label", "Faces Detected: 0"),
            ("known_count_label", "Known: 0"),
            ("unknown_count_label", "Unknown: 0"),
            ("fps_label", "FPS: 0")
        ]

        for attr, text in info_labels:
            lbl = ttk.Label(info_frame, text=text)
            lbl.pack(anchor=tk.W)
            setattr(self, attr, lbl)

        recent_frame = ttk.LabelFrame(right_frame, text="Recent Detections", padding="10")
        recent_frame.pack(fill=tk.BOTH, expand=True)

        self.recent_listbox = tk.Listbox(recent_frame, font=('Consolas', 9))
        self.recent_listbox.pack(fill=tk.BOTH, expand=True)

        today_frame = ttk.LabelFrame(right_frame, text="Today's Summary", padding="10")
        today_frame.pack(fill=tk.X, pady=(10, 0))

        self.today_visits_label = ttk.Label(today_frame, text="Total Visits: 0")
        self.today_visits_label.pack(anchor=tk.W)

        self.today_unique_label = ttk.Label(today_frame, text="Unique Visitors: 0")
        self.today_unique_label.pack(anchor=tk.W)

    def build_students_tab(self):
        toolbar = ttk.Frame(self.students_tab)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        buttons = [
            ("➕ Add Student", self.add_student_dialog),
            ("🖼️ Register from Image", self.register_from_file),
            ("🔄 Refresh", self.refresh_students),
            ("🗑️ Delete Selected", self.delete_student)
        ]

        for text, cmd in buttons:
            ttk.Button(toolbar, text=text, command=cmd).pack(side=tk.LEFT, padx=5)

        ttk.Label(toolbar, text="Search:").pack(side=tk.LEFT, padx=(20, 5))
        self.student_search_var = tk.StringVar()
        self.student_search_var.trace('w', lambda *args: self.filter_students())
        ttk.Entry(toolbar, textvariable=self.student_search_var, width=30).pack(side=tk.LEFT, padx=5)

        table_frame = ttk.Frame(self.students_tab)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ('ID', 'Student ID', 'Name', 'Department', 'Year', 'Registered')
        self.students_tree = ttk.Treeview(table_frame, columns=columns, show='headings')

        widths = {'Name': 200, 'Registered': 150}
        for col in columns:
            self.students_tree.heading(col, text=col)
            self.students_tree.column(col, width=widths.get(col, 100))

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.students_tree.yview)
        self.students_tree.configure(yscrollcommand=scrollbar.set)
        self.students_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def build_logs_tab(self):
        filter_frame = ttk.Frame(self.logs_tab)
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(filter_frame, text="Date:").pack(side=tk.LEFT, padx=5)
        self.log_date_var = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        ttk.Entry(filter_frame, textvariable=self.log_date_var, width=15).pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="Student ID:").pack(side=tk.LEFT, padx=(20, 5))
        self.log_student_var = tk.StringVar()
        ttk.Entry(filter_frame, textvariable=self.log_student_var, width=15).pack(side=tk.LEFT, padx=5)

        buttons = [
            ("🔍 Filter", self.refresh_logs, tk.LEFT),
            ("Clear", self.clear_log_filters, tk.LEFT),
            ("📥 Export CSV", self.export_logs, tk.RIGHT)
        ]

        for text, cmd, side in buttons:
            ttk.Button(filter_frame, text=text, command=cmd).pack(side=side, padx=5)

        table_frame = ttk.Frame(self.logs_tab)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ('ID', 'Date', 'Entry Time', 'Student ID', 'Name', 'Status', 'Duration')
        self.logs_tree = ttk.Treeview(table_frame, columns=columns, show='headings')

        for col in columns:
            self.logs_tree.heading(col, text=col)
            self.logs_tree.column(col, width=180 if col == 'Name' else 100)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.logs_tree.yview)
        self.logs_tree.configure(yscrollcommand=scrollbar.set)
        self.logs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def build_stats_tab(self):
        cards_frame = ttk.Frame(self.stats_tab)
        cards_frame.pack(fill=tk.X, pady=10)

        cards = [
            ("stat_students", "Total Students"),
            ("stat_today", "Today's Visits"),
            ("stat_unique", "Unique Visitors Today"),
            ("stat_unknown", "Unknown Faces Today")
        ]

        for attr, title in cards:
            card = ttk.LabelFrame(cards_frame, text=title, padding="20")
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            lbl = ttk.Label(card, text="0", font=('Segoe UI', 24, 'bold'))
            lbl.pack()
            setattr(self, attr, lbl)

        ttk.Button(self.stats_tab, text="🔄 Refresh Statistics",
                   command=self.refresh_statistics).pack(pady=10)

        details_frame = ttk.LabelFrame(self.stats_tab, text="Detailed Statistics", padding="10")
        details_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.stats_text = tk.Text(details_frame, font=('Consolas', 10), height=15)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

    # --- System control ---

    def update_status(self, text, color="black"):
        self.status_label.config(text=text)

    def toggle_detection(self):
        self.stop_detection() if self.is_running else self.start_detection()

    def start_detection(self):
        if not self.container:
            messagebox.showerror("Error", "System not initialized yet!")
            return

        if config.USE_RTSP:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            for attempt in range(config.RTSP_RECONNECT_ATTEMPTS):
                self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret:
                        break

                if attempt < config.RTSP_RECONNECT_ATTEMPTS - 1:
                    import time
                    time.sleep(config.RTSP_RECONNECT_DELAY)

            if not self.cap.isOpened():
                if self.cap:
                    self.cap.release()
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
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

            if not self.cap.isOpened():
                if self.cap:
                    self.cap.release()
                messagebox.showerror("Error", "Could not open camera!")
                return

        self.is_running = True
        self.start_btn.config(text="⏹ Stop Detection")
        status_msg = "🟢 RTSP Stream Active" if config.USE_RTSP else "🟢 Detection Running"
        self.update_status(status_msg, "green")

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
        failed_reads = 0
        max_failed_reads = 30

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                failed_reads += 1

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
                        self.root.after(0, lambda: messagebox.showerror("Error", "RTSP stream lost!"))
                        break
                elif failed_reads >= max_failed_reads:
                    break

                time.sleep(0.1)
                continue

            failed_reads = 0
            self.current_frame = frame.copy()
            annotated_frame, recognized_people = self.container.frame_processor.process_frame(frame)

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            display_frame = self._create_display_frame(annotated_frame, recognized_people, fps)
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            def update_ui():
                if hasattr(self.video_label, 'imgtk') and self.video_label.imgtk is not None:
                    old_imgtk = self.video_label.imgtk
                    self.video_label.imgtk = None
                    del old_imgtk

                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

                known = sum(1 for p in recognized_people if p['is_known'])
                self.detected_count_label.config(text=f"Faces Detected: {len(recognized_people)}")
                self.known_count_label.config(text=f"Known: {known}")
                self.unknown_count_label.config(text=f"Unknown: {len(recognized_people) - known}")
                self.fps_label.config(text=f"FPS: {fps:.1f}")

                for person in recognized_people:
                    if person['is_known']:
                        time_str = datetime.now().strftime('%H:%M:%S')
                        detection_entry = f"{time_str} - {person['name']}"
                        self.recent_detections.appendleft(detection_entry)

                if recognized_people:
                    self.recent_listbox.delete(0, tk.END)
                    for entry in self.recent_detections:
                        self.recent_listbox.insert(tk.END, entry)

            self.root.after(0, update_ui)
            time.sleep(0.03)

    def _create_display_frame(self, annotated_frame, recognized_people, fps):
        if config.SHOW_WINDOW:
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w = display_frame.shape[:2]
            scale = 640 / w
            return cv2.resize(display_frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            return self._create_names_only_display(recognized_people, fps)

    def _create_names_only_display(self, recognized_people, fps):
        display_width, display_height = 640, 480
        display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        display_frame[:] = (30, 30, 30)

        unique_names = list(dict.fromkeys([
            p.get('name', 'Unknown Person') if p.get('is_known') else 'Unknown Person'
            for p in recognized_people
        ]))

        cv2.putText(display_frame, "People Detected", (display_width // 2 - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.line(display_frame, (40, 70), (display_width - 40, 70), (100, 100, 100), 2)

        if unique_names:
            y_offset = 120
            for i, name in enumerate(unique_names):
                color = (100, 255, 100) if 'Unknown' not in name else (100, 150, 255)
                cv2.putText(display_frame, f"{i+1}. {name}", (80, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                y_offset += 55

                if y_offset > display_height - 100:
                    remaining = len(unique_names) - i - 1
                    if remaining > 0:
                        cv2.putText(display_frame, f"... and {remaining} more", (80, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
                    break
        else:
            cv2.putText(display_frame, "No people detected",
                       (display_width // 2 - 150, display_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)

        cv2.putText(display_frame, f"FPS: {fps:.1f} | Total: {len(unique_names)}",
                   (display_width // 2 - 100, display_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return display_frame

    # --- Registration ---

    def open_registration_dialog(self):
        if not self.is_running:
            messagebox.showwarning("Warning", "Please start detection first!")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Register New Student")
        dialog.geometry("400x500")
        dialog.transient(self.root)
        dialog.grab_set()

        fields = [
            ("Student ID:", ttk.Entry(dialog, width=30)),
            ("Name:", ttk.Entry(dialog, width=30)),
            ("Department:", ttk.Entry(dialog, width=30))
        ]
        entries = {}

        for label, widget in fields:
            ttk.Label(dialog, text=label).pack(pady=5)
            widget.pack()
            entries[label] = widget

        ttk.Label(dialog, text="Year (1-4):").pack(pady=5)
        year_var = tk.StringVar(value="1")
        ttk.Combobox(dialog, textvariable=year_var, values=['1', '2', '3', '4'], width=27).pack()

        ttk.Label(dialog, text="We will capture ~10 guided poses (angles + glasses).",
                  wraplength=340).pack(pady=(8, 4))
        include_glasses = tk.BooleanVar(value=True)
        ttk.Checkbutton(dialog, text="Include glasses captures (if applicable)",
                       variable=include_glasses).pack()

        def do_register():
            student_id = entries["Student ID:"].get().strip()
            name = entries["Name:"].get().strip()
            department = entries["Department:"].get().strip()
            year = int(year_var.get())

            if not student_id or not name:
                messagebox.showerror("Error", "Student ID and Name are required!")
                return

            if not self.cap or not self.cap.isOpened():
                messagebox.showerror("Error", "Camera is not running. Start detection and try again.")
                return

            frames = self.capture_pose_sequence(self.get_pose_script(include_glasses.get()))
            if len(frames) < 6:
                messagebox.showerror("Error", "Could not capture enough samples.")
                return

            success = self.container.registration_service.register_from_frames(
                frames, student_id, name, department, year
            )

            if success:
                self.container.reload_known_faces()
                messagebox.showinfo("Success", f"Student {name} registered with {len(frames)} samples!")
                dialog.destroy()
                self.refresh_students()
            else:
                messagebox.showerror("Error", "Failed to register student.")

        ttk.Button(dialog, text="📷 Capture & Register", command=do_register).pack(pady=20)

    def get_pose_script(self, include_glasses: bool = True):
        script = [
            ("Front - neutral", "Look straight ahead with a relaxed face."),
            ("Front - smile", "Smile naturally while facing the camera."),
            ("Left turn ~30°", "Turn your head slightly left; keep eyes on camera."),
            ("Right turn ~30°", "Turn your head slightly right; keep eyes on camera."),
            ("Left profile ~60°", "Turn further left."),
            ("Right profile ~60°", "Turn further right."),
            ("Chin slightly down", "Tilt your chin down a bit."),
            ("Chin slightly up", "Tilt your chin up a bit."),
            ("Bright light", "Step into brighter light facing the camera."),
            ("Softer light", "Step slightly aside to introduce mild shadows.")
        ]
        if include_glasses:
            script.extend([
                ("With glasses - front", "Put on glasses (if any) and face the camera."),
                ("With glasses - slight angle", "Glasses on; turn 20-30° to either side.")
            ])
        return script

    def capture_pose_sequence(self, pose_script):
        import time
        captured_frames = []

        for idx, (title, tip) in enumerate(pose_script, start=1):
            messagebox.showinfo("Capture Pose",
                              f"Step {idx}/{len(pose_script)}: {title}\n\n{tip}\n\nClick OK when ready.")
            time.sleep(0.4)

            frame = None
            if self.cap:
                ret, raw = self.cap.read()
                if ret:
                    frame = raw.copy()

            if frame is None and self.current_frame is not None:
                frame = self.current_frame.copy()

            if frame is not None:
                captured_frames.append(frame)
            else:
                messagebox.showwarning("Warning", f"Step {idx}: could not capture frame.")

        return captured_frames

    def register_from_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Student Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if not filepath:
            return

        student_id = simpledialog.askstring("Student ID", "Enter Student ID:")
        if not student_id:
            return

        name = simpledialog.askstring("Name", "Enter Full Name:")
        department = simpledialog.askstring("Department", "Enter Department:")
        year = simpledialog.askinteger("Year", "Enter Year (1-4):")

        success = self.container.registration_service.register_from_image(
            filepath, student_id, name, department, year
        )

        if success:
            self.container.reload_known_faces()
            messagebox.showinfo("Success", f"Registered {name}!")
            self.refresh_students()
        else:
            messagebox.showerror("Error", "Registration failed!")

    def add_student_dialog(self):
        messagebox.showinfo("Info", "Use 'Register Face' button in Live Detection tab.")

    # --- Data display ---

    def refresh_students(self):
        for item in self.students_tree.get_children():
            self.students_tree.delete(item)

        for student in self.student_repo.get_all():
            created = student.get('created_at', 'N/A')[:10] if student.get('created_at') else 'N/A'
            self.students_tree.insert('', tk.END, values=(
                student['id'], student['student_id'], student['name'],
                student.get('department', 'N/A'), student.get('year', 'N/A'), created
            ))

    def filter_students(self):
        search = self.student_search_var.get().lower()
        for item in self.students_tree.get_children():
            self.students_tree.delete(item)

        for student in self.student_repo.get_all():
            if (search in student['student_id'].lower() or
                search in student['name'].lower() or
                search in (student.get('department') or '').lower()):

                created = student.get('created_at', 'N/A')[:10] if student.get('created_at') else 'N/A'
                self.students_tree.insert('', tk.END, values=(
                    student['id'], student['student_id'], student['name'],
                    student.get('department', 'N/A'), student.get('year', 'N/A'), created
                ))

    def delete_student(self):
        selected = self.students_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a student to delete!")
            return

        item = self.students_tree.item(selected[0])
        student_id, name = item['values'][1], item['values'][2]

        if messagebox.askyesno("Confirm", f"Delete student {name} ({student_id})?"):
            self.student_repo.delete(student_id)
            self.container.reload_known_faces()
            self.refresh_students()
            messagebox.showinfo("Success", "Student deleted successfully!")

    def refresh_logs(self):
        for item in self.logs_tree.get_children():
            self.logs_tree.delete(item)

        date = self.log_date_var.get().strip() or None
        student_id = self.log_student_var.get().strip() or None
        logs = self.visit_repo.get_visit_logs(date=date, student_id=student_id)

        for log in logs:
            entry_time = log.get('entry_time', 'N/A')[:19] if log.get('entry_time') else 'N/A'
            status = "Known" if log.get('is_known') else "Unknown"
            duration = log.get('duration_minutes')
            duration_str = f"{duration} min" if duration else '-'

            self.logs_tree.insert('', tk.END, values=(
                log['id'], log.get('date', 'N/A'), entry_time,
                log.get('student_id', 'Unknown'), log.get('student_name', 'Unknown'),
                status, duration_str
            ))

    def clear_log_filters(self):
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
            logs = self.visit_repo.get_visit_logs(date=date, student_id=student_id)

            with open(filepath, 'w') as f:
                f.write("ID,Date,Entry Time,Student ID,Name,Status,Duration\n")
                for log in logs:
                    status = "Known" if log.get('is_known') else "Unknown"
                    f.write(f"{log['id']},{log.get('date', '')},{log.get('entry_time', '')},"
                           f"{log.get('student_id', '')},{log.get('student_name', '')},"
                           f"{status},{log.get('duration_minutes', '')}\n")

            messagebox.showinfo("Success", f"Logs exported to {filepath}")

    def refresh_statistics(self):
        stats = self.visit_repo.get_daily_statistics()
        students = self.student_repo.get_all()

        self.stat_students.config(text=str(len(students)))
        self.stat_today.config(text=str(stats['total_visits']))
        self.stat_unique.config(text=str(stats['unique_visitors']))
        self.stat_unknown.config(text=str(stats['unknown_visitors']))

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
        self.today_visits_label.config(text=f"Total Visits: {stats['total_visits']}")
        self.today_unique_label.config(text=f"Unique Visitors: {stats['unique_visitors']}")

    def refresh_all_data(self):
        self.refresh_students()
        self.refresh_logs()
        self.refresh_statistics()

    def take_screenshot(self):
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame to capture!")
            return

        screenshots_dir = os.path.join(os.path.dirname(__file__), "..", "..", "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        cv2.imwrite(os.path.join(screenshots_dir, filename), self.current_frame)
        messagebox.showinfo("Success", f"Screenshot saved: {filename}")

    def on_closing(self):
        self.is_running = False

        if hasattr(self, 'video_label') and hasattr(self.video_label, 'imgtk'):
            if self.video_label.imgtk is not None:
                del self.video_label.imgtk
                self.video_label.imgtk = None

        if hasattr(self, 'recent_detections'):
            self.recent_detections.clear()

        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.container.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
