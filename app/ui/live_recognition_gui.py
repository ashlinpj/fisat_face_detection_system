"""Lightweight live recognition board GUI.

Displays only currently detected people on a black background:
- Known people in green
- Unknown people in red
"""

import os
import threading
import time
import tkinter as tk

import cv2

import config


class LiveRecognitionBoardGUI:
    """Simple black-background GUI for live recognition status."""

    def __init__(self, container):
        self.container = container
        self.cap = None
        self.is_running = False
        self.capture_thread = None

        self.root = tk.Tk()
        self.root.title("Live Recognition Board")
        self.root.geometry("760x560")
        self.root.minsize(520, 420)
        self.root.configure(bg="black")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        main = tk.Frame(self.root, bg="black")
        main.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        title_label = tk.Label(
            main,
            text="CURRENT RECOGNITIONS",
            bg="black",
            fg="white",
            font=("Segoe UI", 20, "bold")
        )
        title_label.pack(anchor="w", pady=(0, 8))

        self.status_label = tk.Label(
            main,
            text="Starting camera...",
            bg="black",
            fg="#9fa3a9",
            font=("Segoe UI", 11)
        )
        self.status_label.pack(anchor="w", pady=(0, 10))

        self.counts_label = tk.Label(
            main,
            text="Known: 0 | Unknown: 0",
            bg="black",
            fg="white",
            font=("Consolas", 12, "bold")
        )
        self.counts_label.pack(anchor="w", pady=(0, 10))

        self.names_text = tk.Text(
            main,
            bg="black",
            fg="#57f287",
            insertbackground="white",
            relief=tk.FLAT,
            borderwidth=0,
            font=("Consolas", 18, "bold"),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.names_text.pack(fill=tk.BOTH, expand=True)

        self.names_text.tag_configure("known", foreground="#57f287")
        self.names_text.tag_configure("unknown", foreground="#ff5f56")
        self.names_text.tag_configure("muted", foreground="#9fa3a9")

        footer = tk.Label(
            main,
            text="Known: green | Unknown: red",
            bg="black",
            fg="#9fa3a9",
            font=("Segoe UI", 10)
        )
        footer.pack(anchor="w", pady=(10, 0))

    def _open_capture(self):
        if config.USE_RTSP:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
            return cap

        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        return cap

    def start(self):
        self.cap = self._open_capture()
        if not self.cap or not self.cap.isOpened():
            self.status_label.config(text="Camera not available", fg="#ff5f56")
            return

        self.is_running = True
        self.status_label.config(text="Live detection running", fg="#57f287")
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.08)
                continue

            _, recognized_people = self.container.frame_processor.process_frame(frame)
            rows = self._build_current_rows(recognized_people)
            self.root.after(0, self._render_rows, rows)
            time.sleep(0.03)

    def _build_current_rows(self, recognized_people):
        rows = []
        seen = set()

        for person in recognized_people:
            is_known = bool(person.get("is_known"))
            if is_known:
                name = person.get("name") or "Known Person"
            else:
                name = "Unknown"

            key = (name, is_known)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"name": name, "is_known": is_known})

        return rows

    def _render_rows(self, rows):
        known_count = sum(1 for row in rows if row["is_known"])
        unknown_count = len(rows) - known_count
        self.counts_label.config(text=f"Known: {known_count} | Unknown: {unknown_count}")

        self.names_text.config(state=tk.NORMAL)
        self.names_text.delete("1.0", tk.END)

        if not rows:
            self.names_text.insert(tk.END, "No people currently detected", "muted")
        else:
            for idx, row in enumerate(rows, start=1):
                tag = "known" if row["is_known"] else "unknown"
                self.names_text.insert(tk.END, f"{idx}. {row['name']}\n", tag)

        self.names_text.config(state=tk.DISABLED)

    def on_close(self):
        self.is_running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()
        self.container.stop()
        self.root.destroy()

    def run(self):
        self.start()
        self.root.mainloop()
