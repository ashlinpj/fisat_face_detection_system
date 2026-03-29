"""
Camera pipeline module.
Encapsulates camera source selection, capture threads, buffering, reconnects,
and multi-camera state management without changing runtime behavior.
"""

import os
import cv2
import time
import queue
from threading import Thread

import config


class CameraPipeline:
    def __init__(self):
        self.cap = None
        self.active_source = config.CAMERA_INDEX
        self.frame_buffer = queue.Queue(maxsize=config.FRAME_BUFFER_SIZE)
        self.capture_thread = None
        self.capture_thread_running = False
        self.multi_cameras = []

    def is_rtsp_source(self, source):
        return isinstance(source, str) and source.lower().startswith("rtsp://")

    def is_network_source(self, source):
        if not isinstance(source, str):
            return False
        lowered = source.lower()
        return (
            lowered.startswith("rtsp://")
            or lowered.startswith("http://")
            or lowered.startswith("https://")
        )

    def normalize_source(self, source):
        if isinstance(source, int):
            return source
        if isinstance(source, str):
            stripped = source.strip()
            if stripped.isdigit():
                return int(stripped)
            return stripped
        return source

    def get_camera_sources(self):
        configured_sources = [
            self.normalize_source(s)
            for s in getattr(config, "CAMERA_SOURCES", [])
            if str(s).strip() != ""
        ]

        if getattr(config, "USE_MULTI_CAMERA", False) and configured_sources:
            return configured_sources
        if len(configured_sources) > 1:
            return configured_sources
        if config.USE_RTSP:
            return [config.RTSP_URL]
        if configured_sources:
            return [configured_sources[0]]
        return [config.CAMERA_INDEX]

    def is_multi_camera_enabled(self):
        return len(self.get_camera_sources()) > 1

    def source_label(self, source, index):
        if isinstance(source, int):
            return f"Cam-{index + 1} (USB:{source})"
        text = str(source)
        if len(text) > 45:
            text = text[:42] + "..."
        return f"Cam-{index + 1} ({text})"

    def open_capture_for_source(self, source):
        is_rtsp = self.is_rtsp_source(source)

        if is_rtsp:
            rtsp_opts = [f"{k};{v}" for k, v in config.RTSP_OPENCV_OPTIONS.items()]
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(rtsp_opts)
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
            if config.RTSP_TARGET_FPS:
                cap.set(cv2.CAP_PROP_FPS, config.RTSP_TARGET_FPS)
        else:
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)

        if not cap.isOpened():
            return None

        # Ask OpenCV to use hardware decoding path when backend supports it.
        if getattr(config, "RTSP_USE_HW_ACCEL", True):
            try:
                if hasattr(cv2, "CAP_PROP_HW_ACCELERATION") and hasattr(cv2, "VIDEO_ACCELERATION_ANY"):
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                if hasattr(cv2, "CAP_PROP_HW_DEVICE"):
                    cap.set(cv2.CAP_PROP_HW_DEVICE, 0)
            except Exception:
                pass

        ret, _ = cap.read()
        if not ret:
            cap.release()
            return None
        return cap

    def new_camera_state(self, source, index):
        return {
            "source": source,
            "label": self.source_label(source, index),
            "cap": None,
            "frame_buffer": queue.Queue(maxsize=config.FRAME_BUFFER_SIZE),
            "thread": None,
            "thread_running": False,
            "failed_reads": 0,
            "last_raw_frame": None,
        }

    def start_multi_capture_thread(self, state):
        self.stop_multi_capture_thread(state)
        state["thread_running"] = True

        def capture_loop():
            while state["thread_running"] and state["cap"]:
                try:
                    if self.is_rtsp_source(state["source"]) and config.RTSP_PREGRAB_COUNT > 0:
                        for _ in range(config.RTSP_PREGRAB_COUNT):
                            state["cap"].grab()

                    ret, frame = state["cap"].read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                    if state["frame_buffer"].full():
                        try:
                            state["frame_buffer"].get_nowait()
                        except queue.Empty:
                            pass

                    try:
                        state["frame_buffer"].put_nowait(frame)
                    except queue.Full:
                        pass
                except Exception:
                    time.sleep(0.01)

        state["thread"] = Thread(target=capture_loop, daemon=True)
        state["thread"].start()

    def stop_multi_capture_thread(self, state):
        state["thread_running"] = False
        if state.get("thread"):
            state["thread"].join(timeout=1.0)
        state["thread"] = None

    def get_latest_frame_from_state(self, state):
        try:
            frame = state["frame_buffer"].get(timeout=config.FRAME_FETCH_TIMEOUT)
        except queue.Empty:
            return None

        while not state["frame_buffer"].empty():
            try:
                frame = state["frame_buffer"].get_nowait()
            except queue.Empty:
                break
        return frame

    def reconnect_camera_state(self, state):
        self.stop_multi_capture_thread(state)
        if state["cap"]:
            state["cap"].release()
            state["cap"] = None

        source = state["source"]
        attempts = config.RTSP_RECONNECT_ATTEMPTS if self.is_network_source(source) else 2
        for attempt in range(attempts):
            cap = self.open_capture_for_source(source)
            if cap is not None:
                state["cap"] = cap
                state["failed_reads"] = 0
                self.start_multi_capture_thread(state)
                print(f"✓ Reconnected {state['label']}")
                return True
            if attempt < attempts - 1:
                time.sleep(config.RTSP_RECONNECT_DELAY)

        print(f"✗ Reconnect failed for {state['label']}")
        return False

    def start_multi_camera(self):
        self.stop_multi_camera()
        sources = self.get_camera_sources()

        print("Connecting to camera sources:")
        connected = 0
        self.multi_cameras = []

        for idx, source in enumerate(sources):
            state = self.new_camera_state(source, idx)
            cap = self.open_capture_for_source(source)
            if cap is None:
                print(f"  ✗ Could not open {state['label']}")
                continue

            state["cap"] = cap
            self.start_multi_capture_thread(state)
            self.multi_cameras.append(state)
            connected += 1
            print(f"  ✓ Connected {state['label']}")

        if connected == 0:
            print("ERROR: No camera source could be opened.")
            return False

        print(f"Connected {connected}/{len(sources)} camera source(s).")
        return True

    def stop_multi_camera(self):
        for state in self.multi_cameras:
            self.stop_multi_capture_thread(state)
            if state.get("cap"):
                state["cap"].release()
                state["cap"] = None
        self.multi_cameras = []

    def get_registration_frame_from_multi(self):
        for state in self.multi_cameras:
            if state.get("last_raw_frame") is not None:
                return state["last_raw_frame"].copy()
        return None

    def start_camera(self):
        """Start the camera capture (webcam or RTSP stream)."""
        source = self.get_camera_sources()[0]
        self.active_source = source
        is_rtsp = self.is_rtsp_source(source)

        if is_rtsp:
            print(f"Connecting to RTSP stream: {source}")

            rtsp_opts = [f"{k};{v}" for k, v in config.RTSP_OPENCV_OPTIONS.items()]
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(rtsp_opts)

            for attempt in range(config.RTSP_RECONNECT_ATTEMPTS):
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
                if config.RTSP_TARGET_FPS:
                    self.cap.set(cv2.CAP_PROP_FPS, config.RTSP_TARGET_FPS)

                if self.cap.isOpened():
                    ret, _ = self.cap.read()
                    if ret:
                        print("✓ RTSP stream connected successfully!")
                        print(f"  Stream: {source}")
                        print(
                            f"  Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                        )
                        self.clear_frame_buffer()
                        self.start_capture_thread()
                        return True

                print(f"  Attempt {attempt + 1}/{config.RTSP_RECONNECT_ATTEMPTS} failed...")
                if attempt < config.RTSP_RECONNECT_ATTEMPTS - 1:
                    time.sleep(config.RTSP_RECONNECT_DELAY)

            print("\nERROR: Could not connect to RTSP stream!")
            print("Please check:")
            print("  1. RTSP server is running (e.g., OBS with RTSP output)")
            print("  2. RTSP URL is correct")
            print(f"  3. Current URL: {source}")
            print("  4. Firewall is not blocking the connection")
            print("\nTip: To use webcam instead, set USE_RTSP = False in config.py")
            return False

        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)

        if not self.cap.isOpened():
            print("ERROR: Could not open camera!")
            print("Please check:")
            print("  1. Camera is connected")
            print("  2. Camera is not being used by another application")
            print(f"  3. Camera source is correct (current: {source})")
            return False

        print(f"✓ Camera opened successfully (Source: {source})")
        self.clear_frame_buffer()
        self.start_capture_thread()
        return True

    def stop_camera(self):
        """Stop the camera capture."""
        self.stop_capture_thread()
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def start_capture_thread(self):
        """Continuously read frames into a tiny buffer to avoid overlap/stale frames."""
        self.stop_capture_thread()
        self.capture_thread_running = True

        def capture_loop():
            while self.capture_thread_running and self.cap:
                try:
                    if self.is_network_source(self.active_source):
                        drain_count = int(getattr(config, "RTSP_PREGRAB_COUNT", 0))
                        if getattr(config, "REALTIME_ONLY_MODE", False):
                            drain_count = max(drain_count, 1)
                        for _ in range(max(0, drain_count)):
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

    def stop_capture_thread(self):
        self.capture_thread_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        self.capture_thread = None

    def clear_frame_buffer(self):
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break

    def get_latest_frame(self):
        """Fetch the most recent frame, dropping any stale queued frames."""
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
