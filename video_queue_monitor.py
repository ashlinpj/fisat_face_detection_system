"""
Video Queue Monitor – Shared Folder Ingestion Pipeline
======================================================
Monitors a configurable watch folder (local or network share) for new .mp4
segments, verifies file stability, then hands them to VideoProcessorQueue.

Safety features
---------------
* File-stability check (size must not change for N seconds)
* Rename to .processing while in-flight
* Move to processed/ (success) or error/ (failure)
* Crash recovery: resume .processing files on restart
* Graceful stop via threading event
"""

import os
import shutil
import time
import logging
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Callable, Dict

import config


class VideoQueueMonitor:
    """Watches a folder, validates file stability, feeds videos to processor."""

    def __init__(self):
        # ── directories ──────────────────────────────────────────
        self.watch_folder: str = getattr(config, "VIDEO_WATCH_FOLDER",
                                          os.path.join(config.BASE_DIR, "videos"))
        self.processing_dir: str = getattr(config, "VIDEO_PROCESSING_DIR",
                                            os.path.join(config.BASE_DIR, "processing"))
        self.processed_dir: str = os.path.join(config.BASE_DIR, "processed_videos")
        self.error_dir: str = getattr(config, "VIDEO_ERROR_DIR",
                                       os.path.join(config.BASE_DIR, "error"))
        self.logs_dir: str = os.path.join(config.BASE_DIR, "logs")

        # ── tuning ───────────────────────────────────────────────
        self.stability_wait: int = getattr(config, "VIDEO_STABILITY_WAIT_SECONDS", 60)
        self.stability_check_interval: int = getattr(config, "VIDEO_STABILITY_CHECK_INTERVAL", 10)
        self.scan_interval: int = getattr(config, "VIDEO_SCAN_INTERVAL", 60)
        self.supported_formats: list = getattr(config, "VIDEO_SUPPORTED_FORMATS",
                                                [".mp4", ".avi", ".mov", ".mkv", ".flv"])

        # ── state ────────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.is_running: bool = False
        self.current_file: Optional[str] = None
        self.pending_count: int = 0

        # ── callbacks (set by GUI / processor) ───────────────────
        # on_file_ready(processing_path: str) -> bool  (returns success)
        self.on_file_ready: Optional[Callable[[str], bool]] = None
        # on_status(message: str)
        self.on_status: Optional[Callable[[str], None]] = None
        # on_error(message: str)
        self.on_error: Optional[Callable[[str], None]] = None

        # ── setup ────────────────────────────────────────────────
        self._ensure_dirs()
        self._setup_logging()

    # ── directory helpers ─────────────────────────────────────────

    def _ensure_dirs(self):
        for d in (self.watch_folder, self.processing_dir,
                  self.processed_dir, self.error_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)

    def _setup_logging(self):
        log_file = os.path.join(self.logs_dir, "video_processing.log")
        # Avoid duplicate handlers on re-init
        logger = logging.getLogger()
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler(),
                ],
            )

    # ── public API ────────────────────────────────────────────────

    def start(self):
        """Start the monitor loop in a background thread."""
        if self.is_running:
            logging.warning("VideoQueueMonitor already running")
            return
        self._stop_event.clear()
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                         name="VideoQueueMonitor")
        self._thread.start()
        logging.info(f"VideoQueueMonitor started – watching: {self.watch_folder}")

    def stop(self):
        """Gracefully stop the monitor."""
        self._stop_event.set()
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=10)
        logging.info("VideoQueueMonitor stopped")

    # ── scanning ──────────────────────────────────────────────────

    def scan_watch_folder(self) -> List[str]:
        """Return list of candidate video paths sorted oldest-first."""
        if not os.path.exists(self.watch_folder):
            return []
        result = []
        try:
            for fname in os.listdir(self.watch_folder):
                fpath = os.path.join(self.watch_folder, fname)
                if not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.supported_formats:
                    continue
                if fname.startswith(".") or fname.startswith("~"):
                    continue
                result.append(fpath)
        except OSError as e:
            logging.error(f"Cannot access watch folder: {e}")
            if self.on_error:
                self.on_error(f"Watch folder unavailable: {e}")
            return []
        result.sort(key=lambda p: os.path.getmtime(p))
        return result

    def scan_processing_recovery(self) -> List[str]:
        """Find .processing files left over from a crash."""
        recovered = []
        if not os.path.exists(self.processing_dir):
            return recovered
        for fname in os.listdir(self.processing_dir):
            if fname.endswith(".processing"):
                recovered.append(os.path.join(self.processing_dir, fname))
        recovered.sort(key=lambda p: os.path.getmtime(p))
        return recovered

    # ── file stability ────────────────────────────────────────────

    def _is_file_stable(self, path: str) -> bool:
        """
        Return True only when the file size has not changed for
        ``stability_wait`` seconds.  Checks every ``stability_check_interval``.
        """
        try:
            prev_size = os.path.getsize(path)
            prev_mtime = os.path.getmtime(path)
        except OSError:
            return False

        elapsed = 0
        while elapsed < self.stability_wait:
            if self._stop_event.is_set():
                return False
            time.sleep(self.stability_check_interval)
            elapsed += self.stability_check_interval
            try:
                cur_size = os.path.getsize(path)
                cur_mtime = os.path.getmtime(path)
            except OSError:
                return False  # file disappeared
            if cur_size != prev_size or cur_mtime != prev_mtime:
                # Still being written – reset
                prev_size = cur_size
                prev_mtime = cur_mtime
                elapsed = 0
                if self.on_status:
                    self.on_status(f"Waiting for file to finish writing: {os.path.basename(path)}")
        return True

    # ── move helpers ──────────────────────────────────────────────

    def _move_to_processing(self, src: str) -> Optional[str]:
        """Move file from watch folder to processing/ and rename .processing."""
        fname = os.path.basename(src)
        name, _ext = os.path.splitext(fname)
        dest = os.path.join(self.processing_dir, f"{name}.processing")
        try:
            shutil.move(src, dest)
            logging.info(f"Moved to processing: {fname} -> {os.path.basename(dest)}")
            return dest
        except Exception as e:
            logging.error(f"Failed to move {fname} to processing: {e}")
            return None

    def mark_success(self, processing_path: str):
        """Move .processing file to processed/ as _done.mp4."""
        fname = os.path.basename(processing_path)
        name = fname.replace(".processing", "")
        dest = os.path.join(self.processed_dir, f"{name}_done.mp4")
        try:
            shutil.move(processing_path, dest)
            logging.info(f"Processed OK: {fname} -> {os.path.basename(dest)}")
        except Exception as e:
            logging.error(f"Failed to move to processed: {e}")

    def mark_error(self, processing_path: str):
        """Move .processing file to error/."""
        fname = os.path.basename(processing_path)
        dest = os.path.join(self.error_dir, fname)
        try:
            shutil.move(processing_path, dest)
            logging.error(f"Moved to error: {fname}")
        except Exception as e:
            logging.error(f"Failed to move to error: {e}")

    # ── main loop ─────────────────────────────────────────────────

    def _run_loop(self):
        """Background loop: scan → stabilise → hand off → repeat."""
        # ── crash recovery first ──────────────────────────────────
        recovered = self.scan_processing_recovery()
        if recovered:
            logging.info(f"Recovering {len(recovered)} files from previous crash")
            for path in recovered:
                if self._stop_event.is_set():
                    break
                self._handle_file(path, already_in_processing=True)

        # ── normal loop ───────────────────────────────────────────
        while not self._stop_event.is_set():
            candidates = self.scan_watch_folder()
            self.pending_count = len(candidates)

            if not candidates:
                if self.on_status:
                    self.on_status("Waiting for new videos...")
                logging.info("No new videos found. Waiting...")
                self._interruptible_sleep(self.scan_interval)
                continue

            for video_path in candidates:
                if self._stop_event.is_set():
                    break
                self._handle_file(video_path, already_in_processing=False)

            # tiny pause between full scans
            self._interruptible_sleep(2)

        logging.info("VideoQueueMonitor loop exited")

    def _handle_file(self, path: str, already_in_processing: bool = False):
        """Process a single file through the stability→process→move pipeline."""
        fname = os.path.basename(path)
        self.current_file = fname

        if not already_in_processing:
            # Stability check on original location
            if self.on_status:
                self.on_status(f"Checking stability: {fname}")
            logging.info(f"Stability check: {fname}")

            if not self._is_file_stable(path):
                logging.warning(f"File not stable or disappeared: {fname}")
                self.current_file = None
                return

            # Move to processing/
            processing_path = self._move_to_processing(path)
            if processing_path is None:
                self.current_file = None
                return
        else:
            processing_path = path
            logging.info(f"Recovering crashed file: {fname}")

        # Hand off to processor callback
        if self.on_file_ready:
            try:
                if self.on_status:
                    self.on_status(f"Processing: {fname}")
                success = self.on_file_ready(processing_path)
                if success:
                    self.mark_success(processing_path)
                else:
                    self.mark_error(processing_path)
            except Exception as e:
                logging.error(f"Processing callback error for {fname}: {e}")
                logging.error(traceback.format_exc())
                self.mark_error(processing_path)
        else:
            logging.warning("No on_file_ready callback set – skipping file")
            self.mark_error(processing_path)

        self.current_file = None

    def _interruptible_sleep(self, seconds: int):
        """Sleep in 1-second increments, checking stop event."""
        for _ in range(seconds):
            if self._stop_event.is_set():
                break
            time.sleep(1)

    # ── status helpers ────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "is_running": self.is_running,
            "watch_folder": self.watch_folder,
            "current_file": self.current_file,
            "pending_count": self.pending_count,
        }
