"""
Visit logging module.
Keeps persistence concerns separate from recognition flow while preserving behavior.
"""

import os
from typing import Callable
from datetime import datetime
import numpy as np
import cv2

import config
import database


class VisitLogger:
    def __init__(self, save_screenshot_fn: Callable[[np.ndarray, dict], str], cooldown_seconds: int = 300):
        self.save_screenshot_fn = save_screenshot_fn
        self.cooldown_seconds = cooldown_seconds
        self.last_seen = {}
        self.unknown_cooldown_seconds = int(getattr(config, 'UNKNOWN_FACE_SAVE_COOLDOWN_SECONDS', 15))
        self.last_unknown_saved_at = None

    def try_log_known_visit(self, student: dict, confidence: float, timestamp: datetime, face_crop: np.ndarray):
        """Log known visits with the same cooldown and console messages as before."""
        if not student or confidence <= config.FACE_RECOGNITION_THRESHOLD:
            return

        student_id = student["student_id"]
        should_log = True

        if student_id in self.last_seen:
            time_diff = (timestamp - self.last_seen[student_id]).total_seconds()
            if time_diff < self.cooldown_seconds:
                should_log = False

        if should_log:
            screenshot_path = self.save_screenshot_fn(face_crop, student)
            database.log_visit(
                student["id"],
                student_id,
                student["name"],
                screenshot_path=screenshot_path,
                is_known=True,
            )
            self.last_seen[student_id] = timestamp
            print(f"✓ LOGGED: {student['name']} at {timestamp.strftime('%H:%M:%S')}")
        else:
            print(f"  Cooldown: {student['name']} (seen recently)")

    def _save_unknown_snapshot(self, face_crop: np.ndarray, timestamp: datetime) -> str:
        """Persist unknown face crop to disk and return file path."""
        database.ensure_directories()
        unknown_dir = getattr(config, 'UNKNOWN_FACES_DIR', os.path.join(config.FACES_DIR, 'unknown_faces'))
        os.makedirs(unknown_dir, exist_ok=True)

        filename = f"unknown_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(unknown_dir, filename)
        cv2.imwrite(filepath, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath

    def try_log_unknown_face(self, timestamp: datetime, face_crop: np.ndarray, embedding: np.ndarray):
        """Save unknown faces with global cooldown to avoid duplicate spam."""
        if face_crop is None or embedding is None:
            return

        if self.last_unknown_saved_at is not None:
            elapsed = (timestamp - self.last_unknown_saved_at).total_seconds()
            if elapsed < self.unknown_cooldown_seconds:
                return

        image_path = self._save_unknown_snapshot(face_crop, timestamp)
        face_id = database.add_unknown_face(image_path, embedding)
        if face_id != -1:
            self.last_unknown_saved_at = timestamp
            print(f"✓ UNKNOWN SAVED: #{face_id} at {timestamp.strftime('%H:%M:%S')} (cooldown {self.unknown_cooldown_seconds}s)")
