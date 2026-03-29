"""
Visit logging module.
Keeps persistence concerns separate from recognition flow while preserving behavior.
"""

from typing import Callable
from datetime import datetime
import numpy as np

import config
import database


class VisitLogger:
    def __init__(self, save_screenshot_fn: Callable[[np.ndarray, dict], str], cooldown_seconds: int = 300):
        self.save_screenshot_fn = save_screenshot_fn
        self.cooldown_seconds = cooldown_seconds
        self.last_seen = {}

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
