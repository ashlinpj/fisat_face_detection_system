"""Visit service - Visit logging with cooldown and screenshot management"""

import logging
import os
import cv2
import numpy as np
from datetime import datetime

import config
from app.repositories.connection_pool import ensure_directories

logger = logging.getLogger(__name__)


class VisitService:
    """Manages visit logging with cooldown tracking and screenshots.

    Dependencies (visit_repo) are injected via the constructor.
    """

    def __init__(self, visit_repo):
        self.visit_repo = visit_repo
        self.last_seen = {}
        self.active_visits = {}

    def should_log_visit(self, student_id: str, timestamp: datetime) -> bool:
        """Check if enough time has passed since last log for this student"""
        if student_id in self.last_seen:
            time_diff = (timestamp - self.last_seen[student_id]).total_seconds()
            cooldown = getattr(config, 'VISIT_COOLDOWN_SECONDS', 300)
            if time_diff < cooldown:
                return False
        return True

    def log_visit(self, student: dict, timestamp: datetime, face_crop: np.ndarray):
        """Log a visit if cooldown has passed"""
        student_id = student['student_id']

        if not self.should_log_visit(student_id, timestamp):
            logger.debug("Cooldown: %s (seen recently)", student['name'])
            return

        screenshot_path = self.save_visit_screenshot(face_crop, student)
        self.visit_repo.log_visit(
            student['id'], student_id, student['name'],
            screenshot_path=screenshot_path, is_known=True
        )
        self.last_seen[student_id] = timestamp
        logger.info("LOGGED: %s at %s", student['name'], timestamp.strftime('%H:%M:%S'))

    def save_visit_screenshot(self, face_crop: np.ndarray, student: dict) -> str:
        """Save face screenshot for visit log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visit_{student['student_id']}_{timestamp}.jpg"
        filepath = os.path.join(config.SCREENSHOTS_DIR, filename)
        ensure_directories()
        cv2.imwrite(filepath, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath
