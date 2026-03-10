"""Input validation utilities for the Face Detection System"""

import re
from typing import Optional, Tuple
import numpy as np


class ValidationError(Exception):
    """Raised when input validation fails"""
    pass


def validate_student_id(student_id: str) -> Tuple[bool, Optional[str]]:
    """Validate student ID format."""
    if not student_id:
        return False, "Student ID cannot be empty"
    if len(student_id) < 3:
        return False, "Student ID must be at least 3 characters"
    if len(student_id) > 20:
        return False, "Student ID must be at most 20 characters"
    if not re.match(r'^[A-Za-z0-9_-]+$', student_id):
        return False, "Student ID can only contain letters, numbers, dash, and underscore"
    return True, None


def validate_student_name(name: str) -> Tuple[bool, Optional[str]]:
    """Validate student name."""
    if not name:
        return False, "Name cannot be empty"
    if len(name) < 2:
        return False, "Name must be at least 2 characters"
    if len(name) > 100:
        return False, "Name must be at most 100 characters"
    if not re.match(r"^[A-Za-z\s.'-]+$", name):
        return False, "Name contains invalid characters"
    return True, None


def validate_department(department: str) -> Tuple[bool, Optional[str]]:
    """Validate department code."""
    if not department:
        return True, None
    if len(department) > 50:
        return False, "Department code must be at most 50 characters"
    return True, None


def validate_year(year: int) -> Tuple[bool, Optional[str]]:
    """Validate year of study."""
    if year is None:
        return True, None
    if not isinstance(year, int):
        return False, "Year must be an integer"
    if year < 1 or year > 10:
        return False, "Year must be between 1 and 10"
    return True, None


def validate_face_embedding(embedding: np.ndarray) -> Tuple[bool, Optional[str]]:
    """Validate face embedding array."""
    if embedding is None:
        return False, "Embedding cannot be None"
    if not isinstance(embedding, np.ndarray):
        return False, "Embedding must be a numpy array"
    if embedding.size == 0:
        return False, "Embedding cannot be empty"
    if len(embedding.shape) != 1:
        return False, "Embedding must be 1-dimensional"
    if np.any(np.isnan(embedding)):
        return False, "Embedding contains NaN values"
    if np.any(np.isinf(embedding)):
        return False, "Embedding contains infinite values"
    return True, None


def validate_confidence_threshold(threshold: float) -> Tuple[bool, Optional[str]]:
    """Validate confidence threshold value."""
    if not isinstance(threshold, (int, float)):
        return False, "Threshold must be a number"
    if threshold < 0 or threshold > 1:
        return False, "Threshold must be between 0 and 1"
    return True, None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and invalid characters."""
    filename = filename.replace('/', '_').replace('\\', '_')
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        name = name[:250]
        filename = f"{name}.{ext}" if ext else name
    return filename


def sanitize_student_id(student_id: str) -> str:
    """Sanitize student ID to prevent path traversal and invalid filenames."""
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', student_id)
    return sanitized[:20]


def validate_frame_dimensions(frame: np.ndarray,
                              min_width: int = 160,
                              min_height: int = 160) -> Tuple[bool, Optional[str]]:
    """Validate video frame dimensions."""
    if frame is None or frame.size == 0:
        return False, "Frame is empty"
    if len(frame.shape) not in [2, 3]:
        return False, "Frame must be 2D or 3D array"
    height, width = frame.shape[:2]
    if width < min_width:
        return False, f"Frame width {width} is less than minimum {min_width}"
    if height < min_height:
        return False, f"Frame height {height} is less than minimum {min_height}"
    return True, None


def validate_bounding_box(bbox: tuple,
                          frame_width: int,
                          frame_height: int) -> Tuple[bool, Optional[str]]:
    """Validate bounding box coordinates."""
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return False, "Bounding box must be a tuple of 4 values (x1, y1, x2, y2)"
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0:
        return False, "Bounding box coordinates cannot be negative"
    if x2 > frame_width or y2 > frame_height:
        return False, f"Bounding box exceeds frame dimensions ({frame_width}x{frame_height})"
    if x2 <= x1 or y2 <= y1:
        return False, "Invalid bounding box: x2 must be > x1 and y2 must be > y1"
    return True, None
