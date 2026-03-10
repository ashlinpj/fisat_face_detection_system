"""Domain entities for the Face Detection System.

These dataclasses define the core data structures used throughout the application.
They support dict-like access (student['name']) for backward compatibility
with code that previously used plain dicts.
"""

from dataclasses import dataclass, fields
from typing import Optional, Any
import numpy as np


class DictMixin:
    """Adds dict-like access for backward compatibility with dict-based code."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        return [f.name for f in fields(self)]

    def values(self):
        return [getattr(self, f.name) for f in fields(self)]

    def items(self):
        return [(f.name, getattr(self, f.name)) for f in fields(self)]


@dataclass
class Student(DictMixin):
    """Represents a registered student."""
    id: int = 0
    student_id: str = ""
    name: str = ""
    department: Optional[str] = None
    year: Optional[int] = None
    face_embedding: Optional[np.ndarray] = None
    face_image_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class Visit(DictMixin):
    """Represents a canteen visit log entry."""
    id: int = 0
    student_db_id: Optional[int] = None
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    duration_minutes: Optional[int] = None
    date: Optional[str] = None
    is_known: bool = True
    screenshot_path: Optional[str] = None


@dataclass
class UnknownFace(DictMixin):
    """Represents an unrecognized face captured by the system."""
    id: int = 0
    face_image_path: Optional[str] = None
    face_embedding: Optional[np.ndarray] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    times_seen: int = 1
