"""Student repository - CRUD operations for student records"""

import logging
import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import Optional, List

from app.exceptions import DatabaseError, DuplicateStudentError, InvalidInputError
from app.utils.validators import sanitize_student_id

logger = logging.getLogger(__name__)


class StudentRepository:
    """Handles all database operations for student records.

    Dependencies are injected via the constructor (connection pool).
    """

    def __init__(self, pool):
        self.pool = pool

    def add(self, student_id: str, name: str, department: str, year: int,
            face_embedding: np.ndarray, face_image_path: str) -> bool:
        """Add a new student to the database."""
        if not student_id or not name:
            raise InvalidInputError("Student ID and name are required")

        if year and (year < 1 or year > 10):
            raise InvalidInputError("Year must be between 1 and 10")

        student_id = sanitize_student_id(student_id)

        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('SELECT id FROM students WHERE student_id = ?', (student_id,))
                if cursor.fetchone():
                    raise DuplicateStudentError(f"Student {student_id} already exists")

                embedding_json = json.dumps(face_embedding.tolist())
                cursor.execute('''
                    INSERT INTO students (student_id, name, department, year, face_embedding, face_image_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (student_id, name, department, year, embedding_json, face_image_path))
                conn.commit()
                logger.info("Student %s (%s) added successfully", name, student_id)
                return True

        except sqlite3.IntegrityError as e:
            raise DuplicateStudentError(f"Student {student_id} already exists: {str(e)}")
        except sqlite3.Error as e:
            raise DatabaseError(f"Database error adding student: {str(e)}")

    def get_all(self) -> List[dict]:
        """Get all students from database.

        Returns list of student dicts with face_embedding deserialized to np.ndarray.
        """
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM students ORDER BY name')
                rows = cursor.fetchall()

                students = []
                for row in rows:
                    student = dict(row)
                    if student['face_embedding']:
                        try:
                            student['face_embedding'] = np.array(json.loads(student['face_embedding']))
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning("Invalid embedding for student %s: %s", student.get('student_id'), e)
                            student['face_embedding'] = None
                    students.append(student)
                return students

        except sqlite3.Error as e:
            raise DatabaseError(f"Database error retrieving students: {str(e)}")

    def get_by_id(self, student_id: str) -> Optional[dict]:
        """Get student by student ID"""
        student_id = sanitize_student_id(student_id)

        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
                row = cursor.fetchone()

                if row:
                    student = dict(row)
                    if student['face_embedding']:
                        try:
                            student['face_embedding'] = np.array(json.loads(student['face_embedding']))
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning("Invalid embedding for student %s: %s", student_id, e)
                            student['face_embedding'] = None
                    return student
                return None

        except sqlite3.Error as e:
            raise DatabaseError(f"Database error retrieving student {student_id}: {str(e)}")

    def update(self, student_id: str, name: str = None, department: str = None,
               year: int = None, face_embedding: np.ndarray = None,
               face_image_path: str = None) -> bool:
        """Update student information"""
        student_id = sanitize_student_id(student_id)

        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                updates = []
                values = []

                if name:
                    updates.append("name = ?")
                    values.append(name)
                if department:
                    updates.append("department = ?")
                    values.append(department)
                if year:
                    updates.append("year = ?")
                    values.append(year)
                if face_embedding is not None:
                    updates.append("face_embedding = ?")
                    values.append(json.dumps(face_embedding.tolist()))
                if face_image_path:
                    updates.append("face_image_path = ?")
                    values.append(face_image_path)

                updates.append("updated_at = ?")
                values.append(datetime.now().isoformat())

                values.append(student_id)

                query = f"UPDATE students SET {', '.join(updates)} WHERE student_id = ?"
                cursor.execute(query, values)
                conn.commit()
                return True
        except Exception as e:
            logger.error("Error updating student: %s", e)
            return False

    def delete(self, student_id: str) -> bool:
        """Delete student from database"""
        student_id = sanitize_student_id(student_id)
        if not student_id:
            logger.warning("Delete called with empty/invalid student_id")
            return False

        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error("Error deleting student: %s", e)
            return False
