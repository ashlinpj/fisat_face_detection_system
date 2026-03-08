"""Database module for Face Detection System"""

import sqlite3
import os
import json
import re
import numpy as np
from datetime import datetime
from typing import Optional, List
import config


# Custom Exceptions
class DatabaseError(Exception):
    """Base exception for database operations"""
    pass


class StudentNotFoundError(DatabaseError):
    """Student not found in database"""
    pass


class DuplicateStudentError(DatabaseError):
    """Student already exists in database"""
    pass


class InvalidInputError(DatabaseError):
    """Invalid input data"""
    pass


def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)
    os.makedirs(config.FACES_DIR, exist_ok=True)


def get_connection():
    """Get database connection"""
    ensure_directories()
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def sanitize_student_id(student_id: str) -> str:
    """
    Sanitize student ID to prevent path traversal and invalid filenames.
    
    Parameters
    ----------
    student_id : str
        Raw student ID input
        
    Returns
    -------
    str
        Sanitized student ID (alphanumeric, dash, underscore only)
    """
    # Remove any character that isn't alphanumeric, dash, or underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', student_id)
    
    # Limit length
    return sanitized[:20]


def init_database():
    """Initialize database with required tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT,
            year INTEGER,
            face_embedding TEXT,
            face_image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS visit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_db_id INTEGER,
            student_id TEXT,
            student_name TEXT,
            entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            exit_time TIMESTAMP,
            duration_minutes INTEGER,
            date DATE DEFAULT (DATE('now', 'localtime')),
            is_known INTEGER DEFAULT 1,
            screenshot_path TEXT,
            FOREIGN KEY (student_db_id) REFERENCES students(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS unknown_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_image_path TEXT,
            face_embedding TEXT,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            times_seen INTEGER DEFAULT 1
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")


def add_student(student_id: str, name: str, department: str, year: int, 
                face_embedding: np.ndarray, face_image_path: str) -> bool:
    """
    Add a new student to the database.
    
    Parameters
    ----------
    student_id : str
        Unique student identifier
    name : str
        Full name of the student
    department : str
        Department code
    year : int
        Year of study
    face_embedding : np.ndarray
        Face embedding vector
    face_image_path : str
        Path to stored face image
        
    Returns
    -------
    bool
        True if successful, False otherwise
        
    Raises
    ------
    DuplicateStudentError
        If student_id already exists
    InvalidInputError
        If input validation fails
    """
    # Validate inputs
    if not student_id or not name:
        raise InvalidInputError("Student ID and name are required")
    
    if year and (year < 1 or year > 10):
        raise InvalidInputError("Year must be between 1 and 10")
    
    # Sanitize student_id to prevent path traversal
    student_id = sanitize_student_id(student_id)
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check for existing student
        cursor.execute('SELECT id FROM students WHERE student_id = ?', (student_id,))
        if cursor.fetchone():
            raise DuplicateStudentError(f"Student {student_id} already exists")
        
        embedding_json = json.dumps(face_embedding.tolist())
        cursor.execute('''
            INSERT INTO students (student_id, name, department, year, face_embedding, face_image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (student_id, name, department, year, embedding_json, face_image_path))
        conn.commit()
        print(f"Student {name} ({student_id}) added successfully!")
        return True
        
    except sqlite3.IntegrityError as e:
        raise DuplicateStudentError(f"Student {student_id} already exists: {str(e)}")
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error adding student: {str(e)}")
    finally:
        if conn:
            conn.close()

def get_all_students() -> List[dict]:
    """
    Get all students from database.
    
    Returns
    -------
    List[dict]
        List of student dictionaries with face embeddings
        
    Raises
    ------
    DatabaseError
        If database query fails
    """
    conn = None
    try:
        conn = get_connection()
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
                    print(f"Warning: Invalid embedding for student {student.get('student_id')}: {e}")
                    student['face_embedding'] = None
            students.append(student)
        
        return students
        
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error retrieving students: {str(e)}")
    finally:
        if conn:
            conn.close()

def get_student_by_id(student_id: str) -> Optional[dict]:
    """
    Get student by student ID.
    
    Parameters
    ----------
    student_id : str
        Student identifier
        
    Returns
    -------
    Optional[dict]
        Student dictionary if found, None otherwise
        
    Raises
    ------
    DatabaseError
        If database query fails
    """
    # Sanitize student_id for safe database query
    student_id = sanitize_student_id(student_id)
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
        row = cursor.fetchone()
        
        if row:
            student = dict(row)
            if student['face_embedding']:
                try:
                    student['face_embedding'] = np.array(json.loads(student['face_embedding']))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Warning: Invalid embedding for student {student_id}: {e}")
                    student['face_embedding'] = None
            return student
        return None
        
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error retrieving student {student_id}: {str(e)}")
    finally:
        if conn:
            conn.close()

def update_student(student_id: str, name: str = None, department: str = None, 
                   year: int = None, face_embedding: np.ndarray = None, 
                   face_image_path: str = None) -> bool:
    """Update student information"""
    # Sanitize student_id
    student_id = sanitize_student_id(student_id)
    
    try:
        conn = get_connection()
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
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating student: {e}")
        return False

def delete_student(student_id: str) -> bool:
    """Delete student from database"""
    # Sanitize student_id
    student_id = sanitize_student_id(student_id)
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting student: {e}")
        return False


def log_visit(student_db_id: int, student_id: str, student_name: str, 
              screenshot_path: str = None, is_known: bool = True) -> int:
    """
    Log a canteen visit with screenshot.
    
    Parameters
    ----------
    student_db_id : int
        Student's database ID
    student_id : str
        Student identifier
    student_name : str
        Student name
    screenshot_path : str, optional
        Path to visit screenshot
    is_known : bool
        Whether student is known/registered
        
    Returns
    -------
    int
        Log ID if successful, -1 on error
        
    Raises
    ------
    DatabaseError
        If database operation fails
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO visit_logs (student_db_id, student_id, student_name, is_known, screenshot_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (student_db_id, student_id, student_name, 1 if is_known else 0, screenshot_path))
        log_id = cursor.lastrowid
        conn.commit()
        return log_id
        
    except sqlite3.Error as e:
        print(f"Error logging visit: {e}")
        raise DatabaseError(f"Failed to log visit: {str(e)}")
    finally:
        if conn:
            conn.close()


def update_visit_exit(log_id: int):
    """Update visit log with exit time"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE visit_logs 
            SET exit_time = CURRENT_TIMESTAMP,
                duration_minutes = CAST((JULIANDAY(CURRENT_TIMESTAMP) - JULIANDAY(entry_time)) * 24 * 60 AS INTEGER)
            WHERE id = ?
        ''', (log_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating visit exit: {e}")


def get_visit_logs(date: str = None, student_id: str = None) -> List[dict]:
    """Get visit logs with optional filters"""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = 'SELECT * FROM visit_logs WHERE 1=1'
    params = []
    
    if date:
        query += ' AND date = ?'
        params.append(date)
    if student_id:
        query += ' AND student_id = ?'
        params.append(student_id)
    
    query += ' ORDER BY entry_time DESC'
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    logs = [dict(row) for row in rows]
    conn.close()
    return logs

def get_recent_visit(student_id: str) -> Optional[dict]:
    """Get the most recent visit for a student"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM visit_logs 
        WHERE student_id = ? 
        ORDER BY entry_time DESC 
        LIMIT 1
    ''', (student_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None


def add_unknown_face(face_image_path: str, face_embedding: np.ndarray) -> int:
    """Add unknown face for later registration"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        embedding_json = json.dumps(face_embedding.tolist())
        cursor.execute('''
            INSERT INTO unknown_faces (face_image_path, face_embedding)
            VALUES (?, ?)
        ''', (face_image_path, embedding_json))
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return face_id
    except Exception as e:
        print(f"Error adding unknown face: {e}")
        return -1


def get_unknown_faces() -> List[dict]:
    """Get all unknown faces"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM unknown_faces ORDER BY last_seen DESC')
    rows = cursor.fetchall()
    
    faces = []
    for row in rows:
        face = dict(row)
        if face['face_embedding']:
            face['face_embedding'] = np.array(json.loads(face['face_embedding']))
        faces.append(face)
    
    conn.close()
    return faces

def get_daily_statistics(date: str = None) -> dict:
    """Get daily visit statistics"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM visit_logs WHERE date = ?', (date,))
    total_visits = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT student_id) FROM visit_logs WHERE date = ? AND is_known = 1', (date,))
    unique_visitors = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM visit_logs WHERE date = ? AND is_known = 0', (date,))
    unknown_visitors = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(duration_minutes) FROM visit_logs WHERE date = ? AND duration_minutes IS NOT NULL', (date,))
    avg_duration = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        'date': date,
        'total_visits': total_visits,
        'unique_visitors': unique_visitors,
        'unknown_visitors': unknown_visitors,
        'average_duration_minutes': round(avg_duration, 2)
    }


if __name__ == "__main__":
    init_database()
