"""
Database module for Face Detection System
Handles all database operations including student records and visit logs
"""

import sqlite3
import os
import json
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple
import config

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

def init_database():
    """Initialize database with required tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Students table - stores student information and face embeddings
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
    
    # Visit logs table - tracks canteen visits
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
            source_type TEXT DEFAULT 'live',
            video_name TEXT,
            video_timestamp TEXT,
            FOREIGN KEY (student_db_id) REFERENCES students(id)
        )
    ''')
    
    # Add new columns to existing visit_logs table if they don't exist
    try:
        cursor.execute("ALTER TABLE visit_logs ADD COLUMN source_type TEXT DEFAULT 'live'")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE visit_logs ADD COLUMN video_name TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE visit_logs ADD COLUMN video_timestamp TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Unknown faces table - stores unrecognized faces for later registration
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
    """Add a new student to the database"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Convert embedding to JSON string for storage
        embedding_json = json.dumps(face_embedding.tolist())
        
        cursor.execute('''
            INSERT INTO students (student_id, name, department, year, face_embedding, face_image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (student_id, name, department, year, embedding_json, face_image_path))
        
        conn.commit()
        conn.close()
        print(f"Student {name} ({student_id}) added successfully!")
        return True
    except sqlite3.IntegrityError:
        print(f"Student ID {student_id} already exists!")
        return False
    except Exception as e:
        print(f"Error adding student: {e}")
        return False

def get_all_students() -> List[dict]:
    """Get all students from database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM students ORDER BY name')
    rows = cursor.fetchall()
    
    students = []
    for row in rows:
        student = dict(row)
        if student['face_embedding']:
            student['face_embedding'] = np.array(json.loads(student['face_embedding']))
        students.append(student)
    
    conn.close()
    return students

def get_student_by_id(student_id: str) -> Optional[dict]:
    """Get student by student ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        student = dict(row)
        if student['face_embedding']:
            student['face_embedding'] = np.array(json.loads(student['face_embedding']))
        return student
    return None

def update_student(student_id: str, name: str = None, department: str = None, 
                   year: int = None, face_embedding: np.ndarray = None, 
                   face_image_path: str = None) -> bool:
    """Update student information"""
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

def log_visit(student_db_id: int, student_id: str, student_name: str, screenshot_path: str = None, 
              is_known: bool = True, source_type: str = 'live', video_name: str = None, 
              video_timestamp: str = None) -> int:
    """Log a canteen visit with screenshot and source tracking"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO visit_logs (student_db_id, student_id, student_name, is_known, screenshot_path, 
                                   source_type, video_name, video_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (student_db_id, student_id, student_name, 1 if is_known else 0, screenshot_path,
              source_type, video_name, video_timestamp))
        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return log_id
    except Exception as e:
        print(f"Error logging visit: {e}")
        return -1

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
    
    # Total visits
    cursor.execute('SELECT COUNT(*) FROM visit_logs WHERE date = ?', (date,))
    total_visits = cursor.fetchone()[0]
    
    # Unique visitors
    cursor.execute('SELECT COUNT(DISTINCT student_id) FROM visit_logs WHERE date = ? AND is_known = 1', (date,))
    unique_visitors = cursor.fetchone()[0]
    
    # Unknown visitors
    cursor.execute('SELECT COUNT(*) FROM visit_logs WHERE date = ? AND is_known = 0', (date,))
    unknown_visitors = cursor.fetchone()[0]
    
    # Average duration
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

# Initialize database when module is imported
if __name__ == "__main__":
    init_database()
