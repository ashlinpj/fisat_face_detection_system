"""Visit repository - CRUD operations for visit logs"""

import logging
import sqlite3
from datetime import datetime
from typing import Optional, List

from app.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class VisitRepository:
    """Handles all database operations for visit logs.

    Dependencies are injected via the constructor (connection pool).
    """

    def __init__(self, pool):
        self.pool = pool

    def log_visit(self, student_db_id: int, student_id: str, student_name: str,
                  screenshot_path: str = None, is_known: bool = True) -> int:
        """Log a canteen visit. Returns log ID if successful, -1 on error."""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO visit_logs (student_db_id, student_id, student_name, is_known, screenshot_path)
                    VALUES (?, ?, ?, ?, ?)
                ''', (student_db_id, student_id, student_name, 1 if is_known else 0, screenshot_path))
                log_id = cursor.lastrowid
                conn.commit()
                return log_id

        except sqlite3.Error as e:
            logger.error("Error logging visit: %s", e)
            raise DatabaseError(f"Failed to log visit: {str(e)}")

    def update_visit_exit(self, log_id: int):
        """Update visit log with exit time"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE visit_logs 
                    SET exit_time = CURRENT_TIMESTAMP,
                        duration_minutes = CAST((JULIANDAY(CURRENT_TIMESTAMP) - JULIANDAY(entry_time)) * 24 * 60 AS INTEGER)
                    WHERE id = ?
                ''', (log_id,))
                conn.commit()
        except Exception as e:
            logger.error("Error updating visit exit: %s", e)

    def get_visit_logs(self, date: str = None, student_id: str = None) -> List[dict]:
        """Get visit logs with optional filters"""
        with self.pool.get_connection() as conn:
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
            return [dict(row) for row in rows]

    def get_recent_visit(self, student_id: str) -> Optional[dict]:
        """Get the most recent visit for a student"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM visit_logs 
                WHERE student_id = ? 
                ORDER BY entry_time DESC 
                LIMIT 1
            ''', (student_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_daily_statistics(self, date: str = None) -> dict:
        """Get daily visit statistics"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM visit_logs WHERE date = ?', (date,))
            total_visits = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT student_id) FROM visit_logs WHERE date = ? AND is_known = 1', (date,))
            unique_visitors = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM visit_logs WHERE date = ? AND is_known = 0', (date,))
            unknown_visitors = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(duration_minutes) FROM visit_logs WHERE date = ? AND duration_minutes IS NOT NULL', (date,))
            avg_duration = cursor.fetchone()[0] or 0

        return {
            'date': date,
            'total_visits': total_visits,
            'unique_visitors': unique_visitors,
            'unknown_visitors': unknown_visitors,
            'average_duration_minutes': round(avg_duration, 2)
        }
