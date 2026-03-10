"""Unknown face repository - CRUD operations for unknown face records"""

import json
import logging
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


class UnknownFaceRepository:
    """Handles all database operations for unknown face records.

    Dependencies are injected via the constructor (connection pool).
    """

    def __init__(self, pool):
        self.pool = pool

    def add(self, face_image_path: str, face_embedding: np.ndarray) -> int:
        """Add unknown face for later registration"""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                embedding_json = json.dumps(face_embedding.tolist())
                cursor.execute('''
                    INSERT INTO unknown_faces (face_image_path, face_embedding)
                    VALUES (?, ?)
                ''', (face_image_path, embedding_json))
                face_id = cursor.lastrowid
                conn.commit()
                return face_id
        except Exception as e:
            logger.error("Error adding unknown face: %s", e)
            return -1

    def get_all(self) -> List[dict]:
        """Get all unknown faces"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM unknown_faces ORDER BY last_seen DESC')
            rows = cursor.fetchall()

            faces = []
            for row in rows:
                face = dict(row)
                if face['face_embedding']:
                    face['face_embedding'] = np.array(json.loads(face['face_embedding']))
                faces.append(face)
            return faces
