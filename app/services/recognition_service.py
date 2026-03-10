"""Recognition service - Face embedding extraction, matching, and threaded recognition"""

import logging
import cv2
import numpy as np
from typing import Optional, List
import threading
import queue

import config
from app.utils.image_utils import enhance_face

logger = logging.getLogger(__name__)


class RecognitionService:
    """Handles face recognition: embedding extraction, similarity matching, threaded worker.

    Dependencies (known_faces list) are injected via the constructor.
    """

    def __init__(self, known_faces: List[dict], use_threaded: bool = True):
        self.known_faces = known_faces
        self._faces_lock = threading.Lock()
        self.recognition_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = True
        self.recognition_thread = None

        if use_threaded:
            self._start_recognition_thread()

    def _start_recognition_thread(self):
        """Start background thread for face recognition"""
        def recognition_worker():
            while self.running:
                try:
                    data = self.recognition_queue.get(timeout=0.1)
                    if data is None:
                        continue

                    face_enhanced = data['face']
                    bbox = data['bbox']
                    timestamp = data['timestamp']
                    face_crop = data['original_crop']

                    embedding = self.get_face_embedding(face_enhanced)
                    if embedding is None:
                        continue

                    from app.utils.face_matcher import find_best_match, cosine_similarity
                    with self._faces_lock:
                        known = self.known_faces
                    best_match = find_best_match(
                        embedding,
                        known,
                        threshold=getattr(config, 'FACE_RECOGNITION_THRESHOLD', 0.5),
                        margin=0.15
                    )

                    best_score = 0.0
                    if best_match is not None and best_match.get('face_embedding') is not None:
                        best_score = cosine_similarity(embedding, best_match['face_embedding'])

                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass

                    self.result_queue.put({
                        'student': best_match,
                        'confidence': best_score,
                        'bbox': bbox,
                        'timestamp': timestamp,
                        'face_crop': face_crop
                    })

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error("Recognition error: %s", e)

        self.recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
        self.recognition_thread.start()
        logger.info("Recognition thread started")

    def stop(self):
        """Stop the recognition thread"""
        self.running = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=1.0)

    def reload_known_faces(self, new_faces: List[dict]):
        """Update the known faces list (thread-safe via lock)"""
        with self._faces_lock:
            self.known_faces = new_faces

    def get_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding using DeepFace"""
        try:
            if face_image.shape[0] < 30 or face_image.shape[1] < 30:
                return None

            face_resized = cv2.resize(face_image, (160, 160), interpolation=cv2.INTER_LINEAR)

            from deepface import DeepFace
            embedding = DeepFace.represent(
                face_resized,
                model_name=config.FACE_EMBEDDING_MODEL,
                enforce_detection=False,
                detector_backend='skip'
            )

            if embedding:
                return np.array(embedding[0]['embedding'])
        except Exception:
            pass

        return None
