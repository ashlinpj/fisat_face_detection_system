"""Registration service - Student face registration workflow"""

import logging
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, List

import config
from app.utils.image_utils import enhance_face
from app.repositories.connection_pool import ensure_directories

logger = logging.getLogger(__name__)


class RegistrationService:
    """Handles student registration from images and camera frames.

    Dependencies are injected via the constructor.
    """

    def __init__(self, detection_service, recognition_service, student_repo):
        self.detection = detection_service
        self.recognition = recognition_service
        self.student_repo = student_repo

    def register_from_image(self, image_path: str, student_id: str,
                            name: str, department: str, year: int) -> bool:
        """Register student from high-resolution image file"""
        print(f"\nRegistering from image: {name} ({student_id})")

        image = cv2.imread(image_path)
        if image is None:
            logger.warning("Could not load image: %s", image_path)
            return False

        faces = self.detection.detect_faces(image)
        if not faces:
            try:
                from deepface import DeepFace
                detected = DeepFace.extract_faces(
                    image,
                    detector_backend='retinaface',
                    enforce_detection=True,
                    align=True
                )
                if detected and detected[0]['confidence'] > 0.9:
                    fa = detected[0]['facial_area']
                    faces = [(fa['x'], fa['y'], fa['x'] + fa['w'], fa['y'] + fa['h'])]
            except Exception:
                try:
                    from deepface import DeepFace
                    detected = DeepFace.extract_faces(
                        image,
                        detector_backend='opencv',
                        enforce_detection=False,
                        align=True
                    )
                    if detected and detected[0]['confidence'] > 0.5:
                        fa = detected[0]['facial_area']
                        faces = [(fa['x'], fa['y'], fa['x'] + fa['w'], fa['y'] + fa['h'])]
                except Exception:
                    pass

        if not faces:
            logger.warning("No face detected in image for %s", student_id)
            return False

        if len(faces) > 1:
            logger.info("%d faces found for %s, using largest", len(faces), student_id)
            faces = sorted(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]), reverse=True)

        x1, y1, x2, y2 = faces[0]
        h, w = image.shape[:2]
        pad = int((x2 - x1) * 0.2)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        face_crop = image[y1:y2, x1:x2]

        face_enhanced = enhance_face(face_crop)

        embedding = self.recognition.get_face_embedding(face_enhanced)
        if embedding is None:
            logger.warning("Could not extract features from image for %s", student_id)
            return False

        ensure_directories()
        face_filename = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_upload.jpg"
        face_path = os.path.join(config.FACES_DIR, face_filename)
        cv2.imwrite(face_path, face_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

        existing = self.student_repo.get_by_id(student_id)
        if existing:
            success = self.student_repo.update(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=embedding, face_image_path=face_path
            )
            logger.info("Updated: %s (from image upload)", name)
        else:
            success = self.student_repo.add(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=embedding, face_image_path=face_path
            )
            logger.info("Added: %s (from image upload)", name)

        return success

    def register_from_frames(self, frames: List[np.ndarray], student_id: str,
                             name: str, department: str, year: int) -> bool:
        """Register a student using multiple frames (averages embeddings)"""
        print(f"\nRegistering: {name} ({student_id}) with {len(frames)} sample(s)")

        if not frames:
            logger.warning("No frames provided for registration of %s", student_id)
            return False

        embeddings = []
        saved_paths = []

        ensure_directories()
        student_dir = os.path.join(config.FACES_DIR, student_id)
        os.makedirs(student_dir, exist_ok=True)

        for idx, frame in enumerate(frames, start=1):
            embedding, face_image = self._extract_face_sample(frame)
            if embedding is None or face_image is None:
                logger.debug("Sample %d: face/embedding not usable for %s", idx, student_id)
                continue

            embeddings.append(embedding)
            face_filename = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_s{idx}.jpg"
            face_path = os.path.join(student_dir, face_filename)
            cv2.imwrite(face_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(face_path)
            logger.debug("Captured sample %d -> %s", idx, face_filename)

        if not embeddings:
            logger.warning("No valid samples captured for %s. Registration aborted.", student_id)
            return False

        final_embedding = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(final_embedding)
        if norm > 0:
            final_embedding = final_embedding / norm

        primary_image_path = saved_paths[0] if saved_paths else None

        existing = self.student_repo.get_by_id(student_id)
        if existing:
            success = self.student_repo.update(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=final_embedding,
                face_image_path=primary_image_path
            )
            logger.info("Updated: %s with %d samples", name, len(embeddings))
        else:
            success = self.student_repo.add(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=final_embedding,
                face_image_path=primary_image_path
            )
            logger.info("Added: %s with %d samples", name, len(embeddings))

        return success

    def register_new_student(self, frame: np.ndarray, student_id: str,
                             name: str, department: str, year: int) -> bool:
        """Backward-compatible single-frame registration wrapper"""
        return self.register_from_frames([frame], student_id, name, department, year)

    def _extract_face_sample(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect, crop, enhance, and embed a single face from a frame"""
        if frame is None:
            return None, None

        faces = self.detection.detect_faces(frame)

        if not faces:
            try:
                from deepface import DeepFace
                detected = DeepFace.extract_faces(
                    frame, detector_backend='opencv',
                    enforce_detection=False
                )
                for d in detected:
                    if d.get('confidence', 0) > 0.5:
                        fa = d['facial_area']
                        faces.append((fa['x'], fa['y'], fa['x'] + fa['w'], fa['y'] + fa['h']))
            except Exception:
                pass

        if not faces:
            return None, None

        faces = sorted(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]), reverse=True)
        x1, y1, x2, y2 = faces[0]

        h, w = frame.shape[:2]
        pad = int((x2 - x1) * 0.35)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        face_image = frame[y1:y2, x1:x2]
        if face_image.size == 0:
            return None, None

        face_enhanced = enhance_face(face_image)
        embedding = self.recognition.get_face_embedding(face_enhanced)
        if embedding is None:
            return None, None

        return embedding, face_enhanced
