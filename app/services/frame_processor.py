"""Frame processor - Orchestrates the detect -> recognize -> log -> annotate pipeline"""

import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional
import threading
import queue

import config
from app.utils.image_utils import enhance_face


class FrameProcessor:
    """Processes video frames through the face detection/recognition pipeline.

    Coordinates DetectionService, RecognitionService, and VisitService
    to detect faces, recognize them, log visits, and annotate frames.

    Dependencies are injected via the constructor.
    """

    def __init__(self, detection_service, recognition_service, visit_service):
        self.detection = detection_service
        self.recognition = recognition_service
        self.visits = visit_service

        self.frame_count = 0
        self.current_recognized_faces = {}
        self.faces_lock = threading.Lock()

        # Frame caching
        self.frame_cache = {}
        self.cache_ttl_seconds = getattr(config, 'FRAME_CACHE_TTL_SECONDS', 2)
        self.cache_max_size = getattr(config, 'FRAME_CACHE_MAX_SIZE', 10)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Process frame: Detect -> Crop -> Enhance -> Recognize -> Log -> Annotate"""
        self.frame_count += 1
        annotated_frame = frame.copy()
        current_time = datetime.now()
        h, w = frame.shape[:2]

        # Expire old recognized face overlays
        with self.faces_lock:
            expired_bboxes = [
                bbox for bbox, info in self.current_recognized_faces.items()
                if (current_time - info['timestamp']).total_seconds() > getattr(config, 'CURRENT_FACES_TTL_SECONDS', 10)
            ]
            for bbox in expired_bboxes:
                del self.current_recognized_faces[bbox]

        # Check frame cache before detecting faces
        frame_hash = self._compute_frame_hash(frame)
        cached_faces = self._get_cached_detections(frame_hash)

        if cached_faces is not None:
            faces = cached_faces
        else:
            faces = self.detection.detect_faces(frame)
            self._cache_detections(frame_hash, faces)

        # Submit face for recognition on nth frame
        process_interval = getattr(config, 'PROCESS_INTERVAL_FRAMES', 10)
        if self.frame_count % process_interval == 0 and faces:
            for (x1, y1, x2, y2) in faces:
                pad = int((x2 - x1) * 0.3)
                x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
                x2_p, y2_p = min(w, x2 + pad), min(h, y2 + pad)
                face_crop = frame[y1_p:y2_p, x1_p:x2_p].copy()
                face_enhanced = enhance_face(face_crop)

                try:
                    self.recognition.recognition_queue.put_nowait({
                        'face': face_enhanced,
                        'bbox': (x1, y1, x2, y2),
                        'timestamp': current_time,
                        'original_crop': face_crop
                    })
                except queue.Full:
                    pass

        # Collect recognition results
        try:
            result = self.recognition.result_queue.get_nowait()
            student = result['student']
            confidence = result['confidence']
            timestamp = result['timestamp']
            face_crop = result['face_crop']
            bbox = result.get('bbox', None)

            display_threshold = getattr(config, 'DISPLAY_CONFIDENCE_THRESHOLD', 0.50)
            if bbox:
                with self.faces_lock:
                    if student and confidence > display_threshold:
                        self.current_recognized_faces[bbox] = {
                            'name': student['name'],
                            'student_id': student['student_id'],
                            'confidence': confidence,
                            'timestamp': current_time,
                            'is_known': True
                        }
                    else:
                        self.current_recognized_faces[bbox] = {
                            'name': 'Unknown',
                            'student_id': None,
                            'confidence': confidence,
                            'timestamp': current_time,
                            'is_known': False
                        }

            if student and confidence > config.FACE_RECOGNITION_THRESHOLD:
                self.visits.log_visit(student, timestamp, face_crop)
        except queue.Empty:
            pass

        # Annotate frame with bounding boxes and labels
        recognized_people = self._annotate_frame(annotated_frame, faces, current_time)

        cv2.putText(annotated_frame, current_time.strftime("%H:%M:%S"),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated_frame, recognized_people

    def _annotate_frame(self, annotated_frame: np.ndarray,
                        faces: List[Tuple], current_time: datetime) -> List[dict]:
        """Draw bounding boxes and labels, return recognized_people list"""
        recognized_people = []

        for (x1, y1, x2, y2) in faces:
            name_to_display = "Unknown"
            color = (0, 0, 255)
            confidence = 0.0
            is_known = False

            best_match = None
            min_distance = float('inf')

            with self.faces_lock:
                for recognized_bbox, info in list(self.current_recognized_faces.items()):
                    rx1, ry1, rx2, ry2 = recognized_bbox
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    rcenter_x, rcenter_y = (rx1 + rx2) / 2, (ry1 + ry2) / 2
                    distance = ((center_x - rcenter_x) ** 2 + (center_y - rcenter_y) ** 2) ** 0.5

                    face_size = ((x2 - x1) + (y2 - y1)) / 2
                    distance_multiplier = getattr(config, 'FACE_CENTER_DISTANCE_MULTIPLIER', 0.5)
                    if distance < face_size * distance_multiplier and distance < min_distance:
                        min_distance = distance
                        best_match = info

            if best_match:
                name_to_display = best_match['name']
                confidence = best_match['confidence']
                is_known = best_match.get('is_known', False)
                color = (0, 255, 0) if is_known else (0, 0, 255)

            recognized_people.append({
                'name': name_to_display,
                'confidence': confidence,
                'is_known': is_known,
                'bbox': (x1, y1, x2, y2)
            })

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            label = name_to_display
            if confidence > 0:
                label += f" ({confidence:.2f})"

            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_label = max(y1, label_size[1] + 10)
            cv2.rectangle(annotated_frame,
                          (x1, y1_label - label_size[1] - 10),
                          (x1 + label_size[0], y1_label),
                          color, -1)
            cv2.putText(annotated_frame, label, (x1, y1_label - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return recognized_people

    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute a hash of the frame for caching (uses downsampled mean values)"""
        small_frame = cv2.resize(frame, (64, 64))
        hash_vals = small_frame.mean(axis=(0, 1))
        return f"{hash_vals[0]:.1f}_{hash_vals[1]:.1f}_{hash_vals[2]:.1f}"

    def _get_cached_detections(self, frame_hash: str) -> Optional[List]:
        """Get cached face detections if available and not expired"""
        if frame_hash in self.frame_cache:
            cache_entry = self.frame_cache[frame_hash]
            age = (datetime.now() - cache_entry['timestamp']).total_seconds()

            if age < self.cache_ttl_seconds:
                return cache_entry['faces']
            else:
                del self.frame_cache[frame_hash]

        return None

    def _cache_detections(self, frame_hash: str, faces: List):
        """Cache face detections for reuse"""
        if len(self.frame_cache) >= self.cache_max_size:
            oldest_hash = min(self.frame_cache.keys(),
                              key=lambda k: self.frame_cache[k]['timestamp'])
            del self.frame_cache[oldest_hash]

        self.frame_cache[frame_hash] = {
            'faces': faces,
            'timestamp': datetime.now()
        }
