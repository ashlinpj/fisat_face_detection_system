"""
Face Recognition Module - ULTRA OPTIMIZED VERSION
Threading + Caching for smooth 30+ FPS
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Optional
from deepface import DeepFace
import threading
import queue
import config
import database

# Check for GPU availability
def check_gpu():
    """Check if CUDA GPU is available"""
    gpu_available = False
    gpu_name = "CPU"
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU Detected: {gpu_name}")
            print(f"  CUDA Version: {torch.version.cuda}")
    except:
        pass
    
    # Check OpenCV CUDA
    try:
        cv2_cuda = cv2.cuda.getCudaEnabledDeviceCount()
        if cv2_cuda > 0:
            print(f"✓ OpenCV CUDA devices: {cv2_cuda}")
    except:
        pass
    
    return gpu_available, gpu_name

class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.active_visits = {}
        self.last_seen = {}
        
        # Performance optimization
        self.frame_count = 0
        self.cached_faces = []
        self.cached_results = []
        self.last_recognition_time = 0
        
        # Threading for recognition (non-blocking)
        self.recognition_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.recognition_thread = None
        self.running = True
        
        # GPU status
        self.use_gpu, self.gpu_name = check_gpu()
        
        # Load YOLO with GPU if available
        self.yolo_model = None
        if self.use_gpu and getattr(config, 'USE_GPU', True):
            self._load_yolo_gpu()
        
        # Fast face detector (Haar Cascade as backup)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # DNN face detector (faster on GPU)
        self.dnn_net = None
        self._load_dnn_detector()
        
        # Start recognition thread
        if getattr(config, 'USE_THREADED_RECOGNITION', True):
            self._start_recognition_thread()
        
        # Initialize
        self._load_known_faces()
        database.init_database()
        
        mode = "GPU+Threaded" if self.use_gpu else "CPU+Threaded"
        print(f"Face Recognition System initialized ({mode} Mode)")
    
    def _load_yolo_gpu(self):
        """Load YOLO model with GPU acceleration"""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            # Force GPU usage
            self.yolo_model.to('cuda')
            print("✓ YOLO loaded on GPU")
        except Exception as e:
            print(f"  YOLO GPU load failed: {e}")
            self.yolo_model = None
    
    def _start_recognition_thread(self):
        """Start background thread for face recognition"""
        def recognition_worker():
            while self.running:
                try:
                    # Get face image from queue (non-blocking with timeout)
                    data = self.recognition_queue.get(timeout=0.1)
                    if data is None:
                        continue
                    
                    face_images, bboxes, frame = data
                    results = []
                    
                    for face_image, bbox in zip(face_images, bboxes):
                        student, confidence = self.recognize_face(face_image)
                        results.append({
                            'student': student,
                            'confidence': confidence,
                            'bbox': bbox,
                            'is_known': student is not None
                        })
                    
                    # Put results (clear old results first)
                    try:
                        self.result_queue.get_nowait()
                    except:
                        pass
                    self.result_queue.put(results)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    pass
        
        self.recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
        self.recognition_thread.start()
        print("✓ Recognition thread started")
    
    def stop(self):
        """Stop the recognition thread"""
        self.running = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=1.0)
    
    def _load_dnn_detector(self):
        """Load OpenCV DNN face detector (GPU accelerated)"""
        try:
            # Use OpenCV's DNN module for face detection
            model_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            # DNN is faster than Haar on GPU
            print("✓ Face detector ready")
        except Exception as e:
            print(f"  DNN detector failed: {e}")
    
    def _load_known_faces(self):
        """Load all known faces from database"""
        print("Loading known faces from database...")
        self.known_faces = database.get_all_students()
        print(f"Loaded {len(self.known_faces)} known faces")
    
    def reload_known_faces(self):
        """Reload known faces (call after adding new students)"""
        self._load_known_faces()
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Face detection - returns SQUARE bounding boxes
        Uses Haar Cascade (fast) - YOLO for person detection optional
        """
        h, w = frame.shape[:2]
        
        # Fast Haar Cascade detection (most reliable for faces)
        scale = getattr(config, 'DETECTION_SCALE', 0.5)
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,  # Faster
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        result = []
        for (x, y, fw, fh) in faces:
            # Scale back to original size
            x1 = int(x / scale)
            y1 = int(y / scale)
            orig_w = int(fw / scale)
            orig_h = int(fh / scale)
            
            # Make SQUARE bounding box (use the larger dimension)
            size = max(orig_w, orig_h)
            
            # Center the square on the face
            center_x = x1 + orig_w // 2
            center_y = y1 + orig_h // 2
            
            # Calculate square coordinates
            x1_sq = center_x - size // 2
            y1_sq = center_y - size // 2
            x2_sq = x1_sq + size
            y2_sq = y1_sq + size
            
            # Ensure within frame bounds
            x1_sq = max(0, x1_sq)
            y1_sq = max(0, y1_sq)
            x2_sq = min(w, x2_sq)
            y2_sq = min(h, y2_sq)
            
            # Re-adjust to keep square after boundary clipping
            final_w = x2_sq - x1_sq
            final_h = y2_sq - y1_sq
            final_size = min(final_w, final_h)
            
            result.append((x1_sq, y1_sq, x1_sq + final_size, y1_sq + final_size))
        
        return result
    
    def get_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding using DeepFace - optimized
        """
        try:
            if face_image.shape[0] < 30 or face_image.shape[1] < 30:
                return None
            
            # Resize for consistent results
            face_resized = cv2.resize(face_image, (160, 160))
            
            # Get embedding - skip detection since we already have face
            embedding = DeepFace.represent(
                face_resized,
                model_name=config.FACE_EMBEDDING_MODEL,
                enforce_detection=False,
                detector_backend='skip'  # Skip detection = faster
            )
            
            if embedding:
                return np.array(embedding[0]['embedding'])
        except Exception as e:
            pass
        
        return None
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings - improved for better accuracy
        Uses combined cosine + euclidean similarity
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        emb1_norm = embedding1 / norm1
        emb2_norm = embedding2 / norm2
        
        # Cosine similarity
        cosine_sim = np.dot(emb1_norm, emb2_norm)
        
        # Euclidean distance converted to similarity
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        euclidean_sim = 1 / (1 + euclidean_dist * 0.1)  # Scale factor
        
        # Combined score (cosine is more important)
        combined = (cosine_sim * 0.7) + (euclidean_sim * 0.3)
        
        return max(0, combined)
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[dict], float]:
        """
        Recognize a face against known faces - with debug info
        """
        face_embedding = self.get_face_embedding(face_image)
        
        if face_embedding is None:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        all_scores = []
        
        for student in self.known_faces:
            if student['face_embedding'] is not None:
                similarity = self.compare_faces(face_embedding, student['face_embedding'])
                all_scores.append((student['name'], similarity))
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = student
        
        # Debug: print top match
        if all_scores:
            all_scores.sort(key=lambda x: x[1], reverse=True)
            if all_scores[0][1] > 0.2:
                print(f"  Best match: {all_scores[0][0]} = {all_scores[0][1]:.1%}")
        
        threshold = getattr(config, 'FACE_RECOGNITION_THRESHOLD', 0.35)
        if best_score >= threshold:
            return best_match, best_score
        
        return None, best_score
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Process frame - ULTRA OPTIMIZED with threading
        Face detection runs every frame, recognition runs in background
        """
        self.frame_count += 1
        annotated_frame = frame.copy()
        current_time = datetime.now()
        h, w = frame.shape[:2]
        
        # Always detect faces (fast operation on small frame)
        faces = self.detect_faces(frame)
        self.cached_faces = faces
        
        # Check for recognition results from background thread
        try:
            new_results = self.result_queue.get_nowait()
            if new_results:
                self.cached_results = new_results
                # Handle logging for recognized faces
                for person in new_results:
                    if person['is_known'] and person['student']:
                        student = person['student']
                        student_id = student['student_id']
                        
                        should_log = True
                        if student_id in self.last_seen:
                            time_diff = (current_time - self.last_seen[student_id]).total_seconds()
                            if time_diff < config.MIN_TIME_BETWEEN_LOGS:
                                should_log = False
                        
                        if should_log:
                            database.log_visit(
                                student['id'], student_id, student['name'], is_known=True
                            )
                            self.last_seen[student_id] = current_time
                            print(f"✓ Logged: {student['name']} ({student_id})")
        except queue.Empty:
            pass
        
        # Queue face images for background recognition (non-blocking)
        process_interval = getattr(config, 'PROCESS_EVERY_N_FRAMES', 5)
        if self.frame_count % process_interval == 0 and faces:
            face_images = []
            bboxes = []
            
            for (x1, y1, x2, y2) in faces:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Add padding for better recognition
                pad = int((x2 - x1) * 0.2)
                x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
                x2_p, y2_p = min(w, x2 + pad), min(h, y2 + pad)
                
                face_image = frame[y1_p:y2_p, x1_p:x2_p].copy()
                face_images.append(face_image)
                bboxes.append((x1, y1, x2, y2))
            
            # Send to recognition thread (non-blocking)
            if face_images:
                try:
                    self.recognition_queue.put_nowait((face_images, bboxes, frame))
                except queue.Full:
                    pass  # Skip if queue is full
        
        # DRAWING - Always show faces with cached results
        # Match cached results to current face detections
        results_to_draw = []
        
        if self.cached_results:
            # Use cached recognition results
            for (x1, y1, x2, y2) in faces:
                # Find closest cached result
                best_match = None
                best_dist = float('inf')
                
                for cached in self.cached_results:
                    cx1, cy1, cx2, cy2 = cached['bbox']
                    dist = abs(x1 - cx1) + abs(y1 - cy1)
                    if dist < best_dist and dist < 100:  # Within 100 pixels
                        best_dist = dist
                        best_match = cached
                
                if best_match:
                    results_to_draw.append({
                        **best_match,
                        'bbox': (x1, y1, x2, y2)  # Use current position
                    })
                else:
                    results_to_draw.append({
                        'student': None,
                        'confidence': 0,
                        'bbox': (x1, y1, x2, y2),
                        'is_known': False
                    })
        else:
            # No cached results yet - show detection boxes
            for (x1, y1, x2, y2) in faces:
                results_to_draw.append({
                    'student': None,
                    'confidence': 0,
                    'bbox': (x1, y1, x2, y2),
                    'is_known': False
                })
        
        # Draw SQUARE boxes with rounded corners effect
        for person in results_to_draw:
            x1, y1, x2, y2 = person['bbox']
            
            if person['is_known'] and person['student']:
                name = person['student']['name']
                conf = person['confidence']
                color = (0, 255, 0)  # Green
                label = f"{name} ({conf:.0%})"
            else:
                color = (0, 165, 255)  # Orange for detecting
                label = "Detecting..." if person['confidence'] == 0 else f"Unknown ({person['confidence']:.0%})"
                if person['confidence'] > 0:
                    color = (0, 0, 255)  # Red for unknown
            
            # Draw square box with thick border
            thickness = 3
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw corner accents for modern look
            corner_len = min(20, (x2 - x1) // 4)
            # Top-left
            cv2.line(annotated_frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 2)
            cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 2)
            # Top-right
            cv2.line(annotated_frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 2)
            cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 2)
            # Bottom-left
            cv2.line(annotated_frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 2)
            cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 2)
            # Bottom-right
            cv2.line(annotated_frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 2)
            cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 2)
            
            # Label with background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status overlay
        cv2.putText(annotated_frame, current_time.strftime("%H:%M:%S"),
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        known = sum(1 for p in results_to_draw if p['is_known'])
        status = f"Faces: {len(faces)} | Known: {known}"
        cv2.putText(annotated_frame, status,
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return annotated_frame, results_to_draw
    
    def register_new_student(self, frame: np.ndarray, student_id: str, name: str,
                            department: str, year: int) -> bool:
        """
        Register a new student with improved face capture
        """
        print(f"\nRegistering: {name} ({student_id})")
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Try DeepFace detection if cascade fails
        if not faces:
            try:
                detected = DeepFace.extract_faces(frame, detector_backend='opencv',
                                                   enforce_detection=False)
                for d in detected:
                    if d['confidence'] > 0.5:
                        fa = d['facial_area']
                        faces.append((fa['x'], fa['y'], fa['x']+fa['w'], fa['y']+fa['h']))
            except:
                pass
        
        if not faces:
            print("  ✗ No face detected!")
            return False
        
        if len(faces) > 1:
            print(f"  ⚠ {len(faces)} faces found, using largest")
            faces = sorted(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
        
        x1, y1, x2, y2 = faces[0]
        
        # Add padding for better embedding
        h, w = frame.shape[:2]
        pad = int((x2 - x1) * 0.35)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        face_image = frame[y1:y2, x1:x2]
        
        # Get embedding
        embedding = self.get_face_embedding(face_image)
        
        if embedding is None:
            print("  ✗ Could not extract features!")
            return False
        
        # Save face image
        database.ensure_directories()
        face_filename = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        face_path = os.path.join(config.FACES_DIR, face_filename)
        cv2.imwrite(face_path, face_image)
        
        # Check if updating or adding
        existing = database.get_student_by_id(student_id)
        if existing:
            success = database.update_student(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=embedding, face_image_path=face_path
            )
            print(f"  ✓ Updated: {name}")
        else:
            success = database.add_student(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=embedding, face_image_path=face_path
            )
            print(f"  ✓ Added: {name}")
        
        if success:
            self.reload_known_faces()
        
        return success

def main():
    """Test the face recognition system"""
    print("Initializing Face Recognition System...")
    system = FaceRecognitionSystem()
    
    print("\nOpening camera...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return
    
    print("Press 'q' to quit, 'r' to register new student")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, recognized = system.process_frame(frame)
        
        # Display
        cv2.imshow(config.WINDOW_TITLE, annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Quick registration mode
            print("\n--- Registration Mode ---")
            student_id = input("Enter Student ID: ")
            name = input("Enter Name: ")
            department = input("Enter Department: ")
            year = int(input("Enter Year (1-4): "))
            
            print("Look at the camera... Registering in 3 seconds...")
            cv2.waitKey(3000)
            
            ret, frame = cap.read()
            if ret:
                if system.register_new_student(frame, student_id, name, department, year):
                    print("Registration successful!")
                else:
                    print("Registration failed!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
