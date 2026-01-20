"""
Face Recognition Module - ULTRA OPTIMIZED VERSION
Threading + Caching + Image Upload Support
"""

import cv2
from PIL import Image, ImageEnhance
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
    gpu_available = False
    gpu_name = "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
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
        self.recognition_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.recognition_thread = None
        self.running = True
        
        # GPU status
        self.use_gpu, self.gpu_name = check_gpu()
        
        # Load YOLO with GPU if available
        self.yolo_model = None
        if self.use_gpu:
            self._load_yolo_gpu()
        
        # Fast detector fallback
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Start recognition thread
        self._start_recognition_thread()
        
        # Initialize DB
        database.init_database()
        self._load_known_faces()
        
        print(f"Face Recognition System initialized ({self.gpu_name})")
    
    def _load_yolo_gpu(self):
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n-face.pt') # Ensure you have the face version or standard
            self.yolo_model.to('cuda')
        except:
            self.yolo_model = None
    
    def _load_known_faces(self):
        """Load all known faces from database"""
        self.known_faces = database.get_all_students()
        print(f"Loaded {len(self.known_faces)} known faces")

    def reload_known_faces(self):
        self._load_known_faces()

    def _start_recognition_thread(self):
        """Start background thread for face recognition"""
        def recognition_worker():
            while self.running:
                try:
                    data = self.recognition_queue.get(timeout=0.1)
                    if data is None: continue
                    
                    face_img = data['face']
                    bbox = data['bbox']
                    timestamp = data['timestamp']
                    original_crop = data['original_crop']
                    
                    # 1. Get Embedding (Facenet512)
                    embedding = self.get_face_embedding(face_img)
                    if embedding is None: continue
                    
                    # 2. Find Best Match (Vector Math)
                    best_match = None
                    best_score = 1.0 # Cosine distance: 0 is same, 1 is opposite
                    
                    # Threshold: 0.35 is good for Facenet512
                    # The lower this number, the stricter the match
                    threshold = 0.35 

                    for student in self.known_faces:
                        db_emb = student['face_embedding']
                        if db_emb is None: continue

                        # Calculate Cosine Distance
                        a = np.matmul(np.transpose(db_emb), embedding)
                        b = np.sum(np.multiply(db_emb, db_emb))
                        c = np.sum(np.multiply(embedding, embedding))
                        distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
                        
                        if distance < best_score:
                            best_score = distance
                            best_match = student

                    # 3. Filter by Threshold
                    final_match = None
                    if best_score < threshold:
                        final_match = best_match
                        confidence = (1 - best_score) * 100 # Convert to %
                    else:
                        confidence = 0

                    # 4. Send Result
                    self.result_queue.put({
                        'student': final_match,
                        'confidence': confidence,
                        'bbox': bbox,
                        'timestamp': timestamp,
                        'face_crop': original_crop
                    })
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Recognition Error: {e}")
        
        self.recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
        self.recognition_thread.start()

    def get_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            if face_image.size == 0: return None
            # DeepFace Represent
            embedding = DeepFace.represent(
                img_path=face_image,
                model_name="Facenet512", # MUST MATCH TRAINER
                enforce_detection=False,
                detector_backend="skip"
            )
            return np.array(embedding[0]['embedding'])
        except:
            return None

    def detect_faces(self, frame):
        """Hybrid Detection: YOLO (if loaded) -> Haar Cascade (fallback)"""
        # 1. Try YOLO (GPU)
        if self.yolo_model:
            results = self.yolo_model(frame, verbose=False, conf=0.5)
            if results[0].boxes:
                boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))
                return boxes

        # 2. Fallback to Haar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        self.frame_count += 1
        annotated_frame = frame.copy()
        
        # 1. Detect
        faces = self.detect_faces(frame)
        
        # 2. Queue for Recognition (every 5th frame to save resources)
        if self.frame_count % 5 == 0 and faces:
            # Pick the largest face to recognize
            largest_face = max(faces, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            x1, y1, x2, y2 = largest_face
            
            face_crop = frame[y1:y2, x1:x2]
            
            try:
                self.recognition_queue.put_nowait({
                    'face': face_crop,
                    'bbox': largest_face,
                    'timestamp': datetime.now(),
                    'original_crop': face_crop
                })
            except queue.Full:
                pass

        # 3. Check Results
        try:
            result = self.result_queue.get_nowait()
            student = result['student']
            
            if student:
                # Log Visit
                name = student['name']
                sid = student['student_id']
                
                # Draw Green Box
                x1, y1, x2, y2 = result['bbox']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{name}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Check Cooldown & DB Log
                now = datetime.now()
                last_time = self.last_seen.get(sid)
                if not last_time or (now - last_time).total_seconds() > config.LOG_COOLDOWN:
                    
                    # Save Screenshot
                    shot_path = os.path.join(config.DB_FOLDER, "screenshots", f"{sid}_{int(datetime.now().timestamp())}.jpg")
                    os.makedirs(os.path.dirname(shot_path), exist_ok=True)
                    cv2.imwrite(shot_path, result['face_crop'])

                    database.log_visit(
                        student_db_id=student['id'],
                        student_id=sid,
                        student_name=name,
                        screenshot_path=shot_path,
                        is_known=True
                    )
                    self.last_seen[sid] = now
                    print(f"✅ Logged: {name}")

        except queue.Empty:
            pass

        # Draw generic boxes for all faces
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        return annotated_frame, []

    def register_new_student(self, frame, student_id, name, department, year):
        # 1. Detect
        faces = self.detect_faces(frame)
        if not faces: return False
        
        # 2. Crop Largest Face
        x1, y1, x2, y2 = max(faces, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        face_img = frame[y1:y2, x1:x2]
        
        # 3. Save Raw Image
        filename = f"{student_id}_{datetime.now().strftime('%Y%m%d')}.jpg"
        path = os.path.join(config.FACES_DIR, filename)
        cv2.imwrite(path, face_img)
        
        # 4. Initial Database Entry (Embedding will be updated by train_model.py)
        # We put a placeholder embedding or the single one for now
        emb = self.get_face_