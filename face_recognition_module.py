"""
Face Recognition Module - LOGGING FIXED
Enforces strict cooldowns to prevent log spam.
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
        self.last_seen = {} # Memory of who we saw recently: {'ID': timestamp}
        
        # Threading Queues
        self.frame_count = 0
        self.recognition_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.recognition_thread = None
        self.running = True
        
        # GPU & Models
        self.use_gpu, self.gpu_name = check_gpu()
        self.yolo_model = None
        if self.use_gpu: self._load_yolo_gpu()
        
        # Fast detector fallback
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Start
        self._start_recognition_thread()
        database.init_database()
        self.reload_known_faces()
        
        print(f"Face Recognition System initialized ({self.gpu_name})")
        print(f"Log Cooldown set to: {config.LOG_COOLDOWN} seconds")
    
    def _load_yolo_gpu(self):
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n-face.pt')
            self.yolo_model.to('cuda')
        except:
            self.yolo_model = None
    
    def reload_known_faces(self):
        """Load known faces from the database into memory"""
        self.known_faces = database.get_all_students()
        print(f"Loaded {len(self.known_faces)} students from DB.")

    def _start_recognition_thread(self):
        def recognition_worker():
            while self.running:
                try:
                    data = self.recognition_queue.get(timeout=0.1)
                    if data is None: continue

                    face_img = data['face']
                    bbox = data['bbox']
                    timestamp = data['timestamp']
                    original_crop = data['original_crop']
                    
                    # 1. Get Embedding
                    embedding = self.get_face_embedding(face_img)
                    if embedding is None: continue
                    
                    # 2. Compare with DB
                    best_match = None
                    best_score = 1.0 
                    threshold = config.FACE_RECOGNITION_THRESHOLD

                    for student in self.known_faces:
                        db_emb = student['face_embedding']
                        if db_emb is None: continue

                        # Cosine Distance
                        a = np.matmul(np.transpose(db_emb), embedding)
                        b = np.sum(np.multiply(db_emb, db_emb))
                        c = np.sum(np.multiply(embedding, embedding))
                        distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
                        
                        if distance < best_score:
                            best_score = distance
                            best_match = student

                    # 3. Match Found?
                    final_match = None
                    confidence = 0
                    if best_score < threshold:
                        final_match = best_match
                        confidence = (1 - best_score) * 100
                    
                    # 4. Send Result back to Main Thread
                    self.result_queue.put({
                        'student': final_match,
                        'confidence': confidence,
                        'bbox': bbox,
                        'face_crop': original_crop
                    })
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"AI Thread Error: {e}")
        
        self.recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
        self.recognition_thread.start()

    def get_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            if face_image.size == 0: return None
            embedding = DeepFace.represent(
                img_path=face_image,
                model_name=config.FACE_EMBEDDING_MODEL,
                enforce_detection=False,
                detector_backend="skip"
            )
            return np.array(embedding[0]['embedding'])
        except:
            return None

    def detect_faces(self, frame):
        """Detect faces using YOLO (if available) or Haar Cascade"""
        if self.yolo_model:
            results = self.yolo_model(frame, verbose=False, conf=0.5)
            if results[0].boxes:
                boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))
                return boxes
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Main processing loop called by GUI"""
        self.frame_count += 1
        annotated_frame = frame.copy()
        processed_result = []
        
        # 1. Detect
        faces = self.detect_faces(frame)
        
        # 2. Recognize (Every 5th frame to save CPU/GPU)
        if self.frame_count % 5 == 0 and faces:
            # Pick the largest face to recognize (focus on the main person)
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

        # 3. Handle Recognition Results (LOGGING LOGIC HERE)
        try:
            result = self.result_queue.get_nowait()
            student = result['student']
            
            if student:
                name = student['name']
                sid = student['student_id']
                
                # Visuals (Green Box)
                x1, y1, x2, y2 = result['bbox']
                color = (0, 255, 0) # Green
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"{name} ({result['confidence']:.0f}%)", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                processed_result.append(result)

                # --- CRITICAL LOGGING LOGIC ---
                now = datetime.now()
                last_time = self.last_seen.get(sid)
                
                # Check Cooldown: Only log if we haven't seen them recently
                if last_time is None or (now - last_time).total_seconds() > config.LOG_COOLDOWN:
                    
                    print(f"✅ CONFIRMED VISIT: {name} (Confidence: {result['confidence']:.0f}%)")
                    
                    # 1. Save Screenshot
                    shot_name = f"{sid}_{int(now.timestamp())}.jpg"
                    shot_path = os.path.join(config.SCREENSHOTS_DIR, shot_name)
                    cv2.imwrite(shot_path, result['face_crop'])
                    
                    # 2. Write to DB
                    log_id = database.log_visit(
                        student_db_id=student['id'],
                        student_id=sid,
                        student_name=name,
                        screenshot_path=shot_path,
                        is_known=True
                    )
                    
                    if log_id > 0:
                        print("   -> Database Update: SUCCESS")
                        self.last_seen[sid] = now # Update cooldown timer
                    else:
                        print("   -> Database Update: FAILED")
                else:
                    # Debug: Show we see them but are waiting
                    remaining = config.LOG_COOLDOWN - (now - last_time).total_seconds()
                    # Uncomment below line if you want to see cooldown countdown in console
                    # if int(remaining) % 10 == 0: print(f"⏳ Cooldown for {name}: {int(remaining)}s")

        except queue.Empty:
            pass

        # Draw generic boxes for other faces (Red)
        for (x1, y1, x2, y2) in faces:
            # Simple check to avoid drawing red over green box
            # In a real system you'd use Intersection over Union (IoU)
            # here we just rely on the queue timing.
            if not processed_result: 
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        return annotated_frame, processed_result

    def register_new_student(self, frame, student_id, name, department, year):
        """Register a new student directly from a frame"""
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
        
        # 4. Initial Database Entry
        emb = self.get_face_embedding(face_img)
        
        return database.add_student(student_id, name, department, year, emb, path)

    def stop(self):
        """Stops the running threads safely."""
        self.running = False
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=1.0)