"""
Face Recognition Module - ULTRA OPTIMIZED VERSION
Threading + Caching + Image Upload Support
"""

import cv2
from skimage import exposure
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
                    data = self.recognition_queue.get(timeout=0.1)
                    if data is None:
                        continue
                    
                    face_enhanced = data['face']
                    bbox = data['bbox']
                    timestamp = data['timestamp']
                    face_crop = data['original_crop']
                    
                    # Generate embedding from enhanced face
                    embedding = self.get_face_embedding(face_enhanced)
                    if embedding is None:
                        continue
                    
                    # Use robust matcher
                    from face_matcher import find_best_match, cosine_similarity
                    best_match = find_best_match(
                        embedding,
                        self.known_faces,
                        threshold=getattr(config, 'FACE_RECOGNITION_THRESHOLD', 0.5),
                        margin=0.15
                    )
                    best_score = 0.0
                    if best_match is not None and best_match.get('face_embedding') is not None:
                        best_score = cosine_similarity(embedding, best_match['face_embedding'])
                    # Send result back
                    try:
                        self.result_queue.get_nowait()  # Clear old
                    except:
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
                    print(f"Recognition error: {e}")
        
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
    
    def enhance_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Enhance face image quality for better recognition
        """
        if face_image is None or face_image.size == 0:
            return face_image
        
        # 1. Resize to optimal size (DeepFace works best at 160x160 or 224x224)
        target_size = 224
        face_resized = cv2.resize(face_image, (target_size, target_size), 
                                  interpolation=cv2.INTER_LANCZOS4)
        
        # 2. Convert to PIL for better enhancement
        face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
        
        # 3. Enhance sharpness
        enhancer = ImageEnhance.Sharpness(face_pil)
        face_pil = enhancer.enhance(1.5)
        
        # 4. Enhance contrast
        enhancer = ImageEnhance.Contrast(face_pil)
        face_pil = enhancer.enhance(1.2)
        
        # 5. Adjust brightness slightly
        enhancer = ImageEnhance.Brightness(face_pil)
        face_pil = enhancer.enhance(1.1)
        
        # Convert back to OpenCV format
        face_enhanced = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)
        
        # 6. Apply histogram equalization (better lighting normalization)
        lab = cv2.cvtColor(face_enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        face_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 7. Denoise
        face_enhanced = cv2.fastNlMeansDenoisingColored(face_enhanced, None, 10, 10, 7, 21)
        
        return face_enhanced
    
    def register_from_image(self, image_path: str, student_id: str,
                            name: str, department: str, year: int) -> bool:
        """
        Register student from high-resolution image file (admin upload)
        """
        print(f"\nRegistering from image: {name} ({student_id})")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Could not load image: {image_path}")
            return False

        faces = self.detect_faces(image)
        if not faces:
            # Try DeepFace detector (better for high-res)
            try:
                detected = DeepFace.extract_faces(
                    image,
                    detector_backend='retinaface',
                    enforce_detection=True,
                    align=True
                )
                if detected and detected[0]['confidence'] > 0.9:
                    fa = detected[0]['facial_area']
                    faces = [(fa['x'], fa['y'], fa['x']+fa['w'], fa['y']+fa['h'])]
            except Exception as e:
                print(f"  Trying opencv detector...")
                try:
                    detected = DeepFace.extract_faces(
                        image,
                        detector_backend='opencv',
                        enforce_detection=False,
                        align=True
                    )
                    if detected and detected[0]['confidence'] > 0.5:
                        fa = detected[0]['facial_area']
                        faces = [(fa['x'], fa['y'], fa['x']+fa['w'], fa['y']+fa['h'])]
                except:
                    pass

        if not faces:
            print("✗ No face detected in image!")
            return False

        # Use largest/best face
        if len(faces) > 1:
            print(f"  ⚠ {len(faces)} faces found, using largest")
            faces = sorted(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
        
        x1, y1, x2, y2 = faces[0]
        h, w = image.shape[:2]
        pad = int((x2 - x1) * 0.2)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        face_crop = image[y1:y2, x1:x2]

        # Enhance face quality
        face_enhanced = self.enhance_face(face_crop)
        
        # Get embedding
        embedding = self.get_face_embedding(face_enhanced)
        if embedding is None:
            print("✗ Could not extract features from image!")
            return False

        # Save face image
        database.ensure_directories()
        face_filename = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_upload.jpg"
        face_path = os.path.join(config.FACES_DIR, face_filename)
        cv2.imwrite(face_path, face_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Check if updating or adding
        existing = database.get_student_by_id(student_id)
        if existing:
            success = database.update_student(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=embedding, face_image_path=face_path
            )
            print(f"✓ Updated: {name} (from image upload)")
        else:
            success = database.add_student(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=embedding, face_image_path=face_path
            )
            print(f"✓ Added: {name} (from image upload)")
        
        if success:
            self.reload_known_faces()
        
        return success
    
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
            scaleFactor=1.15,
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
            
            # Make SQUARE bounding box
            size = max(orig_w, orig_h)
            center_x = x1 + orig_w // 2
            center_y = y1 + orig_h // 2
            
            x1_sq = center_x - size // 2
            y1_sq = center_y - size // 2
            x2_sq = x1_sq + size
            y2_sq = y1_sq + size
            
            # Ensure within frame bounds
            x1_sq = max(0, x1_sq)
            y1_sq = max(0, y1_sq)
            x2_sq = min(w, x2_sq)
            y2_sq = min(h, y2_sq)
            
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
                detector_backend='skip'
            )
            
            if embedding:
                return np.array(embedding[0]['embedding'])
        except Exception as e:
            pass
        
        return None
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings - improved for better accuracy
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
        euclidean_sim = 1 / (1 + euclidean_dist * 0.1)
        
        # Combined score
        combined = (cosine_sim * 0.7) + (euclidean_sim * 0.3)
        
        return max(0, combined)
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[dict], float]:
        """
        Recognize a face against known faces
        """
        face_embedding = self.get_face_embedding(face_image)
        
        if face_embedding is None:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for student in self.known_faces:
            if student['face_embedding'] is not None:
                similarity = self.compare_faces(face_embedding, student['face_embedding'])
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = student
        
        threshold = getattr(config, 'FACE_RECOGNITION_THRESHOLD', 0.35)
        if best_score >= threshold:
            return best_match, best_score
        
        return None, best_score
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Simplified: Detect → Crop → Enhance → Recognize (bg) → Log
        """
        self.frame_count += 1
        annotated_frame = frame.copy()
        current_time = datetime.now()
        h, w = frame.shape[:2]
        
        # STEP 1: Always detect faces (fast)
        faces = self.detect_faces(frame)
        
        # STEP 2: Queue enhanced crops for recognition
        process_interval = 10
        if self.frame_count % process_interval == 0 and faces:
            for (x1, y1, x2, y2) in faces:
                # Crop face with padding
                pad = int((x2 - x1) * 0.3)
                x1_p = max(0, x1 - pad)
                y1_p = max(0, y1 - pad)
                x2_p = min(w, x2 + pad)
                y2_p = min(h, y2 + pad)
                face_crop = frame[y1_p:y2_p, x1_p:x2_p].copy()
                
                # Enhance face quality
                face_enhanced = self.enhance_face(face_crop)
                
                # Queue for background recognition
                try:
                    self.recognition_queue.put_nowait({
                        'face': face_enhanced,
                        'bbox': (x1, y1, x2, y2),
                        'timestamp': current_time,
                        'original_crop': face_crop
                    })
                except queue.Full:
                    pass
        
        # STEP 3: Check for recognition results
        try:
            result = self.result_queue.get_nowait()
            student = result['student']
            confidence = result['confidence']
            timestamp = result['timestamp']
            face_crop = result['face_crop']
            
            if student and confidence > config.FACE_RECOGNITION_THRESHOLD:
                student_id = student['student_id']
                
                # COOLDOWN CHECK (5 minutes)
                should_log = True
                if student_id in self.last_seen:
                    time_diff = (timestamp - self.last_seen[student_id]).total_seconds()
                    if time_diff < 300:  # 5 minutes
                        should_log = False
                
                if should_log:
                    # LOG with screenshot
                    screenshot_path = self.save_visit_screenshot(face_crop, student)
                    database.log_visit(
                        student['id'], 
                        student_id, 
                        student['name'], 
                        screenshot_path=screenshot_path,
                        is_known=True
                    )
                    self.last_seen[student_id] = timestamp
                    print(f"✓ LOGGED: {student['name']} at {timestamp.strftime('%H:%M:%S')}")
                else:
                    print(f"  Cooldown: {student['name']} (seen recently)")
        
        except queue.Empty:
            pass
        
        # STEP 4: Draw simple detection boxes
        for (x1, y1, x2, y2) in faces:
            color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, "Detecting...", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Status overlay
        cv2.putText(annotated_frame, current_time.strftime("%H:%M:%S"),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame, []

    def save_visit_screenshot(self, face_crop: np.ndarray, student: dict) -> str:
        """Save enhanced face screenshot for visit log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visit_{student['student_id']}_{timestamp}.jpg"
        filepath = os.path.join(config.SCREENSHOTS_DIR, filename)
        database.ensure_directories()
        cv2.imwrite(filepath, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath
    
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
    
    system.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()