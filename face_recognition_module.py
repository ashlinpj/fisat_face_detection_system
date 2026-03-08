"""Face Recognition Module with GPU acceleration and threading support"""

import os
import cv2
from PIL import Image, ImageEnhance
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional
import threading
import queue
import config
import database

try:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✓ TensorFlow GPU configured: {len(physical_devices)} device(s)")
except Exception:
    pass

from deepface import DeepFace


def check_gpu():
    """Check if CUDA GPU is available"""
    gpu_available = False
    gpu_name = "CPU"
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ PyTorch GPU Detected: {gpu_name}")
            print(f"  CUDA Version: {torch.version.cuda}")
    except:
        pass
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_available = True
            print(f"✓ TensorFlow GPU: {len(gpus)} device(s) available")
            for gpu in gpus:
                print(f"  {gpu.name}")
    except:
        pass
    
    try:
        cv2_cuda = cv2.cuda.getCudaEnabledDeviceCount()
        if cv2_cuda > 0:
            print(f"✓ OpenCV CUDA devices: {cv2_cuda}")
    except:
        pass
    
    if not gpu_available:
        print("⚠ No GPU detected - running on CPU mode")
    
    return gpu_available, gpu_name


class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.active_visits = {}
        self.last_seen = {}
        self.current_recognized_faces = {}
        self.faces_lock = threading.Lock()
        self.frame_count = 0
        self.recognition_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.recognition_thread = None
        self.running = True
        self.use_gpu, self.gpu_name = check_gpu()
        self.yolo_model = None
        
        if self.use_gpu and getattr(config, 'USE_GPU', True):
            self._load_yolo_gpu()
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.dnn_net = None
        self._load_dnn_detector()
        
        if getattr(config, 'USE_THREADED_RECOGNITION', True):
            self._start_recognition_thread()
        
        self._load_known_faces()
        database.init_database()
        
        mode = "GPU+Threaded" if self.use_gpu else "CPU+Threaded"
        print(f"Face Recognition System initialized ({mode} Mode)")
    
    def _load_yolo_gpu(self):
        """Load YOLO model with GPU acceleration"""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
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
                    
                    embedding = self.get_face_embedding(face_enhanced)
                    if embedding is None:
                        continue
                    
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
                    
                    try:
                        self.result_queue.get_nowait()
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
        """Load OpenCV DNN face detector with GPU acceleration"""
        try:
            model_file = "res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "deploy.prototxt"
            
            if not os.path.exists(model_file):
                print("Downloading DNN face detector model...")
                import urllib.request
                base_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/"
                urllib.request.urlretrieve(base_url + model_file, model_file)
            
            if not os.path.exists(config_file):
                import urllib.request
                prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                urllib.request.urlretrieve(prototxt_url, config_file)
            
            self.dnn_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            
            if self.use_gpu:
                try:
                    self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    print("✓ DNN face detector loaded on GPU")
                except:
                    self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    print("✓ DNN face detector loaded on CPU")
            else:
                self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("✓ DNN face detector loaded on CPU")
        except Exception as e:
            print(f"  DNN detector load failed: {e}")
            self.dnn_net = None
    
    def _load_known_faces(self):
        """Load all known faces from database"""
        print("Loading known faces from database...")
        self.known_faces = database.get_all_students()
        print(f"Loaded {len(self.known_faces)} known faces")
    
    def reload_known_faces(self):
        """Reload known faces after adding new students"""
        self._load_known_faces()
    
    def enhance_face(self, face_image: np.ndarray) -> np.ndarray:
        """Enhance face image quality for better recognition"""
        if face_image is None or face_image.size == 0:
            return face_image
        
        target_size = getattr(config, 'FACE_ENHANCEMENT_TARGET_SIZE', 224)
        face_resized = cv2.resize(face_image, (target_size, target_size), 
                                  interpolation=cv2.INTER_LANCZOS4)
        
        face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
        
        enhancer = ImageEnhance.Sharpness(face_pil)
        face_pil = enhancer.enhance(getattr(config, 'FACE_SHARPNESS_FACTOR', 1.5))
        
        enhancer = ImageEnhance.Contrast(face_pil)
        face_pil = enhancer.enhance(getattr(config, 'FACE_CONTRAST_FACTOR', 1.2))
        
        enhancer = ImageEnhance.Brightness(face_pil)
        face_pil = enhancer.enhance(getattr(config, 'FACE_BRIGHTNESS_FACTOR', 1.1))
        
        face_enhanced = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)
        
        lab = cv2.cvtColor(face_enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        face_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        face_enhanced = cv2.fastNlMeansDenoisingColored(face_enhanced, None, 10, 10, 7, 21)
        
        return face_enhanced
    
    def register_from_image(self, image_path: str, student_id: str,
                            name: str, department: str, year: int) -> bool:
        """Register student from high-resolution image file"""
        print(f"\nRegistering from image: {name} ({student_id})")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Could not load image: {image_path}")
            return False

        faces = self.detect_faces(image)
        if not faces:
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
            except:
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

        if len(faces) > 1:
            print(f"  ⚠ {len(faces)} faces found, using largest")
            faces = sorted(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
        
        x1, y1, x2, y2 = faces[0]
        h, w = image.shape[:2]
        pad = int((x2 - x1) * 0.2)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        face_crop = image[y1:y2, x1:x2]

        face_enhanced = self.enhance_face(face_crop)
        
        embedding = self.get_face_embedding(face_enhanced)
        if embedding is None:
            print("✗ Could not extract features from image!")
            return False

        database.ensure_directories()
        face_filename = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_upload.jpg"
        face_path = os.path.join(config.FACES_DIR, face_filename)
        cv2.imwrite(face_path, face_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

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
        """Face detection - returns SQUARE bounding boxes"""
        h, w = frame.shape[:2]
        result = []
        min_face_size = getattr(config, 'MIN_FACE_SIZE', 60)  # Minimum face size in pixels
        
        use_dnn = getattr(config, 'USE_DNN_DETECTOR', False)
        if use_dnn and self.dnn_net is not None:
            try:
                # Prepare blob for DNN
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.dnn_net.setInput(blob)
                detections = self.dnn_net.forward()
                
                confidence_threshold = getattr(config, 'DNN_CONFIDENCE_THRESHOLD', 0.7)
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > confidence_threshold:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        
                        width, height = x2 - x1, y2 - y1
                        
                        # Filter out too small detections
                        if width < min_face_size or height < min_face_size:
                            continue
                        
                        # Filter out non-face aspect ratios
                        aspect_ratio = width / max(height, 1)
                        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                            continue
                        
                        size = max(width, height)
                        center_x, center_y = x1 + width // 2, y1 + height // 2
                        
                        x1_sq = max(0, center_x - size // 2)
                        y1_sq = max(0, center_y - size // 2)
                        x2_sq = min(w, x1_sq + size)
                        y2_sq = min(h, y1_sq + size)
                        
                        result.append((x1_sq, y1_sq, x2_sq, y2_sq))
                
                if result:
                    return result
            except:
                pass
        
        scale = getattr(config, 'DETECTION_SCALE', 0.5)
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Increase minNeighbors to reduce false positives
        min_neighbors = getattr(config, 'CASCADE_MIN_NEIGHBORS', 5)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=min_neighbors,
            minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, fw, fh) in faces:
            x1, y1 = int(x / scale), int(y / scale)
            orig_w, orig_h = int(fw / scale), int(fh / scale)
            
            # Filter out too small faces
            if orig_w < min_face_size or orig_h < min_face_size:
                continue
            
            # Filter out non-face aspect ratios
            aspect_ratio = orig_w / max(orig_h, 1)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            size = max(orig_w, orig_h)
            center_x, center_y = x1 + orig_w // 2, y1 + orig_h // 2
            
            x1_sq = max(0, center_x - size // 2)
            y1_sq = max(0, center_y - size // 2)
            x2_sq = min(w, x1_sq + size)
            y2_sq = min(h, y1_sq + size)
            
            final_size = min(x2_sq - x1_sq, y2_sq - y1_sq)
            
            # Final size check
            if final_size >= min_face_size:
                result.append((x1_sq, y1_sq, x1_sq + final_size, y1_sq + final_size))
        
        return result
    
    def get_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding using DeepFace"""
        try:
            if face_image.shape[0] < 30 or face_image.shape[1] < 30:
                return None
            
            face_resized = cv2.resize(face_image, (160, 160), interpolation=cv2.INTER_LINEAR)
            
            embedding = DeepFace.represent(
                face_resized,
                model_name=config.FACE_EMBEDDING_MODEL,
                enforce_detection=False,
                detector_backend='skip'
            )
            
            if embedding:
                return np.array(embedding[0]['embedding'])
        except:
            pass
        
        return None
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        norm1, norm2 = np.linalg.norm(embedding1), np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        emb1_norm = embedding1 / norm1
        emb2_norm = embedding2 / norm2
        
        # Cosine similarity
        cosine_sim = np.dot(emb1_norm, emb2_norm)
        
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        euclidean_sim = 1 / (1 + euclidean_dist * 0.1)
        
        combined = (cosine_sim * 0.7) + (euclidean_sim * 0.3)
        return max(0, combined)
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[dict], float]:
        """Recognize a face against known faces"""
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
        """Process frame: Detect → Crop → Enhance → Recognize → Log"""
        self.frame_count += 1
        annotated_frame = frame.copy()
        current_time = datetime.now()
        h, w = frame.shape[:2]
        
        with self.faces_lock:
            expired_bboxes = [
                bbox for bbox, info in self.current_recognized_faces.items()
                if (current_time - info['timestamp']).total_seconds() > getattr(config, 'CURRENT_FACES_TTL_SECONDS', 10)
            ]
            for bbox in expired_bboxes:
                del self.current_recognized_faces[bbox]
        
        faces = self.detect_faces(frame)
        
        process_interval = getattr(config, 'PROCESS_INTERVAL_FRAMES', 10)
        if self.frame_count % process_interval == 0 and faces:
            for (x1, y1, x2, y2) in faces:
                pad = int((x2 - x1) * 0.3)
                x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
                x2_p, y2_p = min(w, x2 + pad), min(h, y2 + pad)
                face_crop = frame[y1_p:y2_p, x1_p:x2_p].copy()
                face_enhanced = self.enhance_face(face_crop)
                
                try:
                    self.recognition_queue.put_nowait({
                        'face': face_enhanced,
                        'bbox': (x1, y1, x2, y2),
                        'timestamp': current_time,
                        'original_crop': face_crop
                    })
                except queue.Full:
                    pass
        
        try:
            result = self.result_queue.get_nowait()
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
                student_id = student['student_id']
                should_log = True
                
                if student_id in self.last_seen:
                    time_diff = (timestamp - self.last_seen[student_id]).total_seconds()
                    cooldown = getattr(config, 'VISIT_COOLDOWN_SECONDS', 300)
                    if time_diff < cooldown:
                        should_log = False
                
                if should_log:
                    screenshot_path = self.save_visit_screenshot(face_crop, student)
                    database.log_visit(
                        student['id'], student_id, student['name'], 
                        screenshot_path=screenshot_path, is_known=True
                    )
                    self.last_seen[student_id] = timestamp
                    print(f"✓ LOGGED: {student['name']} at {timestamp.strftime('%H:%M:%S')}")
                else:
                    print(f"  Cooldown: {student['name']} (seen recently)")
        except queue.Empty:
            pass
        
        recognized_people: List[dict] = []
        
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
        
        cv2.putText(annotated_frame, current_time.strftime("%H:%M:%S"),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame, recognized_people

    def save_visit_screenshot(self, face_crop: np.ndarray, student: dict) -> str:
        """Save face screenshot for visit log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visit_{student['student_id']}_{timestamp}.jpg"
        filepath = os.path.join(config.SCREENSHOTS_DIR, filename)
        database.ensure_directories()
        cv2.imwrite(filepath, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath

    def _extract_face_sample(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect, crop, enhance, and embed a single face from a frame"""
        if frame is None:
            return None, None

        faces = self.detect_faces(frame)

        if not faces:
            try:
                detected = DeepFace.extract_faces(
                    frame, detector_backend='opencv',
                    enforce_detection=False
                )
                for d in detected:
                    if d.get('confidence', 0) > 0.5:
                        fa = d['facial_area']
                        faces.append((fa['x'], fa['y'], fa['x']+fa['w'], fa['y']+fa['h']))
            except:
                pass

        if not faces:
            return None, None

        faces = sorted(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
        x1, y1, x2, y2 = faces[0]

        h, w = frame.shape[:2]
        pad = int((x2 - x1) * 0.35)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        face_image = frame[y1:y2, x1:x2]
        if face_image.size == 0:
            return None, None

        face_enhanced = self.enhance_face(face_image)
        embedding = self.get_face_embedding(face_enhanced)
        if embedding is None:
            return None, None

        return embedding, face_enhanced

    def register_student_from_frames(self, frames: List[np.ndarray], student_id: str, name: str,
                                     department: str, year: int) -> bool:
        """Register a student using multiple frames (averages embeddings)"""
        print(f"\nRegistering: {name} ({student_id}) with {len(frames)} sample(s)")

        if not frames:
            print("  ✗ No frames provided for registration")
            return False

        embeddings = []
        saved_paths = []

        database.ensure_directories()
        student_dir = os.path.join(config.FACES_DIR, student_id)
        os.makedirs(student_dir, exist_ok=True)

        for idx, frame in enumerate(frames, start=1):
            embedding, face_image = self._extract_face_sample(frame)
            if embedding is None or face_image is None:
                print(f"  ✗ Sample {idx}: face/embedding not usable")
                continue

            embeddings.append(embedding)
            face_filename = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_s{idx}.jpg"
            face_path = os.path.join(student_dir, face_filename)
            cv2.imwrite(face_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(face_path)
            print(f"  ✓ Captured sample {idx} -> {face_filename}")

        if not embeddings:
            print("  ✗ No valid samples captured. Registration aborted.")
            return False

        final_embedding = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(final_embedding)
        if norm > 0:
            final_embedding = final_embedding / norm

        primary_image_path = saved_paths[0] if saved_paths else None

        existing = database.get_student_by_id(student_id)
        if existing:
            success = database.update_student(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=final_embedding,
                face_image_path=primary_image_path
            )
            print(f"  ✓ Updated: {name} with {len(embeddings)} samples")
        else:
            success = database.add_student(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=final_embedding,
                face_image_path=primary_image_path
            )
            print(f"  ✓ Added: {name} with {len(embeddings)} samples")

        if success:
            self.reload_known_faces()

        return success
    
    def register_new_student(self, frame: np.ndarray, student_id: str, name: str,
                            department: str, year: int) -> bool:
        """Backward-compatible single-frame registration wrapper"""
        return self.register_student_from_frames([frame], student_id, name, department, year)


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