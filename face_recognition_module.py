"""
Face Recognition Module - ULTRA OPTIMIZED VERSION
Threading + Caching + Image Upload Support + GPU Acceleration
"""

import os
import cv2
from skimage import exposure
from PIL import Image, ImageEnhance
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional
import threading
import queue
import config
import database

# Configure GPU for TensorFlow/Keras (used by DeepFace)
try:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import tensorflow as tf
    
    # Enable GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✓ TensorFlow GPU configured: {len(physical_devices)} device(s)")
except Exception as e:
    pass

from deepface import DeepFace

# Check for GPU availability
def check_gpu():
    """Check if CUDA GPU is available"""
    gpu_available = False
    gpu_name = "CPU"
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ PyTorch GPU Detected: {gpu_name}")
            print(f"  CUDA Version: {torch.version.cuda}")
    except:
        pass
    
    # Check TensorFlow GPU (used by DeepFace)
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
    
    # Check OpenCV CUDA
    try:
        cv2_cuda = cv2.cuda.getCudaEnabledDeviceCount()
        if cv2_cuda > 0:
            gpu_available = True
            if gpu_name == "CPU":
                gpu_name = "OpenCV CUDA"
            print(f"✓ OpenCV CUDA devices: {cv2_cuda}")
    except:
        pass
    
    if not gpu_available:
        print("⚠ No GPU detected - running on CPU mode")
        print("  To enable GPU: Install tensorflow-gpu or torch with CUDA support")
    else:
        print(f"✓ GPU mode enabled via: {gpu_name}")
    
    return gpu_available, gpu_name


class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.active_visits = {}
        self.last_seen = {}
        
        # Store current frame's recognized faces for display
        self.current_recognized_faces = {}  # bbox -> {name, confidence, timestamp}
        self.pending_labels = {}  # bbox -> {label, count, timestamp, confidence, is_known, student_id}
        
        # Performance optimization
        self.frame_count = 0
        self.cached_faces = []
        self.cached_results = []
        self.last_recognition_time = 0
        self.no_face_streak = 0
        
        # Threading for recognition (non-blocking)
        self.recognition_queue = queue.Queue(maxsize=int(getattr(config, 'RECOGNITION_QUEUE_SIZE', 6)))
        self.result_queue = queue.Queue(maxsize=int(getattr(config, 'RESULT_QUEUE_SIZE', 12)))
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
        self.dnn_on_cuda = False
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
                        margin=getattr(config, 'FACE_MATCH_MARGIN', 0.20)
                    )
                    best_score = 0.0
                    if best_match is not None and best_match.get('face_embedding') is not None:
                        best_score = cosine_similarity(embedding, best_match['face_embedding'])
                    # Send result back
                    try:
                        self.result_queue.put_nowait({
                            'student': best_match,
                            'confidence': best_score,
                            'bbox': bbox,
                            'timestamp': timestamp,
                            'face_crop': face_crop
                        })
                    except queue.Full:
                        # Keep latest results while avoiding stale queue buildup.
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.result_queue.put_nowait({
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

    def _match_bbox_key(self, bbox: Tuple[int, int, int, int], candidates: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Return nearest bbox key when current face is likely the same person."""
        if not candidates:
            return None

        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        face_size = max(1.0, ((x2 - x1) + (y2 - y1)) / 2)
        max_factor = float(getattr(config, 'BBOX_MATCH_DISTANCE_FACTOR', 0.30))
        max_distance = face_size * max_factor

        best_key = None
        min_distance = float('inf')
        for key in candidates:
            kx1, ky1, kx2, ky2 = key
            key_center_x, key_center_y = (kx1 + kx2) / 2, (ky1 + ky2) / 2
            distance = ((center_x - key_center_x) ** 2 + (center_y - key_center_y) ** 2) ** 0.5
            if distance <= max_distance and distance < min_distance:
                min_distance = distance
                best_key = key

        return best_key

    def _cleanup_label_caches(self, current_time: datetime):
        """Remove expired stable and pending labels."""
        label_ttl_sec = float(getattr(config, 'RECOGNITION_LABEL_TTL_SEC', 1.5))
        pending_ttl_sec = float(getattr(config, 'PENDING_LABEL_TTL_SEC', 1.5))

        expired_stable = []
        for bbox, info in self.current_recognized_faces.items():
            if (current_time - info['timestamp']).total_seconds() > label_ttl_sec:
                expired_stable.append(bbox)
        for bbox in expired_stable:
            del self.current_recognized_faces[bbox]

        expired_pending = []
        for bbox, info in self.pending_labels.items():
            if (current_time - info['timestamp']).total_seconds() > pending_ttl_sec:
                expired_pending.append(bbox)
        for bbox in expired_pending:
            del self.pending_labels[bbox]

    def _update_stable_label(self, bbox: Tuple[int, int, int, int], student: Optional[dict], confidence: float, current_time: datetime):
        """Use short frame confirmation before switching labels."""
        display_threshold = float(
            getattr(
                config,
                'DISPLAY_CONFIDENCE_THRESHOLD',
                getattr(config, 'FACE_RECOGNITION_THRESHOLD', 0.40)
            )
        )

        is_known = bool(student and confidence > display_threshold)
        next_label = student['name'] if is_known else 'Unknown'
        next_student_id = student['student_id'] if is_known else None

        known_confirm = int(getattr(config, 'KNOWN_CONFIRM_FRAMES', 2))
        unknown_confirm = int(getattr(config, 'UNKNOWN_CONFIRM_FRAMES', 3))
        required = known_confirm if is_known else unknown_confirm

        candidate_keys = list(self.pending_labels.keys()) + list(self.current_recognized_faces.keys())
        matched_key = self._match_bbox_key(bbox, candidate_keys)
        label_key = matched_key if matched_key is not None else bbox

        pending = self.pending_labels.get(label_key)
        if pending and pending['label'] == next_label:
            pending['count'] += 1
            pending['timestamp'] = current_time
            pending['confidence'] = confidence
            pending['is_known'] = is_known
            pending['student_id'] = next_student_id
        else:
            pending = {
                'label': next_label,
                'count': 1,
                'timestamp': current_time,
                'confidence': confidence,
                'is_known': is_known,
                'student_id': next_student_id,
            }
            self.pending_labels[label_key] = pending

        if pending['count'] >= required:
            self.current_recognized_faces[label_key] = {
                'name': pending['label'],
                'student_id': pending['student_id'],
                'confidence': pending['confidence'],
                'timestamp': current_time,
                'is_known': pending['is_known']
            }
    
    def stop(self):
        """Stop the recognition thread"""
        self.running = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=1.0)
    
    def _load_dnn_detector(self):
        """Load OpenCV DNN face detector (GPU accelerated)"""
        try:
            # Load pre-trained face detection model
            model_file = "res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "deploy.prototxt"
            
            # Try to load from OpenCV samples or local directory
            import urllib.request
            base_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/"
            
            # Download models if not present
            if not os.path.exists(model_file):
                print("Downloading DNN face detector model...")
                urllib.request.urlretrieve(base_url + model_file, model_file)
            
            if not os.path.exists(config_file):
                prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                urllib.request.urlretrieve(prototxt_url, config_file)
            
            self.dnn_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            build_info = ""
            try:
                build_info = cv2.getBuildInformation().lower()
            except Exception:
                build_info = ""

            supports_dnn_cuda = (
                hasattr(cv2.dnn, 'DNN_BACKEND_CUDA')
                and hasattr(cv2.dnn, 'DNN_TARGET_CUDA')
                and ('nvidia cuda' in build_info or 'cuda' in build_info)
                and ('cudnn' in build_info)
            )
            
            # Enable GPU if available
            if self.use_gpu and supports_dnn_cuda:
                try:
                    self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.dnn_on_cuda = True
                    print("✓ DNN face detector loaded on GPU")
                except:
                    self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.dnn_on_cuda = False
                    print("✓ DNN face detector loaded on CPU")
            else:
                self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.dnn_on_cuda = False
                if self.use_gpu and not supports_dnn_cuda:
                    print("✓ DNN face detector loaded on CPU (OpenCV DNN CUDA backend unavailable)")
                else:
                    print("✓ DNN face detector loaded on CPU")
        except Exception as e:
            print(f"  DNN detector load failed: {e}")
            self.dnn_net = None
            self.dnn_on_cuda = False
    
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

        # Store as single-item gallery for consistency with gallery matching
        multi_embeddings = [embedding]

        # Check if updating or adding
        existing = database.get_student_by_id(student_id)
        if existing:
            success = database.update_student(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=embedding, face_image_path=face_path,
                face_embeddings_multi=multi_embeddings
            )
            print(f"✓ Updated: {name} (from image upload)")
        else:
            success = database.add_student(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=embedding, face_image_path=face_path,
                face_embeddings_multi=multi_embeddings
            )
            print(f"✓ Added: {name} (from image upload)")
        
        if success:
            self.reload_known_faces()
        
        return success

    def register_from_images(self, image_paths: List[str], student_id: str,
                             name: str, department: str, year: int,
                             progress_callback=None) -> bool:
        """
        Register student from multiple high-resolution image files.
        Each image is processed for face detection, enhancement, and embedding.
        All embeddings are stored as a gallery for gallery-based matching.
        
        Args:
            image_paths: List of image file paths
            progress_callback: Optional callable(current, total, status_text)
        """
        print(f"\nRegistering from {len(image_paths)} images: {name} ({student_id})")
        
        embeddings = []
        saved_paths = []
        failed_images = []
        
        database.ensure_directories()
        student_dir = os.path.join(config.FACES_DIR, student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        for idx, image_path in enumerate(image_paths, start=1):
            if progress_callback:
                progress_callback(idx, len(image_paths), f"Processing image {idx}/{len(image_paths)}...")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"  ✗ Image {idx}: Could not load {os.path.basename(image_path)}")
                failed_images.append(os.path.basename(image_path))
                continue
            
            # Detect face
            faces = self.detect_faces(image)
            if not faces:
                try:
                    detected = DeepFace.extract_faces(
                        image,
                        detector_backend='opencv',
                        enforce_detection=False,
                        align=True
                    )
                    for d in detected:
                        if d.get('confidence', 0) > 0.5:
                            fa = d['facial_area']
                            faces.append((fa['x'], fa['y'], fa['x']+fa['w'], fa['y']+fa['h']))
                except Exception:
                    pass
            
            if not faces:
                print(f"  ✗ Image {idx}: No face detected in {os.path.basename(image_path)}")
                failed_images.append(os.path.basename(image_path))
                continue
            
            # Use largest face
            if len(faces) > 1:
                faces = sorted(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
            
            x1, y1, x2, y2 = faces[0]
            h, w = image.shape[:2]
            pad = int((x2 - x1) * 0.25)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                print(f"  ✗ Image {idx}: Empty face crop")
                failed_images.append(os.path.basename(image_path))
                continue
            
            # Enhance and get embedding
            face_enhanced = self.enhance_face(face_crop)
            embedding = self.get_face_embedding(face_enhanced)
            
            if embedding is None:
                # Try raw resize fallback
                raw_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LANCZOS4)
                embedding = self.get_face_embedding(raw_resized)
            
            if embedding is None:
                print(f"  ✗ Image {idx}: Embedding extraction failed for {os.path.basename(image_path)}")
                failed_images.append(os.path.basename(image_path))
                continue
            
            embeddings.append(embedding)
            
            # Save face image
            face_filename = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_img{idx}.jpg"
            face_path = os.path.join(student_dir, face_filename)
            cv2.imwrite(face_path, face_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(face_path)
            
            print(f"  ✓ Image {idx}: {os.path.basename(image_path)} → embedding OK")
        
        if progress_callback:
            progress_callback(len(image_paths), len(image_paths), "Finalizing registration...")
        
        if not embeddings:
            print(f"  ✗ No valid embeddings from any of {len(image_paths)} images!")
            return False
        
        print(f"\n  Summary: {len(embeddings)}/{len(image_paths)} images produced valid embeddings")
        if failed_images:
            print(f"  Failed: {', '.join(failed_images[:10])}" +
                  (f" ... and {len(failed_images)-10} more" if len(failed_images) > 10 else ""))
        
        # Store all individual embeddings as gallery
        multi_embeddings = [e.copy() for e in embeddings]
        
        # Average embedding for backward-compatible single-vector matching
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
                face_image_path=primary_image_path,
                face_embeddings_multi=multi_embeddings
            )
            print(f"  ✓ Updated: {name} with {len(embeddings)} image embeddings (gallery: {len(multi_embeddings)})")
        else:
            success = database.add_student(
                student_id=student_id, name=name, department=department,
                year=year, face_embedding=final_embedding,
                face_image_path=primary_image_path,
                face_embeddings_multi=multi_embeddings
            )
            print(f"  ✓ Added: {name} with {len(embeddings)} image embeddings (gallery: {len(multi_embeddings)})")
        
        if success:
            self.reload_known_faces()
        
        return success
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Face detection - returns SQUARE bounding boxes
        Uses DNN (GPU accelerated) or Haar Cascade fallback
        """
        h, w = frame.shape[:2]
        result = []
        
        # Try GPU-accelerated DNN detector first
        use_dnn = getattr(config, 'USE_DNN_DETECTOR', False)
        if use_dnn and not self.dnn_on_cuda and not getattr(config, 'ALLOW_CPU_DNN', False):
            use_dnn = False
        if use_dnn and self.dnn_net is not None:
            try:
                # Prepare blob for DNN
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.dnn_net.setInput(blob)
                detections = self.dnn_net.forward()
                
                # Process detections
                confidence_threshold = getattr(config, 'DNN_CONFIDENCE_THRESHOLD', 0.6)
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > confidence_threshold:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Make SQUARE bounding box
                        width = x2 - x1
                        height = y2 - y1
                        size = max(width, height)
                        center_x = x1 + width // 2
                        center_y = y1 + height // 2
                        
                        x1_sq = center_x - size // 2
                        y1_sq = center_y - size // 2
                        x2_sq = x1_sq + size
                        y2_sq = y1_sq + size
                        
                        # Ensure within frame bounds
                        x1_sq = max(0, x1_sq)
                        y1_sq = max(0, y1_sq)
                        x2_sq = min(w, x2_sq)
                        y2_sq = min(h, y2_sq)
                        
                        result.append((x1_sq, y1_sq, x2_sq, y2_sq))
                
                if result:  # If DNN found faces, return them
                    return result
            except Exception as e:
                # Fall through to Haar Cascade
                pass
        
        # Fallback: Fast Haar Cascade detection
        scale = getattr(config, 'DETECTION_SCALE', 0.5)
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
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
        Get face embedding using DeepFace - L2-normalized for reliable cosine similarity
        """
        try:
            if face_image.shape[0] < 30 or face_image.shape[1] < 30:
                return None
            
            # Resize for consistent results
            face_resized = cv2.resize(face_image, (160, 160), interpolation=cv2.INTER_LINEAR)
            
            # Get embedding - skip detection since we already have face
            embedding = DeepFace.represent(
                face_resized,
                model_name=config.FACE_EMBEDDING_MODEL,
                enforce_detection=False,
                detector_backend='skip'
            )
            
            if embedding:
                emb = np.array(embedding[0]['embedding'], dtype=np.float64)
                # L2-normalize so cosine similarity == dot product
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                return emb
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
        
        # Clean up old recognized faces quickly to avoid wrong-name carryover.
        self._cleanup_label_caches(current_time)
        
        # STEP 1: Detect faces every N frames and reuse cached boxes in-between.
        detect_interval = max(1, int(getattr(config, 'DETECT_EVERY_N_FRAMES', 2)))
        if self.frame_count % detect_interval == 0 or not self.cached_faces:
            detected_faces = self.detect_faces(frame)
            if detected_faces:
                self.cached_faces = detected_faces
                self.no_face_streak = 0
            else:
                self.no_face_streak += 1
                hold_frames = int(getattr(config, 'NO_FACE_HOLD_FRAMES', 4))
                if self.no_face_streak > hold_frames:
                    self.cached_faces = []
        faces = self.cached_faces
        
        # STEP 2: Queue enhanced crops for recognition
        process_interval = max(1, int(getattr(config, 'PROCESS_EVERY_N_FRAMES', 10)))
        if self.frame_count % process_interval == 0 and faces:
            for (x1, y1, x2, y2) in faces:
                # Crop face with padding
                pad = int((x2 - x1) * 0.3)
                x1_p = max(0, x1 - pad)
                y1_p = max(0, y1 - pad)
                x2_p = min(w, x2 + pad)
                y2_p = min(h, y2 + pad)
                face_crop = frame[y1_p:y2_p, x1_p:x2_p].copy()
                
                # Enhancement improves robustness but is expensive for low-FPS RTSP.
                if getattr(config, 'ENHANCE_BEFORE_RECOGNITION', False):
                    face_enhanced = self.enhance_face(face_crop)
                else:
                    face_enhanced = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
                
                # Queue for background recognition
                try:
                    self.recognition_queue.put_nowait({
                        'face': face_enhanced,
                        'bbox': (x1, y1, x2, y2),
                        'timestamp': current_time,
                        'original_crop': face_crop
                    })
                except queue.Full:
                    # Keep freshest samples by dropping oldest queued task.
                    try:
                        self.recognition_queue.get_nowait()
                    except queue.Empty:
                        pass
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
        while True:
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break

            student = result['student']
            confidence = result['confidence']
            timestamp = result['timestamp']
            face_crop = result['face_crop']
            bbox = result.get('bbox', None)

            if bbox:
                self._update_stable_label(bbox, student, confidence, current_time)

            # LOG only if confidence exceeds database threshold AND it's a known person
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
        
        recognized_people: List[dict] = []

        # STEP 4: Draw detection boxes with names
        for (x1, y1, x2, y2) in faces:
            # Find matching recognized face (with some tolerance for position changes)
            name_to_display = "Unknown"  # Default to Unknown instead of Detecting
            color = (0, 0, 255)  # Red for unknown
            confidence = 0.0
            is_known = False

            matched_key = self._match_bbox_key((x1, y1, x2, y2), list(self.current_recognized_faces.keys()))
            if matched_key is not None:
                best_match = self.current_recognized_faces.get(matched_key)
            else:
                best_match = None

            if best_match is not None:
                name_to_display = best_match['name']
                confidence = best_match['confidence']
                is_known = best_match.get('is_known', False)
                color = (0, 255, 0) if is_known else (0, 0, 255)  # Green for known, Red for unknown

            recognized_people.append({
                'name': name_to_display,
                'confidence': confidence,
                'is_known': is_known,
                'bbox': (x1, y1, x2, y2)
            })
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw name with background for better visibility
            label = name_to_display
            if confidence > 0:
                label += f" ({confidence:.2f})"
            
            # Calculate text size and draw background
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_label = max(y1, label_size[1] + 10)
            cv2.rectangle(annotated_frame, 
                         (x1, y1_label - label_size[1] - 10), 
                         (x1 + label_size[0], y1_label), 
                         color, -1)
            cv2.putText(annotated_frame, label, (x1, y1_label - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status overlay
        cv2.putText(annotated_frame, current_time.strftime("%H:%M:%S"),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame, recognized_people

    def save_visit_screenshot(self, face_crop: np.ndarray, student: dict) -> str:
        """Save enhanced face screenshot for visit log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visit_{student['student_id']}_{timestamp}.jpg"
        filepath = os.path.join(config.SCREENSHOTS_DIR, filename)
        database.ensure_directories()
        cv2.imwrite(filepath, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath

    def _extract_face_sample(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect, crop, enhance, and embed a single face from a frame."""
        if frame is None:
            print("      → frame is None")
            return None, None

        faces = self.detect_faces(frame)

        # Try DeepFace detection if cascade fails
        if not faces:
            try:
                detected = DeepFace.extract_faces(
                    frame,
                    detector_backend='opencv',
                    enforce_detection=False
                )
                for d in detected:
                    if d.get('confidence', 0) > 0.5:
                        fa = d['facial_area']
                        faces.append((fa['x'], fa['y'], fa['x']+fa['w'], fa['y']+fa['h']))
            except Exception:
                pass

        if not faces:
            print("      → no face detected in frame")
            return None, None

        # Use largest face
        faces = sorted(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
        x1, y1, x2, y2 = faces[0]
        face_w, face_h = x2 - x1, y2 - y1

        h, w = frame.shape[:2]
        base_size = max(1, face_w)

        # Try a few crop paddings and both enhanced/raw variants to increase usable samples.
        best_crop = None
        best_crop_size = 0
        for pad_factor in (0.40, 0.30, 0.20):
            pad = int(base_size * pad_factor)
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)

            face_image = frame[cy1:cy2, cx1:cx2]
            if face_image.size == 0:
                continue

            fh, fw = face_image.shape[:2]
            min_size = int(getattr(config, 'REGISTRATION_MIN_FACE_SIZE', 70))

            # Track best crop even if below min_size for last-resort fallback
            if min(fh, fw) > best_crop_size:
                best_crop = face_image.copy()
                best_crop_size = min(fh, fw)

            if min(fh, fw) < min_size:
                continue

            face_enhanced = self.enhance_face(face_image)
            embedding = self.get_face_embedding(face_enhanced)
            if embedding is not None:
                return embedding, face_enhanced

            # Fallback: try raw crop without heavy enhancement.
            raw_resized = cv2.resize(face_image, (160, 160), interpolation=cv2.INTER_LINEAR)
            embedding = self.get_face_embedding(raw_resized)
            if embedding is not None:
                return embedding, face_image

        # Last resort: if we have any crop at all, upscale it and try embedding
        if best_crop is not None and best_crop.size > 0:
            upscaled = cv2.resize(best_crop, (160, 160), interpolation=cv2.INTER_LANCZOS4)
            embedding = self.get_face_embedding(upscaled)
            if embedding is not None:
                print(f"      → used upscaled crop ({best_crop_size}px face)")
                return embedding, best_crop

        print(f"      → face detected ({face_w}x{face_h}px) but embedding extraction failed")
        return None, None

    def register_student_from_frames(self, frames: List[np.ndarray], student_id: str, name: str,
                                     department: str, year: int) -> bool:
        """Register a student using multiple frames (averages embeddings)."""
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

        min_valid = int(getattr(config, 'MIN_VALID_REGISTRATION_SAMPLES', 12))
        if len(embeddings) < min_valid:
            print(f"  ✗ Only {len(embeddings)} valid samples (minimum required: {min_valid}).")
            print("  ✗ Registration aborted. Please recapture with better lighting and steady face alignment.")
            return False

        # Keep individual embeddings for gallery-based matching
        multi_embeddings = [e.copy() for e in embeddings]

        # Average embeddings for backward-compatible single-vector matching
        final_embedding = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(final_embedding)
        if norm > 0:
            final_embedding = final_embedding / norm

        primary_image_path = saved_paths[0] if saved_paths else None

        existing = database.get_student_by_id(student_id)
        if existing:
            success = database.update_student(
                student_id=student_id,
                name=name,
                department=department,
                year=year,
                face_embedding=final_embedding,
                face_image_path=primary_image_path,
                face_embeddings_multi=multi_embeddings
            )
            print(f"  ✓ Updated: {name} with {len(embeddings)} samples (gallery: {len(multi_embeddings)} embeddings)")
        else:
            success = database.add_student(
                student_id=student_id,
                name=name,
                department=department,
                year=year,
                face_embedding=final_embedding,
                face_image_path=primary_image_path,
                face_embeddings_multi=multi_embeddings
            )
            print(f"  ✓ Added: {name} with {len(embeddings)} samples (gallery: {len(multi_embeddings)} embeddings)")

        if success:
            self.reload_known_faces()

        return success
    
    def register_new_student(self, frame: np.ndarray, student_id: str, name: str,
                            department: str, year: int) -> bool:
        """Backward-compatible single-frame registration wrapper."""
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