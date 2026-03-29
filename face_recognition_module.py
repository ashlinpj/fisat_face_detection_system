"""
Face Recognition Module - ULTRA OPTIMIZED VERSION
Threading + Caching + Image Upload Support + GPU Acceleration
"""

import os
import cv2
from skimage import exposure
from PIL import Image, ImageEnhance
import numpy as np
import time
from datetime import datetime
from typing import List, Tuple, Optional
import threading
import queue
import config
import database
from visit_logger import VisitLogger

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
        self._deepface_weight_repair_attempted_models = set()
        self.visit_logger = VisitLogger(self.save_visit_screenshot, cooldown_seconds=300)
        # Backward compatibility for any existing accesses to last_seen.
        self.last_seen = self.visit_logger.last_seen
        
        # Store current frame's recognized faces for display
        self.current_recognized_faces = {}  # bbox -> {name, confidence, timestamp}
        self.pending_labels = {}  # bbox -> {label, count, timestamp, confidence, is_known, student_id}
        
        # Performance optimization
        self.frame_count = 0
        self.cached_faces = []
        self.cached_results = []
        self.last_recognition_time = 0
        self.no_face_streak = 0

        # Runtime embedding cache (image-hash) to avoid repeated DeepFace.represent calls.
        self.embedding_image_cache = {}
        self.embedding_cache_hits = 0
        self.embedding_cache_misses = 0

        # Stream-isolated runtime state (prevents label/cached-face bleed across cameras)
        self.stream_states = {}
        self.deferred_results_by_stream = {}
        default_state = self._get_stream_state('default')
        self.current_recognized_faces = default_state['current_recognized_faces']
        self.pending_labels = default_state['pending_labels']
        self.cached_faces = default_state['cached_faces']
        
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

    def _get_stream_state(self, stream_id: str = 'default') -> dict:
        """Get or create runtime state for a specific camera stream."""
        stream_key = str(stream_id or 'default')
        if stream_key not in self.stream_states:
            self.stream_states[stream_key] = {
                'frame_count': 0,
                'cached_faces': [],
                'no_face_streak': 0,
                'current_recognized_faces': {},
                'pending_labels': {},
                'embedding_track_cache': {},
            }
        return self.stream_states[stream_key]

    def _cleanup_embedding_track_cache(self, track_cache: dict, now_ts: float):
        """Drop stale/overflow track cache entries to keep memory bounded."""
        ttl_sec = float(getattr(config, 'EMBEDDING_TRACK_CACHE_TTL_SEC', 1.0))
        max_keys = int(getattr(config, 'EMBEDDING_TRACK_CACHE_MAX_KEYS', 200))

        stale = [k for k, v in track_cache.items() if now_ts - float(v.get('timestamp', 0.0)) > ttl_sec]
        for k in stale:
            track_cache.pop(k, None)

        if max_keys > 0 and len(track_cache) > max_keys:
            oldest = sorted(track_cache.items(), key=lambda kv: float(kv[1].get('timestamp', 0.0)))
            for k, _ in oldest[:len(track_cache) - max_keys]:
                track_cache.pop(k, None)

    def _make_embedding_hash_key(self, face_resized: np.ndarray) -> str:
        """Create a compact, illumination-tolerant hash key for embedding cache lookup."""
        hash_size = int(getattr(config, 'EMBEDDING_HASH_SIZE', 16))
        hash_size = max(8, min(32, hash_size))

        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        tiny = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        tiny = cv2.equalizeHist(tiny)
        quantized = (tiny // 8).astype(np.uint8)
        return f"{config.FACE_EMBEDDING_MODEL}:{face_resized.shape[0]}:{quantized.tobytes().hex()}"

    def _get_embedding_from_image_cache(self, cache_key: str) -> Optional[np.ndarray]:
        if not getattr(config, 'USE_EMBEDDING_CACHE', True):
            return None

        item = self.embedding_image_cache.get(cache_key)
        if not item:
            return None

        ttl_sec = float(getattr(config, 'EMBEDDING_IMAGE_CACHE_TTL_SEC', 20.0))
        now_ts = time.time()
        if now_ts - float(item.get('timestamp', 0.0)) > ttl_sec:
            self.embedding_image_cache.pop(cache_key, None)
            return None

        item['timestamp'] = now_ts
        emb = item.get('embedding')
        if emb is None:
            return None
        self.embedding_cache_hits += 1
        return emb.copy()

    def _put_embedding_in_image_cache(self, cache_key: str, embedding: np.ndarray):
        if not getattr(config, 'USE_EMBEDDING_CACHE', True):
            return

        now_ts = time.time()
        self.embedding_image_cache[cache_key] = {
            'embedding': embedding.copy(),
            'timestamp': now_ts,
        }

        max_keys = int(getattr(config, 'EMBEDDING_IMAGE_CACHE_MAX_KEYS', 1024))
        if max_keys > 0 and len(self.embedding_image_cache) > max_keys:
            oldest = sorted(self.embedding_image_cache.items(), key=lambda kv: float(kv[1].get('timestamp', 0.0)))
            for k, _ in oldest[:len(self.embedding_image_cache) - max_keys]:
                self.embedding_image_cache.pop(k, None)
    
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
                    stream_id = str(data.get('stream_id', 'default'))
                    precomputed_embedding = data.get('embedding')
                    track_cache_key = data.get('track_cache_key')
                    
                    # Generate embedding from enhanced face
                    embedding = precomputed_embedding
                    if embedding is None:
                        embedding = self.get_face_embedding(face_enhanced)
                    if embedding is None:
                        continue

                    # Update per-stream track cache with freshest embedding.
                    if getattr(config, 'USE_EMBEDDING_CACHE', True):
                        stream_state = self._get_stream_state(stream_id)
                        track_cache = stream_state.get('embedding_track_cache', {})
                        now_ts = time.time()
                        self._cleanup_embedding_track_cache(track_cache, now_ts)
                        cache_key = track_cache_key if track_cache_key is not None else bbox
                        track_cache[cache_key] = {'embedding': embedding.copy(), 'timestamp': now_ts}
                        track_cache[bbox] = {'embedding': embedding.copy(), 'timestamp': now_ts}

                    # Matching is isolated to keep recognition worker focused on queue processing.
                    best_match, best_score = self._find_best_match_and_score(embedding)
                    # Send result back
                    try:
                        self.result_queue.put_nowait({
                            'student': best_match,
                            'confidence': best_score,
                            'bbox': bbox,
                            'timestamp': timestamp,
                            'face_crop': face_crop,
                            'stream_id': stream_id,
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
                            'face_crop': face_crop,
                            'stream_id': stream_id,
                        })
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Recognition error: {e}")
        
        self.recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
        self.recognition_thread.start()
        print("✓ Recognition thread started")

    def _find_best_match_and_score(self, embedding: np.ndarray) -> Tuple[Optional[dict], float]:
        """Compute accepted identity and fallback confidence score for label smoothing."""
        from face_matcher import find_best_match, cosine_similarity

        best_match = find_best_match(
            embedding,
            self.known_faces,
            threshold=getattr(config, 'FACE_RECOGNITION_THRESHOLD', 0.5),
            margin=getattr(config, 'FACE_MATCH_MARGIN', 0.20)
        )

        top_k = int(getattr(config, 'FACE_GALLERY_TOP_K', 5))
        best_score = 0.0
        for candidate in self.known_faces:
            gallery = candidate.get('face_embeddings_multi')
            avg_emb = candidate.get('face_embedding')

            if gallery and len(gallery) > 0:
                sims = [cosine_similarity(embedding, g) for g in gallery if g is not None]
                if sims:
                    sims.sort(reverse=True)
                    k = min(top_k, len(sims))
                    score = float(np.mean(sims[:k]))
                else:
                    score = 0.0
            elif avg_emb is not None:
                score = cosine_similarity(embedding, avg_emb)
            else:
                continue

            if score > best_score:
                best_score = score

        if best_match is not None and best_match.get('face_embedding') is not None:
            best_score = max(best_score, cosine_similarity(embedding, best_match['face_embedding']))

        return best_match, best_score

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

    def _cleanup_label_caches(self, current_time: datetime, current_recognized_faces=None, pending_labels=None):
        """Remove expired stable and pending labels."""
        if current_recognized_faces is None:
            current_recognized_faces = self.current_recognized_faces
        if pending_labels is None:
            pending_labels = self.pending_labels

        label_ttl_sec = float(getattr(config, 'RECOGNITION_LABEL_TTL_SEC', 1.5))
        pending_ttl_sec = float(getattr(config, 'PENDING_LABEL_TTL_SEC', 1.5))

        expired_stable = []
        for bbox, info in current_recognized_faces.items():
            if (current_time - info['timestamp']).total_seconds() > label_ttl_sec:
                expired_stable.append(bbox)
        for bbox in expired_stable:
            del current_recognized_faces[bbox]

        expired_pending = []
        for bbox, info in pending_labels.items():
            if (current_time - info['timestamp']).total_seconds() > pending_ttl_sec:
                expired_pending.append(bbox)
        for bbox in expired_pending:
            del pending_labels[bbox]

    def _update_stable_label(self, bbox: Tuple[int, int, int, int], student: Optional[dict], confidence: float, current_time: datetime,
                             current_recognized_faces=None, pending_labels=None):
        """Use short frame confirmation before switching labels."""
        if current_recognized_faces is None:
            current_recognized_faces = self.current_recognized_faces
        if pending_labels is None:
            pending_labels = self.pending_labels

        display_threshold = float(
            getattr(
                config,
                'DISPLAY_CONFIDENCE_THRESHOLD',
                getattr(config, 'FACE_RECOGNITION_THRESHOLD', 0.40)
            )
        )

        is_known = bool(student and confidence > display_threshold)
        min_unknown_similarity = float(getattr(config, 'UNKNOWN_MIN_SIMILARITY', 0.25))

        # If there are enrolled identities, suppress very weak unknown signals as noise.
        # When gallery is empty, allow unknown labels so users can still see detections.
        if self.known_faces and (not is_known) and confidence < min_unknown_similarity:
            matched_key = self._match_bbox_key(
                bbox,
                list(pending_labels.keys()) + list(current_recognized_faces.keys())
            )
            if matched_key is not None:
                pending_labels.pop(matched_key, None)
                current_recognized_faces.pop(matched_key, None)
            return

        next_label = student['name'] if is_known else 'Unknown'
        next_student_id = student['student_id'] if is_known else None

        known_confirm = int(getattr(config, 'KNOWN_CONFIRM_FRAMES', 2))
        unknown_confirm = int(getattr(config, 'UNKNOWN_CONFIRM_FRAMES', 3))
        required = known_confirm if is_known else unknown_confirm

        candidate_keys = list(pending_labels.keys()) + list(current_recognized_faces.keys())
        matched_key = self._match_bbox_key(bbox, candidate_keys)
        label_key = matched_key if matched_key is not None else bbox

        pending = pending_labels.get(label_key)
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
            pending_labels[label_key] = pending

        if pending['count'] >= required:
            current_recognized_faces[label_key] = {
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

                        if self._is_valid_face_box(x1_sq, y1_sq, x2_sq, y2_sq, w, h):
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
            minNeighbors=int(getattr(config, 'HAAR_MIN_NEIGHBORS', 6)),
            minSize=(int(getattr(config, 'HAAR_MIN_SIZE', 40)), int(getattr(config, 'HAAR_MIN_SIZE', 40))),
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

            nx2 = x1_sq + final_size
            ny2 = y1_sq + final_size
            if self._is_valid_face_box(x1_sq, y1_sq, nx2, ny2, w, h):
                result.append((x1_sq, y1_sq, nx2, ny2))
        
        return result

    def _is_valid_face_box(self, x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int) -> bool:
        """Reject implausible face boxes that are common on noisy/empty RTSP frames."""
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw <= 0 or bh <= 0:
            return False

        size = min(bw, bh)
        min_face_px = int(getattr(config, 'MIN_FACE_SIZE_PX', 80))
        min_face_ratio = float(getattr(config, 'MIN_FACE_SIZE_RATIO', 0.05))
        max_face_ratio = float(getattr(config, 'MAX_FACE_SIZE_RATIO', 0.70))
        min_required = max(min_face_px, int(min(frame_w, frame_h) * min_face_ratio))
        max_allowed = int(min(frame_w, frame_h) * max_face_ratio)

        if size < min_required:
            return False
        if max_allowed > 0 and size > max_allowed:
            return False
        return True
    
    def get_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding using DeepFace - L2-normalized for reliable cosine similarity
        """
        try:
            if face_image.shape[0] < 30 or face_image.shape[1] < 30:
                return None
            
            # Resize for consistent results
            target_size = int(getattr(config, 'RECOGNITION_INPUT_SIZE', 160))
            target_size = max(96, min(224, target_size))
            face_resized = cv2.resize(face_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

            cache_key = self._make_embedding_hash_key(face_resized)
            cached_embedding = self._get_embedding_from_image_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding
            self.embedding_cache_misses += 1
            
            # Get embedding - skip detection since we already have face
            embedding = DeepFace.represent(
                img_path=face_resized,
                model_name=config.FACE_EMBEDDING_MODEL,
                enforce_detection=False,
                detector_backend='skip'
            )

            # DeepFace may return List[Dict] or List[List[Dict]] depending input shape.
            if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                embedding = embedding[0]

            if embedding and isinstance(embedding, list):
                emb = np.array(embedding[0]['embedding'], dtype=np.float64)
                # L2-normalize so cosine similarity == dot product
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                self._put_embedding_in_image_cache(cache_key, emb)
                return emb
        except Exception as e:
            if self._repair_corrupted_deepface_weights(config.FACE_EMBEDDING_MODEL, e):
                try:
                    embedding = DeepFace.represent(
                        img_path=face_image,
                        model_name=config.FACE_EMBEDDING_MODEL,
                        enforce_detection=False,
                        detector_backend='skip'
                    )

                    if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                        embedding = embedding[0]

                    if embedding and isinstance(embedding, list):
                        emb = np.array(embedding[0]['embedding'], dtype=np.float64)
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                        self._put_embedding_in_image_cache(cache_key, emb)
                        return emb
                except Exception as retry_error:
                    print(f"      → embedding retry failed: {retry_error}")
            elif getattr(config, 'VERBOSE_RTSP_LOGS', False):
                print(f"      → embedding exception: {e}")
        
        return None

    def _repair_corrupted_deepface_weights(self, model_name: str, error: Exception) -> bool:
        """Delete corrupted DeepFace weight file once, then allow re-download on retry."""
        error_text = str(error)
        if not error_text:
            return False

        corruption_markers = (
            "interruption during the download",
            "file signature not found",
            "Unable to synchronously open file",
        )
        if not any(marker in error_text for marker in corruption_markers):
            return False

        model_key = (model_name or "").strip().lower().replace("-", "")
        weight_file_map = {
            "facenet512": "facenet512_weights.h5",
            "facenet": "facenet_weights.h5",
            "vggface": "vgg_face_weights.h5",
            "arcface": "arcface_weights.h5",
            "sface": "face_recognition_sface_2021dec.onnx",
        }
        weight_filename = weight_file_map.get(model_key)
        if not weight_filename:
            return False

        if model_key in self._deepface_weight_repair_attempted_models:
            return False
        self._deepface_weight_repair_attempted_models.add(model_key)

        weights_dir = os.path.join(os.path.expanduser("~"), ".deepface", "weights")
        weight_path = os.path.join(weights_dir, weight_filename)

        if os.path.exists(weight_path):
            try:
                os.remove(weight_path)
                print(f"      → removed corrupted DeepFace weights: {weight_filename}")
                print("      → retrying embedding extraction (weights will re-download)")
                return True
            except Exception as remove_error:
                print(f"      → failed to remove corrupted weights: {remove_error}")
                return False

        return False
    
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
    
    def process_frame(self, frame: np.ndarray, stream_id: str = 'default') -> Tuple[np.ndarray, List[dict]]:
        """
        Simplified: Detect → Crop → Enhance → Recognize (bg) → Log
        """
        stream_key = str(stream_id or 'default')
        stream_state = self._get_stream_state(stream_key)
        stream_state['frame_count'] += 1
        self.frame_count = stream_state['frame_count']

        current_recognized_faces = stream_state['current_recognized_faces']
        pending_labels = stream_state['pending_labels']
        cached_faces = stream_state['cached_faces']
        no_face_streak = stream_state['no_face_streak']

        annotated_frame = frame.copy()
        current_time = datetime.now()
        h, w = frame.shape[:2]

        # Optional downscale for detection/recognition prep to reduce CPU load on high-res RTSP.
        processing_frame = frame
        scale_x = 1.0
        scale_y = 1.0
        if getattr(config, 'ENABLE_PROCESSING_RESIZE', False):
            max_w = int(getattr(config, 'PROCESSING_MAX_WIDTH', w))
            max_h = int(getattr(config, 'PROCESSING_MAX_HEIGHT', h))
            if max_w > 0 and max_h > 0:
                resize_scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
                if resize_scale < 1.0:
                    pw = max(1, int(w * resize_scale))
                    ph = max(1, int(h * resize_scale))
                    processing_frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
                    scale_x = w / pw
                    scale_y = h / ph
        
        # Clean up old recognized faces quickly to avoid wrong-name carryover.
        self._cleanup_label_caches(current_time, current_recognized_faces=current_recognized_faces, pending_labels=pending_labels)
        
        # STEP 1: Detect faces every N frames and reuse cached boxes in-between.
        detect_interval = max(1, int(getattr(config, 'DETECT_EVERY_N_FRAMES', 2)))
        if self.frame_count % detect_interval == 0 or not cached_faces:
            detected_faces = self.detect_faces(processing_frame)

            if scale_x != 1.0 or scale_y != 1.0:
                # Map detection boxes from processing resolution back to original frame.
                mapped_faces = []
                for (x1, y1, x2, y2) in detected_faces:
                    ox1 = max(0, min(w - 1, int(x1 * scale_x)))
                    oy1 = max(0, min(h - 1, int(y1 * scale_y)))
                    ox2 = max(0, min(w, int(x2 * scale_x)))
                    oy2 = max(0, min(h, int(y2 * scale_y)))
                    if ox2 > ox1 and oy2 > oy1:
                        mapped_faces.append((ox1, oy1, ox2, oy2))
                detected_faces = mapped_faces

            if detected_faces:
                cached_faces = detected_faces
                no_face_streak = 0
            else:
                no_face_streak += 1
                hold_frames = int(getattr(config, 'NO_FACE_HOLD_FRAMES', 4))
                if no_face_streak > hold_frames:
                    cached_faces = []

        stream_state['cached_faces'] = cached_faces
        stream_state['no_face_streak'] = no_face_streak
        faces = cached_faces

        track_cache = stream_state.get('embedding_track_cache', {})
        if getattr(config, 'USE_EMBEDDING_CACHE', True):
            self._cleanup_embedding_track_cache(track_cache, time.time())
        
        # STEP 2: Queue enhanced crops for recognition
        process_interval = max(1, int(getattr(config, 'PROCESS_EVERY_N_FRAMES', 10)))
        if self.frame_count % process_interval == 0 and faces:
            faces_for_recognition = faces
            max_faces_for_recognition = int(getattr(config, 'MAX_FACES_FOR_RECOGNITION', 0))
            if max_faces_for_recognition > 0 and len(faces_for_recognition) > max_faces_for_recognition:
                # Prioritize larger faces for better quality/latency tradeoff.
                faces_for_recognition = sorted(
                    faces_for_recognition,
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                    reverse=True,
                )[:max_faces_for_recognition]

            for (x1, y1, x2, y2) in faces_for_recognition:
                # Crop face with padding
                pad = int((x2 - x1) * 0.3)
                x1_p = max(0, x1 - pad)
                y1_p = max(0, y1 - pad)
                x2_p = min(w, x2 + pad)
                y2_p = min(h, y2 + pad)
                face_crop = frame[y1_p:y2_p, x1_p:x2_p].copy()

                matched_track_key = None
                cached_track_embedding = None
                if getattr(config, 'USE_EMBEDDING_CACHE', True) and track_cache:
                    matched_track_key = self._match_bbox_key((x1, y1, x2, y2), list(track_cache.keys()))
                    if matched_track_key is not None:
                        entry = track_cache.get(matched_track_key)
                        if entry is not None:
                            cached_track_embedding = entry.get('embedding')

                face_enhanced = None
                if cached_track_embedding is None:
                    # Enhancement improves robustness but is expensive for low-FPS RTSP.
                    if getattr(config, 'ENHANCE_BEFORE_RECOGNITION', False):
                        face_enhanced = self.enhance_face(face_crop)
                    else:
                        target_size = int(getattr(config, 'RECOGNITION_INPUT_SIZE', 160))
                        target_size = max(96, min(224, target_size))
                        face_enhanced = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                
                # Queue for background recognition
                try:
                    self.recognition_queue.put_nowait({
                        'face': face_enhanced,
                        'embedding': cached_track_embedding,
                        'track_cache_key': matched_track_key if matched_track_key is not None else (x1, y1, x2, y2),
                        'bbox': (x1, y1, x2, y2),
                        'timestamp': current_time,
                        'original_crop': face_crop,
                        'stream_id': stream_key,
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
                            'embedding': cached_track_embedding,
                            'track_cache_key': matched_track_key if matched_track_key is not None else (x1, y1, x2, y2),
                            'bbox': (x1, y1, x2, y2),
                            'timestamp': current_time,
                            'original_crop': face_crop,
                            'stream_id': stream_key,
                        })
                    except queue.Full:
                        pass
        
        def _handle_result(result):
            student = result['student']
            confidence = result['confidence']
            timestamp = result['timestamp']
            face_crop = result['face_crop']
            bbox = result.get('bbox', None)

            if bbox:
                self._update_stable_label(
                    bbox,
                    student,
                    confidence,
                    current_time,
                    current_recognized_faces=current_recognized_faces,
                    pending_labels=pending_labels,
                )

            # Persistence/logging is delegated to VisitLogger to keep process_frame focused on CV flow.
            self.visit_logger.try_log_known_visit(student, confidence, timestamp, face_crop)

        # Process deferred results for this stream first.
        for deferred_result in self.deferred_results_by_stream.pop(stream_key, []):
            _handle_result(deferred_result)

        # STEP 3: Check for recognition results
        while True:
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break

            result_stream = str(result.get('stream_id', 'default'))
            if result_stream != stream_key:
                self.deferred_results_by_stream.setdefault(result_stream, []).append(result)
                continue

            _handle_result(result)
        
        recognized_people: List[dict] = []

        # STEP 4: Draw detection boxes with names
        for (x1, y1, x2, y2) in faces:
            # Find matching recognized face (with some tolerance for position changes)
            name_to_display = "Detecting"
            color = (0, 0, 255)  # Red for unknown
            confidence = 0.0
            is_known = False

            matched_key = self._match_bbox_key((x1, y1, x2, y2), list(current_recognized_faces.keys()))
            if matched_key is not None:
                best_match = current_recognized_faces.get(matched_key)
            else:
                best_match = None

            # Only surface detections after label confirmation to avoid ghost Unknown on empty scenes.
            if best_match is None:
                continue

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