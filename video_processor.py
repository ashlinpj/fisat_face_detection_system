"""
Video Processing Queue System
Automatically processes video files from a folder and runs face recognition

Accuracy features:
  - Blur detection (skip blurry face crops)
  - Per-face consecutive-frame confirmation (MIN_CONSECUTIVE_FRAMES)
  - Majority-vote identity per spatial track after video completes
  - Batch GUI update (no flooding; results shown after video finishes)
"""

import os
import cv2
import time
import threading
import logging
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import config
import database
from face_recognition_module import FaceRecognitionSystem

# ── Accuracy knobs (also settable in config.py) ──────────────────────
MIN_CONSECUTIVE_FRAMES = getattr(config, "VIDEO_MIN_CONSECUTIVE_FRAMES", 3)
BLUR_THRESHOLD         = getattr(config, "VIDEO_BLUR_THRESHOLD", 80.0)
VOTE_RATIO             = getattr(config, "VIDEO_VOTE_RATIO", 0.40)   # ≥40 % of votes to confirm
CONFIDENCE_FLOOR       = getattr(config, "VIDEO_CONFIDENCE_FLOOR", 0.50)


class VideoProcessorQueue:
    """Manages video processing queue and operations"""
    
    def __init__(self):
        """Initialize video processor"""
        # Directories
        self.videos_dir = os.path.join(config.BASE_DIR, "videos")
        self.processed_dir = os.path.join(config.BASE_DIR, "processed_videos")
        self.logs_dir = os.path.join(config.BASE_DIR, "logs")
        
        # Create directories if they don't exist
        self._ensure_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Face recognition system (shared instance)
        self.face_system = None
        
        # Processing state
        self.is_processing = False
        self.is_running = False
        self.current_video = None
        self.processing_thread = None
        
        # Statistics
        self.total_videos_processed = 0
        self.total_faces_recognized = 0
        self.current_video_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'faces_detected': 0,
            'known_faces': 0,
            'unknown_faces': 0
        }
        
        # Callbacks for GUI updates
        self.on_status_update = None
        self.on_recognition_update = None
        self.on_video_complete = None
        
        # Video filters
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        self.max_video_duration = 10 * 60  # 10 minutes in seconds
        
        # Frame processing settings
        self.frame_skip_interval = config.VIDEO_FRAME_SKIP_INTERVAL  # Process every N frames
        
        logging.info("Video Processor Queue initialized")
    
    def _ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.logs_dir, "video_processing.log")
        
        # Configure logger
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def set_face_system(self, face_system: FaceRecognitionSystem):
        """Set the face recognition system instance"""
        self.face_system = face_system
        logging.info("Face recognition system linked to video processor")
    
    def scan_videos_folder(self) -> List[str]:
        """
        Scan videos folder for processable files
        Returns: List of video file paths sorted by creation time (oldest first)
        """
        if not os.path.exists(self.videos_dir):
            return []
        
        video_files = []
        
        for filename in os.listdir(self.videos_dir):
            file_path = os.path.join(self.videos_dir, filename)
            
            # Check if it's a file
            if not os.path.isfile(file_path):
                continue
            
            # Check extension
            ext = os.path.splitext(filename)[1].lower()
            if ext not in self.supported_formats:
                continue
            
            # Skip already processed files (with _done suffix)
            if '_done' in filename:
                continue
            
            # Skip hidden or temporary files
            if filename.startswith('.') or filename.startswith('~'):
                continue
            
            video_files.append(file_path)
        
        # Sort by creation time (oldest first)
        video_files.sort(key=lambda x: os.path.getctime(x))
        
        return video_files
    
    def validate_video(self, video_path: str) -> Tuple[bool, str, dict]:
        """
        Validate video file
        Returns: (is_valid, message, metadata)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return False, "Cannot open video file", {}
            
            # Get metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps == 0:
                cap.release()
                return False, "Invalid FPS (0)", {}
            
            duration = frame_count / fps
            
            metadata = {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'width': width,
                'height': height
            }
            
            cap.release()
            
            # Check duration limit
            if duration > self.max_video_duration:
                return False, f"Video too long ({duration:.1f}s > {self.max_video_duration}s)", metadata
            
            return True, "Valid", metadata
            
        except Exception as e:
            return False, f"Validation error: {str(e)}", {}
    
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    # ── helpers ────────────────────────────────────────────────────
    @staticmethod
    def _is_blurry(face_crop: np.ndarray) -> bool:
        """Return True when a face crop is too blurry for reliable recognition."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

    @staticmethod
    def _bbox_iou(a, b) -> float:
        """Rough IoU between two (x,y,w,h) boxes – used for spatial tracking."""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    # ── main processing method ────────────────────────────────────
    def process_video(self, video_path: str) -> bool:
        """
        Process a single video file with:
          1. Blur filtering
          2. Spatial face tracking + consecutive-frame confirmation
          3. Majority-vote identity per track
          4. Batch GUI/database update AFTER video completes
        Returns True on success.
        """
        video_name = os.path.basename(video_path)
        self.current_video = video_name

        logging.info("=" * 60)
        logging.info(f"Processing video: {video_name}")

        if self.on_status_update:
            self.on_status_update(f"Processing: {video_name}")

        # ── validate ──────────────────────────────────────────────
        is_valid, message, metadata = self.validate_video(video_path)
        if not is_valid:
            logging.warning(f"Skipping video: {message}")
            if self.on_status_update:
                self.on_status_update(f"Skipped: {message}")
            return False

        logging.info(f"Duration: {self.format_timestamp(metadata['duration'])}")
        logging.info(f"FPS: {metadata['fps']:.2f}")
        logging.info(f"Resolution: {metadata['width']}x{metadata['height']}")
        logging.info(f"Total frames: {metadata['frame_count']}")

        # ── reset stats ───────────────────────────────────────────
        self.current_video_stats = {
            'total_frames': metadata['frame_count'],
            'processed_frames': 0,
            'faces_detected': 0,
            'known_faces': 0,
            'unknown_faces': 0,
        }

        # ── data structures for tracking ──────────────────────────
        # Each "track" is a spatial face tracked across frames.
        # track_id -> { "votes": [student_id, …],
        #               "confs": [float, …],
        #               "names": [str, …],
        #               "last_bbox": (x,y,w,h),
        #               "first_ts": str, "last_ts": str,
        #               "consecutive": int,
        #               "last_prediction": student_id | None }
        tracks: Dict[int, dict] = {}
        next_track_id = 0
        IOU_MATCH = 0.30  # min IoU to consider same track

        try:
            cap = cv2.VideoCapture(video_path)
            frame_number = 0

            while cap.isOpened() and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1

                # skip frames
                if frame_number % self.frame_skip_interval != 0:
                    continue

                self.current_video_stats['processed_frames'] += 1

                video_ts_sec = frame_number / metadata['fps']
                video_ts = self.format_timestamp(video_ts_sec)

                if not self.face_system:
                    continue

                faces = self.face_system.detect_faces(frame)
                if not faces:
                    # decay consecutive counters for all active tracks
                    for tid in tracks:
                        tracks[tid]['consecutive'] = 0
                    continue

                self.current_video_stats['faces_detected'] += len(faces)

                matched_tracks = set()

                for bbox in faces:
                    x, y, w, h = bbox
                    face_crop = frame[y:y+h, x:x+w]
                    if face_crop.size == 0:
                        continue

                    # ── blur check ────────────────────────────────
                    if self._is_blurry(face_crop):
                        continue

                    # ── recognise ─────────────────────────────────
                    student, confidence = self.face_system.recognize_face(face_crop)
                    conf_val = float(confidence) if confidence else 0.0

                    if student is not None and conf_val >= CONFIDENCE_FLOOR:
                        pred_id   = student.get('student_id', 'Unknown')
                        pred_name = student.get('name', 'Unknown')
                        pred_dbid = student.get('id')
                    else:
                        pred_id   = None
                        pred_name = 'Unknown'
                        pred_dbid = None

                    # ── match to an existing track by IoU ─────────
                    best_tid = None
                    best_iou = 0.0
                    for tid, trk in tracks.items():
                        if tid in matched_tracks:
                            continue
                        iou = self._bbox_iou(bbox, trk['last_bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_tid = tid

                    if best_tid is not None and best_iou >= IOU_MATCH:
                        tid = best_tid
                    else:
                        # new track
                        tid = next_track_id
                        next_track_id += 1
                        tracks[tid] = {
                            'votes': [], 'confs': [], 'names': [],
                            'db_ids': [],
                            'last_bbox': bbox,
                            'first_ts': video_ts, 'last_ts': video_ts,
                            'consecutive': 0, 'last_prediction': None,
                        }

                    trk = tracks[tid]
                    trk['last_bbox'] = bbox
                    trk['last_ts'] = video_ts

                    # consecutive-frame counter
                    if pred_id == trk['last_prediction'] and pred_id is not None:
                        trk['consecutive'] += 1
                    else:
                        trk['consecutive'] = 1 if pred_id is not None else 0
                    trk['last_prediction'] = pred_id

                    # store vote
                    trk['votes'].append(pred_id)        # None = unknown
                    trk['confs'].append(conf_val)
                    trk['names'].append(pred_name)
                    trk['db_ids'].append(pred_dbid)

                    matched_tracks.add(tid)

                # decay unmatched tracks
                for tid in tracks:
                    if tid not in matched_tracks:
                        tracks[tid]['consecutive'] = 0

                # ── progress ──────────────────────────────────────
                if self.on_status_update and frame_number % (self.frame_skip_interval * 30) == 0:
                    pct = (frame_number / metadata['frame_count']) * 100
                    self.on_status_update(f"Processing: {video_name}  ({pct:.0f}%)")

                if frame_number % (self.frame_skip_interval * 50) == 0:
                    pct = (frame_number / metadata['frame_count']) * 100
                    logging.info(f"Progress: {pct:.1f}% ({frame_number}/{metadata['frame_count']})")

            cap.release()

            # ══════════════════════════════════════════════════════
            #  POST-PROCESSING: majority vote per track
            # ══════════════════════════════════════════════════════
            confirmed_results = []   # list of dicts for GUI + DB
            unknown_count = 0

            for tid, trk in tracks.items():
                valid_votes = [v for v in trk['votes'] if v is not None]

                if not valid_votes:
                    unknown_count += 1
                    confirmed_results.append({
                        'name': 'Unknown', 'student_id': None,
                        'student_db_id': None,
                        'timestamp': trk['first_ts'],
                        'confidence': 0.0, 'is_known': False,
                    })
                    continue

                counter = Counter(valid_votes)
                winner_id, winner_count = counter.most_common(1)[0]
                total = len(trk['votes'])
                ratio = winner_count / total

                # require both majority AND enough consecutive frames
                max_consecutive = 0
                run = 0
                for v in trk['votes']:
                    if v == winner_id:
                        run += 1
                        max_consecutive = max(max_consecutive, run)
                    else:
                        run = 0

                if ratio >= VOTE_RATIO and max_consecutive >= MIN_CONSECUTIVE_FRAMES:
                    # get average confidence for the winning id
                    winner_confs = [c for v, c in zip(trk['votes'], trk['confs']) if v == winner_id]
                    avg_conf = sum(winner_confs) / len(winner_confs) if winner_confs else 0.0

                    # look up name from votes
                    idx = trk['votes'].index(winner_id)
                    name   = trk['names'][idx]
                    db_id  = trk['db_ids'][idx]

                    confirmed_results.append({
                        'name': name, 'student_id': winner_id,
                        'student_db_id': db_id,
                        'timestamp': trk['first_ts'],
                        'confidence': avg_conf, 'is_known': True,
                    })
                else:
                    unknown_count += 1
                    confirmed_results.append({
                        'name': 'Unknown', 'student_id': None,
                        'student_db_id': None,
                        'timestamp': trk['first_ts'],
                        'confidence': 0.0, 'is_known': False,
                    })

            # ── deduplicate (same student_id appearing in multiple tracks) ─
            seen_ids = set()
            final_results = []
            for r in confirmed_results:
                if r['is_known']:
                    if r['student_id'] in seen_ids:
                        continue
                    seen_ids.add(r['student_id'])
                final_results.append(r)

            # ── write to database & update stats ──────────────────
            known_results   = [r for r in final_results if r['is_known']]
            unknown_results = [r for r in final_results if not r['is_known']]

            for r in known_results:
                database.log_visit(
                    student_db_id=r['student_db_id'],
                    student_id=r['student_id'],
                    student_name=r['name'],
                    source_type='video',
                    video_name=video_name,
                    video_timestamp=r['timestamp'],
                )

            self.current_video_stats['known_faces']   = len(known_results)
            self.current_video_stats['unknown_faces']  = len(unknown_results)

            # ── summary log ───────────────────────────────────────
            logging.info("-" * 60)
            logging.info(f"Completed: {video_name}")
            logging.info(f"Tracks found: {len(tracks)}")
            logging.info(f"Confirmed identities: {len(known_results)}")
            logging.info(f"Unknown tracks: {len(unknown_results)}")
            for r in known_results:
                logging.info(f"  ✓ {r['name']} ({r['student_id']}) at {r['timestamp']}  conf={r['confidence']:.2f}")
            if not known_results:
                logging.warning("No faces were confirmed in this video")

            self.total_videos_processed += 1
            self.total_faces_recognized += len(known_results)

            # ── batch GUI update ──────────────────────────────────
            if self.on_video_complete:
                self.on_video_complete(video_name, self.current_video_stats, final_results)

            # ── move video ────────────────────────────────────────
            self._mark_video_processed(video_path)
            return True

        except Exception as e:
            logging.error(f"Error processing video: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def _mark_video_processed(self, video_path: str):
        """Mark video as processed by renaming and moving"""
        try:
            filename = os.path.basename(video_path)
            name, ext = os.path.splitext(filename)
            
            # New filename with _done suffix
            new_filename = f"{name}_done{ext}"
            new_path = os.path.join(self.processed_dir, new_filename)
            
            # Move and rename
            os.rename(video_path, new_path)
            
            logging.info(f"Video marked as processed: {new_filename}")
            logging.info(f"Moved to: {self.processed_dir}")
            
        except Exception as e:
            logging.error(f"Error marking video as processed: {str(e)}")
    
    def start_processing(self):
        """Start the video processing queue"""
        if self.is_running:
            logging.warning("Video processor already running")
            return
        
        if self.face_system is None:
            logging.error("Face recognition system not initialized")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logging.info("Video processing started")
    
    def stop_processing(self):
        """Stop the video processing queue"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logging.info("Video processing stopped")
    
    def _processing_loop(self):
        """Main processing loop (runs in background thread)"""
        logging.info("Entering video processing loop...")
        
        while self.is_running:
            # Scan for videos
            videos = self.scan_videos_folder()
            
            if not videos:
                # No videos found
                if self.on_status_update:
                    self.on_status_update("No videos found. Waiting...")
                
                logging.info("No new videos found. Waiting for 60 seconds...")
                
                # Wait for 60 seconds (check every second for stop signal)
                for _ in range(60):
                    if not self.is_running:
                        break
                    time.sleep(1)
                
                continue
            
            # Process each video
            for video_path in videos:
                if not self.is_running:
                    break
                
                self.is_processing = True
                success = self.process_video(video_path)
                self.is_processing = False
                
                if not success:
                    logging.warning(f"Failed to process: {os.path.basename(video_path)}")
                
                # Small delay between videos
                time.sleep(2)
        
        logging.info("Exited video processing loop")
    
    def get_queue_status(self) -> dict:
        """Get current queue status"""
        videos = self.scan_videos_folder()
        
        return {
            'is_running': self.is_running,
            'is_processing': self.is_processing,
            'current_video': self.current_video,
            'pending_videos': len(videos),
            'total_processed': self.total_videos_processed,
            'total_faces_recognized': self.total_faces_recognized,
            'current_stats': self.current_video_stats
        }
    
    def upload_video(self, source_path: str) -> bool:
        """
        Upload a video file to the processing queue
        Returns: True if successful, False otherwise
        """
        try:
            filename = os.path.basename(source_path)
            dest_path = os.path.join(self.videos_dir, filename)
            
            # Check if file already exists
            if os.path.exists(dest_path):
                logging.warning(f"Video already exists: {filename}")
                return False
            
            # Validate before copying
            is_valid, message, metadata = self.validate_video(source_path)
            
            if not is_valid:
                logging.error(f"Invalid video: {message}")
                return False
            
            # Copy file
            import shutil
            shutil.copy2(source_path, dest_path)
            
            logging.info(f"Video uploaded: {filename}")
            return True
            
        except Exception as e:
            logging.error(f"Error uploading video: {str(e)}")
            return False


# Singleton instance
_video_processor = None

def get_video_processor() -> VideoProcessorQueue:
    """Get singleton video processor instance"""
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessorQueue()
    return _video_processor
