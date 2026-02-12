"""
Video Processing Queue System – Shared-Folder Edition
=====================================================
Processes video files that have been staged into the ``processing/`` directory
by :class:`VideoQueueMonitor`.

Architecture
------------
* **Processing logic** lives here (headless-safe – no tkinter imports).
* **Visualization data** is exposed via ``current_frame_data`` dict so the GUI
  can read it on its own timer without coupling.
* Results are batched: database writes + GUI callback happen *after* the full
  video completes.

Accuracy features
-----------------
* Blur detection (skip blurry face crops)
* Per-face consecutive-frame confirmation (``MIN_CONSECUTIVE_FRAMES``)
* Majority-vote identity per spatial track after video completes
* Unique-per-video deduplication (each student logged only once per file)
"""

import os
import cv2
import time
import threading
import logging
import numpy as np
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import config
import database
from face_recognition_module import FaceRecognitionSystem
from face_matcher import find_best_match, cosine_similarity

# ── Accuracy knobs (also settable in config.py) ──────────────────────
MIN_CONSECUTIVE_FRAMES = getattr(config, "VIDEO_MIN_CONSECUTIVE_FRAMES", 5)
BLUR_THRESHOLD         = getattr(config, "VIDEO_BLUR_THRESHOLD", 80.0)
VOTE_RATIO             = getattr(config, "VIDEO_VOTE_RATIO", 0.40)
CONFIDENCE_FLOOR       = getattr(config, "VIDEO_CONFIDENCE_FLOOR", 0.50)


def _enhance_face_fast(face_image: np.ndarray) -> np.ndarray:
    """Lightweight face enhancement for video pipeline.

    Skips the expensive PIL round-trip, sharpness/brightness tweaks, and
    fastNlMeansDenoisingColored that the full enhance_face() uses.
    Only does resize + CLAHE (the step that matters most for recognition).
    ~10x faster than FaceRecognitionSystem.enhance_face().
    """
    if face_image is None or face_image.size == 0:
        return face_image
    face_resized = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    face_out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return face_out


class VideoProcessorQueue:
    """Processes video files with face detection + recognition."""

    def __init__(self):
        # Directories
        self.videos_dir = os.path.join(config.BASE_DIR, "videos")
        self.processed_dir = os.path.join(config.BASE_DIR, "processed_videos")
        self.processing_dir = getattr(config, "VIDEO_PROCESSING_DIR",
                                       os.path.join(config.BASE_DIR, "processing"))
        self.error_dir = getattr(config, "VIDEO_ERROR_DIR",
                                  os.path.join(config.BASE_DIR, "error"))
        self.logs_dir = os.path.join(config.BASE_DIR, "logs")

        self._ensure_directories()
        self._setup_logging()

        # Face recognition system (shared)
        self.face_system: Optional[FaceRecognitionSystem] = None

        # Processing state
        self.is_processing = False
        self.is_running = False
        self.current_video: Optional[str] = None
        self.processing_thread: Optional[threading.Thread] = None

        # Statistics (session-wide)
        self.total_videos_processed = 0
        self.total_faces_recognized = 0
        self.current_video_stats: Dict[str, Any] = self._empty_stats()

        # ── Visualization data (read by GUI, written by processor) ────
        # Lock protects current_frame_data from partial writes
        self._frame_lock = threading.Lock()
        self.current_frame_data: Dict[str, Any] = {
            "frame": None,            # np.ndarray | None
            "faces": [],              # list of {bbox, name, status, confidence}
            "progress_pct": 0.0,
            "video_name": "",
            "frame_number": 0,
        }

        # Callbacks for GUI (batch updates after video)
        self.on_status_update: Optional[callable] = None
        self.on_recognition_update: Optional[callable] = None  # legacy compat
        self.on_video_complete: Optional[callable] = None

        # Filters
        self.supported_formats = getattr(config, "VIDEO_SUPPORTED_FORMATS",
                                          [".mp4", ".avi", ".mov", ".mkv", ".flv"])
        self.max_video_duration = getattr(config, "VIDEO_MAX_DURATION", 600)
        self.frame_skip_interval = getattr(config, "VIDEO_FRAME_SKIP_INTERVAL", 5)

        logging.info("VideoProcessorQueue initialized")

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "total_frames": 0,
            "processed_frames": 0,
            "faces_detected": 0,
            "known_faces": 0,
            "unknown_faces": 0,
        }

    def _ensure_directories(self):
        for d in (self.videos_dir, self.processed_dir, self.processing_dir,
                  self.error_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)

    def _setup_logging(self):
        log_file = os.path.join(self.logs_dir, "video_processing.log")
        logger = logging.getLogger()
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler(),
                ],
            )
        # Clean summary log path
        self.summary_log = os.path.join(self.logs_dir, "detection_summary.log")
    
    def _write_summary_log(self, video_name: str, known_results: list, unknown_results: list):
        """Write clean, readable summary to detection_summary.log"""
        try:
            with open(self.summary_log, 'a', encoding='utf-8') as f:
                date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"\n{'='*70}\n")
                f.write(f"📹 {video_name}\n")
                f.write(f"⏰ {date_str}\n")
                f.write(f"{'-'*70}\n")
                
                if known_results:
                    f.write(f"✅ Detected: {len(known_results)} person(s)\n\n")
                    for r in known_results:
                        f.write(f"  👤 {r['name']:20s}  [{r['student_id']}]  at {r['timestamp']}  (confidence: {r['confidence']:.0%})\n")
                else:
                    f.write("❌ No recognized faces\n")
                
                if unknown_results:
                    f.write(f"\n⚠️  Unknown: {len(unknown_results)} track(s)\n")
                
                f.write(f"{'='*70}\n")
        except Exception as e:
            logging.error(f"Failed to write summary log: {e}")

    def set_face_system(self, face_system: FaceRecognitionSystem):
        self.face_system = face_system
        logging.info("Face recognition system linked to video processor")

    # ── scanning (legacy compat for Upload-based workflow) ────────

    def scan_videos_folder(self) -> List[str]:
        """Scan local ``videos/`` for processable files (oldest first)."""
        if not os.path.exists(self.videos_dir):
            return []
        result = []
        for fname in os.listdir(self.videos_dir):
            fpath = os.path.join(self.videos_dir, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in self.supported_formats:
                continue
            if "_done" in fname or fname.startswith(".") or fname.startswith("~"):
                continue
            result.append(fpath)
        result.sort(key=lambda x: os.path.getctime(x))
        return result

    # ── validation ────────────────────────────────────────────────

    def validate_video(self, video_path: str) -> Tuple[bool, str, dict]:
        """Validate a video file; return (ok, msg, metadata)."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Cannot open video file", {}
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if fps == 0:
                return False, "Invalid FPS (0)", {}
            duration = frame_count / fps
            metadata = {"fps": fps, "frame_count": frame_count,
                        "duration": duration, "width": width, "height": height}
            if duration > self.max_video_duration:
                return False, f"Video too long ({duration:.1f}s > {self.max_video_duration}s)", metadata
            return True, "Valid", metadata
        except Exception as e:
            return False, f"Validation error: {e}", {}

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        h = td.seconds // 3600
        m = (td.seconds % 3600) // 60
        s = td.seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _is_blurry(face_crop: np.ndarray) -> bool:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

    @staticmethod
    def _bbox_iou(a, b) -> float:
        """IoU for (x1, y1, x2, y2) format bboxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # ── core processing (headless-safe) ───────────────────────────

    def process_video(self, video_path: str) -> bool:
        """
        Process a single video file.

        * Accepts a path that may be a ``.processing`` rename or a normal
          ``.mp4`` – works with both the monitor pipeline and legacy upload.
        * Exposes per-frame data via ``self.current_frame_data`` (thread-safe).
        * Batches results; writes to DB after the whole video.
        * Returns True on success.
        """
        video_name = os.path.basename(video_path)
        self.current_video = video_name
        self.is_processing = True

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
            self.is_processing = False
            return False

        logging.info(f"Duration: {self.format_timestamp(metadata['duration'])}")
        logging.info(f"FPS: {metadata['fps']:.2f}, Resolution: {metadata['width']}x{metadata['height']}")
        logging.info(f"Total frames: {metadata['frame_count']}")

        self.current_video_stats = self._empty_stats()
        self.current_video_stats["total_frames"] = metadata["frame_count"]

        # ── tracking data structures ──────────────────────────────
        tracks: Dict[int, dict] = {}
        next_track_id = 0
        IOU_MATCH = 0.30

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Cannot open video: {video_path}")
                self.is_processing = False
                return False

            frame_number = 0

            while cap.isOpened() and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1

                # skip frames for speed (process ~1 fps)
                if frame_number % self.frame_skip_interval != 0:
                    continue

                self.current_video_stats["processed_frames"] += 1
                video_ts_sec = frame_number / metadata["fps"]
                video_ts = self.format_timestamp(video_ts_sec)
                progress_pct = (frame_number / metadata["frame_count"]) * 100

                if not self.face_system:
                    continue

                faces = self.face_system.detect_faces(frame)

                # Build per-frame visualization data list
                frame_face_data: List[Dict] = []

                if not faces:
                    for tid in tracks:
                        tracks[tid]["consecutive"] = 0
                    # Update vis data (frame ref only – no copy needed for empty frames)
                    if frame_number % (self.frame_skip_interval * 3) == 0:
                        with self._frame_lock:
                            self.current_frame_data = {
                                "frame": frame,
                                "faces": [],
                                "progress_pct": progress_pct,
                                "video_name": video_name,
                                "frame_number": frame_number,
                            }
                    continue

                self.current_video_stats["faces_detected"] += len(faces)
                matched_tracks = set()
                img_h, img_w = frame.shape[:2]

                for bbox in faces:
                    x1, y1, x2, y2 = bbox

                    # Crop with padding (same as webcam process_frame)
                    pad = int((x2 - x1) * 0.3)
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(img_w, x2 + pad)
                    cy2 = min(img_h, y2 + pad)
                    face_crop = frame[cy1:cy2, cx1:cx2]
                    if face_crop.size == 0:
                        continue

                    if self._is_blurry(face_crop):
                        frame_face_data.append({
                            "bbox": bbox, "name": "Blurry", "status": "unknown",
                            "confidence": 0.0,
                        })
                        continue

                    # Fast enhance (skip expensive denoise + PIL round-trip)
                    face_enhanced = _enhance_face_fast(face_crop)

                    # ── Same recognition logic as live detection thread ──
                    # Step 1: get embedding from enhanced face
                    embedding = self.face_system.get_face_embedding(face_enhanced)
                    if embedding is None:
                        frame_face_data.append({
                            "bbox": bbox, "name": "Unknown", "status": "unknown",
                            "confidence": 0.0,
                        })
                        continue

                    # Step 2: find_best_match with margin (same as recognition_worker)
                    rec_threshold = getattr(config, 'FACE_RECOGNITION_THRESHOLD', 0.35)
                    best_match = find_best_match(
                        embedding,
                        self.face_system.known_faces,
                        threshold=rec_threshold,
                        margin=0.15,
                    )

                    # Step 3: compute cosine similarity score
                    conf_val = 0.0
                    if best_match is not None and best_match.get('face_embedding') is not None:
                        conf_val = cosine_similarity(embedding, best_match['face_embedding'])

                    student = best_match

                    if student is not None and conf_val >= CONFIDENCE_FLOOR:
                        pred_id = student.get("student_id", "Unknown")
                        pred_name = student.get("name", "Unknown")
                        pred_dbid = student.get("id")
                    else:
                        pred_id = None
                        pred_name = "Unknown"
                        pred_dbid = None

                    # Match to existing track by IoU
                    best_tid, best_iou = None, 0.0
                    for tid, trk in tracks.items():
                        if tid in matched_tracks:
                            continue
                        iou = self._bbox_iou(bbox, trk["last_bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_tid = tid

                    if best_tid is not None and best_iou >= IOU_MATCH:
                        tid = best_tid
                    else:
                        tid = next_track_id
                        next_track_id += 1
                        tracks[tid] = {
                            "votes": [], "confs": [], "names": [], "db_ids": [],
                            "last_bbox": bbox,
                            "first_ts": video_ts, "last_ts": video_ts,
                            "consecutive": 0, "last_prediction": None,
                        }

                    trk = tracks[tid]
                    trk["last_bbox"] = bbox
                    trk["last_ts"] = video_ts

                    if pred_id == trk["last_prediction"] and pred_id is not None:
                        trk["consecutive"] += 1
                    else:
                        trk["consecutive"] = 1 if pred_id is not None else 0
                    trk["last_prediction"] = pred_id

                    trk["votes"].append(pred_id)
                    trk["confs"].append(conf_val)
                    trk["names"].append(pred_name)
                    trk["db_ids"].append(pred_dbid)
                    matched_tracks.add(tid)

                    # Determine visualization status for this face
                    if pred_id is None:
                        vis_status = "unknown"
                    elif trk["consecutive"] >= MIN_CONSECUTIVE_FRAMES:
                        vis_status = "confirmed"
                    else:
                        vis_status = "pending"

                    frame_face_data.append({
                        "bbox": bbox,
                        "name": pred_name,
                        "status": vis_status,
                        "confidence": conf_val,
                    })

                # Decay unmatched
                for tid in tracks:
                    if tid not in matched_tracks:
                        tracks[tid]["consecutive"] = 0

                # ── expose frame for visualization (thread-safe) ──
                # Downscale before copying to reduce memory + time
                preview_w = getattr(config, "VIDEO_PREVIEW_WIDTH", 640)
                scale_x = preview_w / frame.shape[1]
                ph = int(frame.shape[0] * scale_x)
                preview = cv2.resize(frame, (preview_w, ph), interpolation=cv2.INTER_NEAREST)
                # Rescale bbox coords to preview size
                scaled_face_data = []
                for fd in frame_face_data:
                    bx1, by1, bx2, by2 = fd["bbox"]
                    scaled_face_data.append({
                        **fd,
                        "bbox": (int(bx1 * scale_x), int(by1 * scale_x),
                                 int(bx2 * scale_x), int(by2 * scale_x)),
                    })
                with self._frame_lock:
                    self.current_frame_data = {
                        "frame": preview,
                        "faces": scaled_face_data,
                        "progress_pct": progress_pct,
                        "video_name": video_name,
                        "frame_number": frame_number,
                    }

                # Periodic logging
                if self.on_status_update and frame_number % (self.frame_skip_interval * 30) == 0:
                    self.on_status_update(f"Processing: {video_name}  ({progress_pct:.0f}%)")
                if frame_number % (self.frame_skip_interval * 50) == 0:
                    logging.info(f"Progress: {progress_pct:.1f}% ({frame_number}/{metadata['frame_count']})")

            cap.release()

            # ══════════════════════════════════════════════════════
            #  POST-PROCESSING: majority vote per track
            # ══════════════════════════════════════════════════════
            confirmed_results = []
            unknown_count = 0

            for tid, trk in tracks.items():
                valid_votes = [v for v in trk["votes"] if v is not None]

                if not valid_votes:
                    unknown_count += 1
                    confirmed_results.append({
                        "name": "Unknown", "student_id": None,
                        "student_db_id": None,
                        "timestamp": trk["first_ts"],
                        "confidence": 0.0, "is_known": False,
                    })
                    continue

                counter = Counter(valid_votes)
                winner_id, winner_count = counter.most_common(1)[0]
                total = len(trk["votes"])
                ratio = winner_count / total

                # Max consecutive run for winner
                max_run, run = 0, 0
                for v in trk["votes"]:
                    if v == winner_id:
                        run += 1
                        max_run = max(max_run, run)
                    else:
                        run = 0

                if ratio >= VOTE_RATIO and max_run >= MIN_CONSECUTIVE_FRAMES:
                    winner_confs = [c for v, c in zip(trk["votes"], trk["confs"]) if v == winner_id]
                    avg_conf = sum(winner_confs) / len(winner_confs) if winner_confs else 0.0
                    idx = trk["votes"].index(winner_id)
                    confirmed_results.append({
                        "name": trk["names"][idx],
                        "student_id": winner_id,
                        "student_db_id": trk["db_ids"][idx],
                        "timestamp": trk["first_ts"],
                        "confidence": avg_conf,
                        "is_known": True,
                    })
                else:
                    unknown_count += 1
                    confirmed_results.append({
                        "name": "Unknown", "student_id": None,
                        "student_db_id": None,
                        "timestamp": trk["first_ts"],
                        "confidence": 0.0, "is_known": False,
                    })

            # ── deduplicate (same student_id in multiple tracks) ──
            seen_ids: set = set()
            final_results = []
            for r in confirmed_results:
                if r["is_known"]:
                    if r["student_id"] in seen_ids:
                        continue
                    seen_ids.add(r["student_id"])
                final_results.append(r)

            # ── write to database ─────────────────────────────────
            known_results = [r for r in final_results if r["is_known"]]
            unknown_results = [r for r in final_results if not r["is_known"]]

            for r in known_results:
                database.log_visit(
                    student_db_id=r["student_db_id"],
                    student_id=r["student_id"],
                    student_name=r["name"],
                    source_type="video",
                    video_name=video_name,
                    video_timestamp=r["timestamp"],
                )

            self.current_video_stats["known_faces"] = len(known_results)
            self.current_video_stats["unknown_faces"] = len(unknown_results)

            # ── summary logs ──────────────────────────────────────
            logging.info("-" * 60)
            logging.info(f"Completed: {video_name}")
            logging.info(f"Tracks found: {len(tracks)}")
            logging.info(f"Confirmed identities: {len(known_results)}")
            logging.info(f"Unknown tracks: {len(unknown_results)}")
            for r in known_results:
                logging.info(f"  OK  {r['name']} ({r['student_id']}) at {r['timestamp']}  conf={r['confidence']:.2f}")
            if not known_results:
                logging.warning("No faces were confirmed in this video")
            
            # Write clean summary log
            self._write_summary_log(video_name, known_results, unknown_results)

            self.total_videos_processed += 1
            self.total_faces_recognized += len(known_results)

            # ── batch GUI update ──────────────────────────────────
            if self.on_video_complete:
                self.on_video_complete(video_name, self.current_video_stats, final_results)

            # Clear frame data
            with self._frame_lock:
                self.current_frame_data = {
                    "frame": None, "faces": [], "progress_pct": 100.0,
                    "video_name": video_name, "frame_number": 0,
                }

            self.is_processing = False
            return True

        except Exception as e:
            logging.error(f"Error processing video: {e}")
            logging.error(traceback.format_exc())
            self.is_processing = False
            return False

    # ── legacy: move processed file (for Upload workflow) ─────────

    def _mark_video_processed(self, video_path: str):
        """Move + rename for legacy upload-based workflow."""
        try:
            fname = os.path.basename(video_path)
            name, ext = os.path.splitext(fname)
            new_fname = f"{name}_done{ext}"
            new_path = os.path.join(self.processed_dir, new_fname)
            os.rename(video_path, new_path)
            logging.info(f"Video marked as processed: {new_fname}")
        except Exception as e:
            logging.error(f"Error marking video as processed: {e}")

    # ── legacy loop (Upload tab) ──────────────────────────────────

    def start_processing(self):
        """Start the legacy processing loop (scans videos/ folder)."""
        if self.is_running:
            return
        if self.face_system is None:
            logging.error("Face recognition system not initialized")
            return
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logging.info("Video processing started (legacy mode)")

    def stop_processing(self):
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logging.info("Video processing stopped")

    def _processing_loop(self):
        """Legacy loop: scan videos/ -> process -> move to processed/."""
        while self.is_running:
            videos = self.scan_videos_folder()
            if not videos:
                if self.on_status_update:
                    self.on_status_update("No videos found. Waiting...")
                logging.info("No new videos found. Waiting for 60 seconds...")
                for _ in range(60):
                    if not self.is_running:
                        break
                    time.sleep(1)
                continue
            for vp in videos:
                if not self.is_running:
                    break
                self.is_processing = True
                success = self.process_video(vp)
                self.is_processing = False
                if success:
                    self._mark_video_processed(vp)
                else:
                    logging.warning(f"Failed to process: {os.path.basename(vp)}")
                time.sleep(2)

    # ── queue status (for GUI) ────────────────────────────────────

    def get_queue_status(self) -> dict:
        videos = self.scan_videos_folder()
        return {
            "is_running": self.is_running,
            "is_processing": self.is_processing,
            "current_video": self.current_video,
            "pending_videos": len(videos),
            "total_processed": self.total_videos_processed,
            "total_faces_recognized": self.total_faces_recognized,
            "current_stats": self.current_video_stats,
        }

    def get_frame_snapshot(self) -> Dict[str, Any]:
        """Thread-safe read of current frame + face annotations."""
        with self._frame_lock:
            return dict(self.current_frame_data)

    # ── upload helper (legacy) ────────────────────────────────────

    def upload_video(self, source_path: str) -> bool:
        try:
            import shutil
            fname = os.path.basename(source_path)
            dest = os.path.join(self.videos_dir, fname)
            if os.path.exists(dest):
                logging.warning(f"Video already exists: {fname}")
                return False
            is_valid, msg, _ = self.validate_video(source_path)
            if not is_valid:
                logging.error(f"Invalid video: {msg}")
                return False
            shutil.copy2(source_path, dest)
            logging.info(f"Video uploaded: {fname}")
            return True
        except Exception as e:
            logging.error(f"Error uploading video: {e}")
            return False


# ── Singleton ─────────────────────────────────────────────────────
_video_processor: Optional[VideoProcessorQueue] = None


def get_video_processor() -> VideoProcessorQueue:
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessorQueue()
    return _video_processor
