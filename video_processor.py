"""
Video Processing Queue System
Automatically processes video files from a folder and runs face recognition
"""

import os
import cv2
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
import config
import database
from face_recognition_module import FaceRecognitionSystem


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
    
    def process_video(self, video_path: str) -> bool:
        """
        Process a single video file
        Returns: True if successful, False otherwise
        """
        video_name = os.path.basename(video_path)
        self.current_video = video_name
        
        logging.info("="*60)
        logging.info(f"Processing video: {video_name}")
        
        if self.on_status_update:
            self.on_status_update(f"Processing: {video_name}")
        
        # Validate video
        is_valid, message, metadata = self.validate_video(video_path)
        
        if not is_valid:
            logging.warning(f"Skipping video: {message}")
            if self.on_status_update:
                self.on_status_update(f"Skipped: {message}")
            return False
        
        # Log metadata
        logging.info(f"Duration: {self.format_timestamp(metadata['duration'])}")
        logging.info(f"FPS: {metadata['fps']:.2f}")
        logging.info(f"Resolution: {metadata['width']}x{metadata['height']}")
        logging.info(f"Total frames: {metadata['frame_count']}")
        
        # Reset statistics
        self.current_video_stats = {
            'total_frames': metadata['frame_count'],
            'processed_frames': 0,
            'faces_detected': 0,
            'known_faces': 0,
            'unknown_faces': 0
        }
        
        # Process video frames
        try:
            cap = cv2.VideoCapture(video_path)
            frame_number = 0
            recognized_in_video = set()  # Track unique recognitions
            last_recognition = {}  # Track last recognition time per person
            
            while cap.isOpened() and self.is_running:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_number += 1
                
                # Skip frames based on interval
                if frame_number % self.frame_skip_interval != 0:
                    continue
                
                self.current_video_stats['processed_frames'] += 1
                
                # Calculate video timestamp
                video_timestamp_seconds = frame_number / metadata['fps']
                video_timestamp = self.format_timestamp(video_timestamp_seconds)
                
                # Run face detection and recognition
                if self.face_system:
                    faces = self.face_system.detect_faces(frame)
                    
                    if faces:
                        self.current_video_stats['faces_detected'] += len(faces)
                        
                        for bbox in faces:
                            x, y, w, h = bbox
                            
                            # Crop face
                            face_crop = frame[y:y+h, x:x+w]
                            
                            if face_crop.size == 0:
                                continue
                            
                            # Recognize face (returns student dict and confidence)
                            student, confidence = self.face_system.recognize_face(face_crop)
                            
                            if student is not None:
                                # Extract student info
                                student_name = student.get('name', 'Unknown')
                                student_id = student.get('student_id', 'Unknown')
                                
                                # Check if we should log this recognition
                                # Avoid duplicate logs within 30 seconds
                                current_time = time.time()
                                should_log = True
                                
                                # Use student_id as key for duplicate tracking
                                if student_id in last_recognition:
                                    time_diff = current_time - last_recognition[student_id]
                                    if time_diff < 30:  # 30 second threshold
                                        should_log = False
                                
                                if should_log:
                                    # Log to console and file
                                    logging.info(f"Recognized: {student_name} ({student_id}) at {video_timestamp} (confidence: {confidence:.2f})")
                                    
                                    # Update GUI
                                    if self.on_recognition_update:
                                        self.on_recognition_update(f"{student_name} ({student_id})", video_timestamp, confidence)
                                    
                                    # Log to database
                                    database.log_visit(
                                        student_db_id=student['id'],
                                        student_id=student_id,
                                        student_name=student_name,
                                        source_type='video',
                                        video_name=video_name,
                                        video_timestamp=video_timestamp
                                    )
                                    
                                    recognized_in_video.add(student_id)
                                    last_recognition[student_id] = current_time
                                    self.current_video_stats['known_faces'] += 1
                            else:
                                # Unknown face
                                self.current_video_stats['unknown_faces'] += 1
                                logging.debug(f"Unknown face at {video_timestamp}")
                
                # Update progress
                progress = (frame_number / metadata['frame_count']) * 100
                if frame_number % (self.frame_skip_interval * 10) == 0:  # Log every 10 processed frames
                    logging.info(f"Progress: {progress:.1f}% ({frame_number}/{metadata['frame_count']} frames)")
            
            cap.release()
            
            # Summary
            logging.info("-"*60)
            logging.info(f"Completed: {video_name}")
            logging.info(f"Total frames: {metadata['frame_count']}")
            logging.info(f"Processed frames: {self.current_video_stats['processed_frames']}")
            logging.info(f"Faces detected: {self.current_video_stats['faces_detected']}")
            logging.info(f"Known faces: {self.current_video_stats['known_faces']}")
            logging.info(f"Unknown faces: {self.current_video_stats['unknown_faces']}")
            logging.info(f"Unique people recognized: {len(recognized_in_video)}")
            
            if len(recognized_in_video) == 0:
                logging.warning("No faces were recognized in this video")
            
            self.total_videos_processed += 1
            self.total_faces_recognized += self.current_video_stats['known_faces']
            
            # Mark video as processed
            self._mark_video_processed(video_path)
            
            if self.on_video_complete:
                self.on_video_complete(video_name, self.current_video_stats)
            
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
