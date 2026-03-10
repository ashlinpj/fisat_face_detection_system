"""Detection service - Face detection using DNN SSD, Cascade, and YOLO"""

import logging
import os
import cv2
import numpy as np
from typing import List, Tuple

import config

logger = logging.getLogger(__name__)


class DetectionService:
    """Handles face detection using multiple backends (DNN SSD, Cascade, YOLO).

    Dependencies (use_gpu, gpu_name) are injected via the constructor.
    """

    def __init__(self, use_gpu=False, gpu_name="CPU"):
        self.use_gpu = use_gpu
        self.gpu_name = gpu_name
        self.yolo_model = None
        self.dnn_net = None

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        if use_gpu and getattr(config, 'USE_GPU', True):
            self._load_yolo_gpu()

        self._load_dnn_detector()

    def _load_yolo_gpu(self):
        """Load YOLO model with GPU acceleration"""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            self.yolo_model.to('cuda')
            logger.info("YOLO loaded on GPU")
        except Exception as e:
            logger.warning("YOLO GPU load failed: %s", e)
            self.yolo_model = None

    def _load_dnn_detector(self):
        """Load OpenCV DNN face detector with GPU acceleration"""
        try:
            model_file = "res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "deploy.prototxt"

            if not os.path.exists(model_file):
                logger.info("Downloading DNN face detector model...")
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
                    logger.info("DNN face detector loaded on GPU")
                except Exception:
                    self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    logger.info("DNN face detector loaded on CPU")
            else:
                self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("DNN face detector loaded on CPU")
        except Exception as e:
            logger.warning("DNN detector load failed: %s", e)
            self.dnn_net = None

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Face detection - returns SQUARE bounding boxes.

        Tries DNN first, falls back to Cascade classifier.
        """
        h, w = frame.shape[:2]
        result = []
        min_face_size = getattr(config, 'MIN_FACE_SIZE', 60)

        use_dnn = getattr(config, 'USE_DNN_DETECTOR', False)
        if use_dnn and self.dnn_net is not None:
            try:
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

                        if width < min_face_size or height < min_face_size:
                            continue

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
            except Exception:
                pass

        # Fallback: Cascade classifier
        scale = getattr(config, 'DETECTION_SCALE', 0.5)
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        min_neighbors = getattr(config, 'CASCADE_MIN_NEIGHBORS', 5)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=min_neighbors,
            minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, fw, fh) in faces:
            x1, y1 = int(x / scale), int(y / scale)
            orig_w, orig_h = int(fw / scale), int(fh / scale)

            if orig_w < min_face_size or orig_h < min_face_size:
                continue

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

            if final_size >= min_face_size:
                result.append((x1_sq, y1_sq, x1_sq + final_size, y1_sq + final_size))

        return result
