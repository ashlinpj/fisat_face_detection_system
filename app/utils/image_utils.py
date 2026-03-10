"""Image processing utilities for the Face Detection System"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import config


def enhance_face(face_image: np.ndarray) -> np.ndarray:
    """
    Enhance face image quality for better recognition.

    7-step pipeline: resize -> sharpen -> contrast -> brightness -> CLAHE -> denoise -> return
    """
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    face_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    face_enhanced = cv2.fastNlMeansDenoisingColored(face_enhanced, None, 10, 10, 7, 21)

    return face_enhanced


def draw_face_box(frame: np.ndarray, bbox, name: str,
                  confidence: float, is_known: bool) -> np.ndarray:
    """Draw a styled face bounding box with label"""
    x1, y1, x2, y2 = bbox

    if is_known:
        color = (0, 255, 0)
        bg_color = (0, 200, 0)
    else:
        color = (0, 0, 255)
        bg_color = (0, 0, 200)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    corner_length = 15
    thickness = 3

    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)

    label = f"{name} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    cv2.rectangle(frame,
                  (x1, y1 - text_height - 10),
                  (x1 + text_width + 10, y1),
                  bg_color, -1)

    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                font, font_scale, (255, 255, 255), font_thickness)

    return frame


def resize_frame(frame: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resize frame maintaining aspect ratio"""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame


def add_timestamp_overlay(frame: np.ndarray) -> np.ndarray:
    """Add timestamp overlay to frame"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.rectangle(frame, (5, 5), (250, 35), (0, 0, 0), -1)
    cv2.putText(frame, timestamp, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame
