"""Quick diagnostic test for detection and recognition services"""

import sys
import os
import cv2
import numpy as np

# Ensure project root is on sys.path so imports work when run from tests/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.utils.logging_config import setup_logging
from app.container import Container


def test_detection_and_recognition():
    """Test if detection and recognition services are initialized correctly"""

    print("=" * 60)
    print("DETECTION & RECOGNITION DIAGNOSTIC TEST")
    print("=" * 60)

    # Setup logging
    setup_logging()
    print("\n✓ Logging configured")

    # Create container
    try:
        container = Container()
        print("✓ Container initialized")
    except Exception as e:
        print(f"✗ Container initialization failed: {e}")
        return False

    # Check detection service
    print("\n--- Detection Service ---")
    try:
        detection = container.detection_service
        print(f"✓ Detection service loaded: {type(detection).__name__}")
        print(f"  - GPU mode: {detection.use_gpu}")
        print(f"  - GPU device: {detection.gpu_name}")
        print(f"  - YOLO loaded: {detection.yolo_model is not None}")
        print(f"  - DNN loaded: {detection.dnn_net is not None}")

        # Test with dummy frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detection.detect_faces(test_frame)
        print(f"✓ Detection method callable (returned {len(faces)} faces on blank frame)")

    except Exception as e:
        print(f"✗ Detection service error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check recognition service
    print("\n--- Recognition Service ---")
    try:
        recognition = container.recognition_service
        print(f"✓ Recognition service loaded: {type(recognition).__name__}")
        print(f"  - Known faces loaded: {len(recognition.known_faces)}")
        print(f"  - Recognition thread exists: {recognition.recognition_thread is not None}")
        if recognition.recognition_thread:
            print(f"  - Background worker running: {recognition.recognition_thread.is_alive()}")

        # Test embedding extraction
        test_face = np.ones((112, 112, 3), dtype=np.uint8) * 128
        embedding = recognition.get_face_embedding(test_face)
        if embedding is not None:
            print(f"✓ Embedding extraction works (dim: {len(embedding)})")
        else:
            print("⚠ Embedding extraction returned None (may need valid face)")

    except Exception as e:
        print(f"✗ Recognition service error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check GPU service
    print("\n--- GPU Service ---")
    try:
        gpu_available = container.gpu_available
        gpu_info = container.gpu_name
        if gpu_available:
            print(f"✓ GPU available: {gpu_info}")
        else:
            print(f"⚠ GPU not available (CPU mode): {gpu_info}")
    except Exception as e:
        print(f"✗ GPU service error: {e}")

    # Check frame processor
    print("\n--- Frame Processor ---")
    try:
        processor = container.frame_processor
        print(f"✓ Frame processor loaded: {type(processor).__name__}")

        # Test frame processing with blank frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, detections = processor.process_frame(test_frame)
        print(f"✓ Frame processing works (frame shape: {annotated.shape}, detections: {len(detections)})")

    except Exception as e:
        print(f"✗ Frame processor error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULT: ✓ ALL CORE SERVICES WORKING")
    print("=" * 60)
    print("\nDetection and recognition are initialized correctly.")
    print("The system is ready for real-time face detection.")

    # Cleanup
    container.stop()

    return True


if __name__ == "__main__":
    try:
        success = test_detection_and_recognition()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
