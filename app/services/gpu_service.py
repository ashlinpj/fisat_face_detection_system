"""GPU detection and configuration"""

import logging
import os
import cv2

logger = logging.getLogger(__name__)


def configure_gpu():
    """Configure TensorFlow GPU memory growth"""
    try:
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info("TensorFlow GPU configured: %d device(s)", len(physical_devices))
    except Exception:
        pass


# Run GPU configuration at import time (matches original behavior)
configure_gpu()


def check_gpu():
    """
    Check if CUDA GPU is available.

    Returns (gpu_available: bool, gpu_name: str)
    """
    gpu_available = False
    gpu_name = "CPU"

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("PyTorch GPU Detected: %s", gpu_name)
            logger.info("  CUDA Version: %s", torch.version.cuda)
    except Exception:
        pass

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_available = True
            logger.info("TensorFlow GPU: %d device(s) available", len(gpus))
            for gpu in gpus:
                logger.info("  %s", gpu.name)
    except Exception:
        pass

    try:
        cv2_cuda = cv2.cuda.getCudaEnabledDeviceCount()
        if cv2_cuda > 0:
            logger.info("OpenCV CUDA devices: %d", cv2_cuda)
    except Exception:
        pass

    if not gpu_available:
        logger.warning("No GPU detected - running on CPU mode")

    return gpu_available, gpu_name
