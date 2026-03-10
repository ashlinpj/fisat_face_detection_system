"""Dependency Injection Container - Wires all layers together.

The Container creates and connects all repositories, services, and other
components following proper dependency injection. Each component receives
its dependencies through its constructor rather than importing them.

Usage:
    container = Container()
    container.frame_processor.process_frame(frame)
    container.student_repo.get_all()
"""

import logging

import config
from app.repositories.connection_pool import ConnectionPool, init_database
from app.repositories.student_repository import StudentRepository
from app.repositories.visit_repository import VisitRepository
from app.repositories.unknown_face_repository import UnknownFaceRepository
from app.services.gpu_service import check_gpu
from app.services.detection_service import DetectionService
from app.services.recognition_service import RecognitionService
from app.services.registration_service import RegistrationService
from app.services.visit_service import VisitService
from app.services.frame_processor import FrameProcessor
from app.services.report_service import ReportService


class Container:
    """Creates and wires all application components.

    Initialization order:
        1. Database (connection pool + schema)
        2. Repositories (receive pool)
        3. GPU check
        4. Services (receive repos and other services)
    """

    _logger = logging.getLogger(__name__)

    def __init__(self):
        # --- Infrastructure ---
        self.connection_pool = ConnectionPool()
        init_database(self.connection_pool)

        # --- Repository layer ---
        self.student_repo = StudentRepository(self.connection_pool)
        self.visit_repo = VisitRepository(self.connection_pool)
        self.unknown_face_repo = UnknownFaceRepository(self.connection_pool)

        # --- GPU ---
        self.gpu_available, self.gpu_name = check_gpu()

        # --- Service layer ---
        self.detection_service = DetectionService(
            use_gpu=self.gpu_available,
            gpu_name=self.gpu_name
        )

        self.known_faces = self.student_repo.get_all()
        self._logger.info("Loaded %d known faces", len(self.known_faces))

        self.recognition_service = RecognitionService(
            self.known_faces,
            use_threaded=getattr(config, 'USE_THREADED_RECOGNITION', True)
        )

        self.registration_service = RegistrationService(
            self.detection_service,
            self.recognition_service,
            self.student_repo
        )

        self.visit_service = VisitService(self.visit_repo)

        self.frame_processor = FrameProcessor(
            self.detection_service,
            self.recognition_service,
            self.visit_service
        )

        self.report_service = ReportService(
            self.student_repo,
            self.visit_repo
        )

        mode = "GPU+Threaded" if self.gpu_available else "CPU+Threaded"
        self._logger.info("Container initialized (%s Mode)", mode)

    def reload_known_faces(self):
        """Reload known faces from database after registration changes"""
        self.known_faces = self.student_repo.get_all()
        self.recognition_service.reload_known_faces(self.known_faces)
        self._logger.info("Loaded %d known faces", len(self.known_faces))

    def stop(self):
        """Clean up resources"""
        self.recognition_service.stop()
