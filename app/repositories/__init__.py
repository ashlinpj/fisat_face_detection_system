"""Repository layer - Database access and persistence"""

from app.repositories.connection_pool import ConnectionPool, init_database
from app.repositories.student_repository import StudentRepository
from app.repositories.visit_repository import VisitRepository
from app.repositories.unknown_face_repository import UnknownFaceRepository
