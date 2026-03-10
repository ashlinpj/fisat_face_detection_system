"""Custom exceptions for the Face Detection System"""


class DatabaseError(Exception):
    """Base exception for database operations"""
    pass


class StudentNotFoundError(DatabaseError):
    """Student not found in database"""
    pass


class DuplicateStudentError(DatabaseError):
    """Student already exists in database"""
    pass


class InvalidInputError(DatabaseError):
    """Invalid input data"""
    pass
