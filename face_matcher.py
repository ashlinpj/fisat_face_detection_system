import numpy as np
from typing import List, Optional, Dict

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    
    a, b = np.array(a), np.array(b)
    if a.shape != b.shape:
        return 0.0
    
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute euclidean distance between two vectors"""
    if a is None or b is None:
        return float('inf')
    
    a, b = np.array(a), np.array(b)
    if a.shape != b.shape:
        return float('inf')
    
    return np.linalg.norm(a - b)


def find_best_match(
    embedding: np.ndarray,
    known_faces: List[Dict],
    threshold: float = 0.5,
    margin: float = 0.15
) -> Optional[Dict]:
    """
    Find the best match for the given embedding in known_faces.
    Returns a match only if best similarity is above threshold and
    the margin to the next best is significant.
    """
    scores = [
        (cosine_similarity(embedding, student.get('face_embedding')), student)
        for student in known_faces
        if student.get('face_embedding') is not None
    ]
    
    if not scores:
        return None
    
    scores.sort(reverse=True, key=lambda x: x[0])
    best_score, best_student = scores[0]
    next_score = scores[1][0] if len(scores) > 1 else 0.0
    
    if best_score >= threshold and (best_score - next_score) > margin:
        return best_student
    
    return None
