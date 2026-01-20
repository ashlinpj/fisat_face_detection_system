import numpy as np
from typing import List, Optional, Dict

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        return 0.0
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute euclidean distance between two vectors."""
    if a is None or b is None:
        return float('inf')
    a = np.array(a)
    b = np.array(b)
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
    Only return a match if the best similarity is above threshold and
    the margin to the next best is significant.
    """
    scores = []
    for student in known_faces:
        emb = student.get('face_embedding')
        if emb is not None:
            score = cosine_similarity(embedding, emb)
            scores.append((score, student))
    if not scores:
        return None
    scores.sort(reverse=True, key=lambda x: x[0])
    best_score, best_student = scores[0]
    next_score = scores[1][0] if len(scores) > 1 else 0.0
    # Only accept if above threshold and margin
    if best_score >= threshold and (best_score - next_score) > margin:
        return best_student
    return None
