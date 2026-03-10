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


def cosine_similarity_vectorized(target_embedding: np.ndarray, 
                                  embeddings_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between target and multiple embeddings at once.
    
    PERFORMANCE OPTIMIZATION: Vectorized computation is 10-50x faster than loop.
    """
    if target_embedding is None or embeddings_matrix is None:
        return np.array([])
    
    if embeddings_matrix.size == 0:
        return np.array([])
    
    target_norm = np.linalg.norm(target_embedding)
    if target_norm == 0:
        return np.zeros(embeddings_matrix.shape[0])
    
    target_normalized = target_embedding / target_norm
    
    embeddings_norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    embeddings_norms = np.where(embeddings_norms == 0, 1, embeddings_norms)
    embeddings_normalized = embeddings_matrix / embeddings_norms
    
    similarities = embeddings_normalized @ target_normalized
    
    return similarities


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
    
    PERFORMANCE OPTIMIZATION: Uses vectorized similarity computation
    for 10-50x speedup when matching against many known faces.
    """
    if not known_faces:
        return None
    
    valid_faces = [
        student for student in known_faces 
        if student.get('face_embedding') is not None
    ]
    
    if not valid_faces:
        return None
    
    try:
        embeddings_matrix = np.vstack([
            student['face_embedding'] for student in valid_faces
        ])
        
        similarities = cosine_similarity_vectorized(embedding, embeddings_matrix)
        
        if len(similarities) == 0:
            return None
        
        sorted_indices = np.argsort(similarities)[::-1]
        
        best_idx = sorted_indices[0]
        best_score = similarities[best_idx]
        next_score = similarities[sorted_indices[1]] if len(similarities) > 1 else 0.0
        
        if best_score >= threshold and (best_score - next_score) > margin:
            return valid_faces[best_idx]
        
        return None
        
    except (ValueError, IndexError):
        scores = [
            (cosine_similarity(embedding, student.get('face_embedding')), student)
            for student in valid_faces
        ]
        
        if not scores:
            return None
        
        scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_student = scores[0]
        next_score = scores[1][0] if len(scores) > 1 else 0.0
        
        if best_score >= threshold and (best_score - next_score) > margin:
            return best_student
        
        return None
