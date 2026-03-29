"""
Face Matcher — Gallery-based matching with adaptive margin.

Instead of comparing a probe embedding against a single averaged embedding
per person, this module compares against ALL stored registration samples
(the "gallery") and uses the top-K average as the match score.  An adaptive
margin ensures that when two candidates are close in score, a proportionally
larger gap is required before declaring a match.
"""

import numpy as np
from typing import List, Optional, Dict
import config


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _gallery_score(probe: np.ndarray, gallery: List[np.ndarray], top_k: int = 5) -> float:
    """Score a probe against a gallery of embeddings using top-K average.

    This is much more robust than comparing against a single averaged
    embedding because it naturally accounts for pose/lighting variation
    captured during registration.
    """
    if not gallery:
        return 0.0

    sims = [cosine_similarity(probe, g) for g in gallery]
    sims.sort(reverse=True)

    # Take average of top-K matches (or all if fewer than K)
    k = min(top_k, len(sims))
    return float(np.mean(sims[:k]))


def find_best_match(
    embedding: np.ndarray,
    known_faces: List[Dict],
    threshold: float = 0.5,
    margin: float = 0.10,
) -> Optional[Dict]:
    """Find the best match using gallery-based matching with adaptive margin.

    For each registered person the function computes:
      • A *gallery score* (top-K average across all stored registration
        samples) when multi-embeddings are available, OR
      • A plain cosine similarity against the single stored embedding
        as a fallback.

    A match is accepted only when:
      1. The best score exceeds *threshold*.
      2. The gap between the best and second-best score exceeds an
         *adaptive margin* that scales up when both candidates score
         high (i.e. when confusion is most likely).
    """
    use_gallery = getattr(config, 'FACE_MULTI_SAMPLE_MATCH', True)
    top_k = int(getattr(config, 'FACE_GALLERY_TOP_K', 5))
    use_adaptive = getattr(config, 'FACE_ADAPTIVE_MARGIN', True)

    scores: List[tuple] = []  # (score, student)

    for student in known_faces:
        # Prefer multi-embedding gallery if available
        gallery = student.get('face_embeddings_multi')
        avg_emb = student.get('face_embedding')

        if use_gallery and gallery and len(gallery) > 0:
            score = _gallery_score(embedding, gallery, top_k=top_k)
        elif avg_emb is not None:
            score = cosine_similarity(embedding, avg_emb)
        else:
            continue

        scores.append((score, student))

    if not scores:
        return None

    scores.sort(reverse=True, key=lambda x: x[0])
    best_score, best_student = scores[0]
    next_score = scores[1][0] if len(scores) > 1 else 0.0

    # --- Adaptive margin ---------------------------------------------------
    # When both top-2 candidates score high the risk of confusion is greatest,
    # so we require a *larger* gap.  The adaptive margin is:
    #   effective_margin = base_margin + scale * next_score
    # Example with base=0.10, scale=0.08, next=0.65:
    #   effective = 0.10 + 0.08*0.65 = 0.152
    # This makes it much harder for a wrong person to "steal" the match.
    if use_adaptive and next_score > 0:
        adaptive_scale = 0.08
        effective_margin = margin + adaptive_scale * next_score
    else:
        effective_margin = margin

    # Accept only if above threshold AND margin is satisfied
    if best_score >= threshold and (best_score - next_score) >= effective_margin:
        return best_student

    return None
