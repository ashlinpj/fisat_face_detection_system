"""
Diagnostic script: Check pairwise similarity between all registered students.
Run this after re-registering all students to verify embedding quality.

Usage:
    python diagnose_embeddings.py
"""

import numpy as np
import database
import config


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    print("=" * 60)
    print("  Face Embedding Diagnostic Report")
    print("=" * 60)

    print(f"\n  Model: {config.FACE_EMBEDDING_MODEL}")
    print(f"  Recognition threshold: {config.FACE_RECOGNITION_THRESHOLD}")
    print(f"  Match margin: {config.FACE_MATCH_MARGIN}")
    print(f"  Multi-sample match: {getattr(config, 'FACE_MULTI_SAMPLE_MATCH', False)}")
    print(f"  Adaptive margin: {getattr(config, 'FACE_ADAPTIVE_MARGIN', False)}")

    database.init_database()
    students = database.get_all_students()

    if not students:
        print("\n  ✗ No students registered!")
        return

    print(f"\n  Registered students: {len(students)}\n")

    # Show each student's info
    for s in students:
        emb = s.get('face_embedding')
        multi = s.get('face_embeddings_multi')
        dim = emb.shape[0] if emb is not None else 0
        n_gallery = len(multi) if multi else 0
        print(f"  • {s['name']:15s} (ID: {s['student_id']:5s})  "
              f"embedding: {dim}-dim, gallery: {n_gallery} samples")

    # Pairwise comparison
    print("\n" + "-" * 60)
    print("  Pairwise Cosine Similarity (averaged embeddings)")
    print("-" * 60)

    threshold = config.FACE_RECOGNITION_THRESHOLD
    warnings = []

    for i, s1 in enumerate(students):
        for j, s2 in enumerate(students):
            if j <= i:
                continue
            e1 = s1.get('face_embedding')
            e2 = s2.get('face_embedding')
            if e1 is None or e2 is None:
                print(f"  {s1['name']:12s} vs {s2['name']:12s}:  ??? (missing embedding)")
                continue

            sim = cosine_similarity(e1, e2)
            status = ""
            if sim >= threshold:
                status = "  ⚠ DANGER: above threshold!"
                warnings.append((s1['name'], s2['name'], sim))
            elif sim >= threshold - 0.15:
                status = "  ⚠ CLOSE to threshold"
                warnings.append((s1['name'], s2['name'], sim))

            print(f"  {s1['name']:12s} vs {s2['name']:12s}:  {sim:.4f}{status}")

    # Gallery pairwise comparison
    has_gallery = any(s.get('face_embeddings_multi') for s in students)
    if has_gallery:
        print("\n" + "-" * 60)
        print("  Gallery Cross-Match (max sim between any pair of gallery samples)")
        print("-" * 60)

        for i, s1 in enumerate(students):
            for j, s2 in enumerate(students):
                if j <= i:
                    continue
                g1 = s1.get('face_embeddings_multi')
                g2 = s2.get('face_embeddings_multi')
                if not g1 or not g2:
                    continue

                max_sim = 0.0
                min_sim = 1.0
                all_sims = []
                for e1 in g1:
                    for e2 in g2:
                        s_val = cosine_similarity(e1, e2)
                        all_sims.append(s_val)
                        max_sim = max(max_sim, s_val)
                        min_sim = min(min_sim, s_val)

                avg_sim = np.mean(all_sims)
                status = ""
                if max_sim >= threshold:
                    status = "  ⚠ SOME SAMPLES OVERLAP"

                print(f"  {s1['name']:12s} vs {s2['name']:12s}:  "
                      f"avg={avg_sim:.4f}  max={max_sim:.4f}  min={min_sim:.4f}{status}")

    # Self-consistency check
    print("\n" + "-" * 60)
    print("  Self-Consistency (gallery intra-class spread)")
    print("-" * 60)

    for s in students:
        gallery = s.get('face_embeddings_multi')
        if not gallery or len(gallery) < 2:
            print(f"  {s['name']:12s}:  N/A (< 2 gallery samples)")
            continue

        sims = []
        for i in range(len(gallery)):
            for j in range(i + 1, len(gallery)):
                sims.append(cosine_similarity(gallery[i], gallery[j]))

        avg_self = np.mean(sims)
        min_self = np.min(sims)
        max_self = np.max(sims)
        print(f"  {s['name']:12s}:  avg={avg_self:.4f}  min={min_self:.4f}  max={max_self:.4f}  "
              f"({len(gallery)} samples)")

    # Summary
    print("\n" + "=" * 60)
    if warnings:
        print("  ⚠ WARNINGS:")
        for n1, n2, sim in warnings:
            print(f"    {n1} and {n2} have high similarity ({sim:.4f})")
        print(f"\n  These pairs may cause confusion. Consider:")
        print(f"  • Re-registering with more varied poses/angles")
        print(f"  • Increasing FACE_RECOGNITION_THRESHOLD above {max(w[2] for w in warnings):.2f}")
    else:
        print("  ✓ All pairwise similarities are well below threshold. Looks good!")
    print("=" * 60)


if __name__ == "__main__":
    main()
