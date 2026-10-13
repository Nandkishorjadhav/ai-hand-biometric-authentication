"""
Biometric Feature Matching Module
Calculates similarity between query hand biometric vectors and enrolled templates
using normalized Cosine Similarity.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class BiometricMatcher:
    """
    Handles biometric comparison and template enrollment.
    """

    DEFAULT_THRESHOLD = 0.90  # 90% cosine similarity threshold for positive match

    @staticmethod
    def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes cosine similarity between two feature vectors:
        S = (u . v) / (||u|| * ||v||)
        Returns a float between -1.0 and 1.0.
        """
        v1 = np.asarray(vec1, dtype=np.float64)
        v2 = np.asarray(vec2, dtype=np.float64)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-7 or norm2 < 1e-7:
            return 0.0

        similarity = float(np.dot(v1, v2) / (norm1 * norm2))
        return max(-1.0, min(1.0, similarity))

    @classmethod
    def similarity_to_percentage(cls, similarity: float) -> float:
        """
        Maps cosine similarity (-1.0 to 1.0) into a user-friendly percentage (0.0 to 100.0).
        For normalized positive hand vectors, scores above 0.85 indicate high morphological fidelity.
        """
        clamped = max(0.0, similarity)
        return round(clamped * 100.0, 2)

    @staticmethod
    def create_master_template(sample_vectors: List[np.ndarray]) -> np.ndarray:
        """
        Aggregates multiple captured sample vectors into an ultra-stable master template.
        Averages landmark coordinates across samples to filter out hand tremor and sensor noise.
        """
        if not sample_vectors:
            raise ValueError("No sample vectors provided to create master template")

        stacked = np.array(sample_vectors, dtype=np.float64)
        mean_vector = np.mean(stacked, axis=0)

        norm = np.linalg.norm(mean_vector)
        if norm > 1e-6:
            mean_vector = mean_vector / norm

        return mean_vector

    @classmethod
    def match_hand(
        cls,
        query_vector: np.ndarray,
        enrolled_templates: List[Dict[str, Any]],
        threshold: float = DEFAULT_THRESHOLD
    ) -> Tuple[Optional[str], Optional[str], float, bool]:
        """
        Compares query_vector against all enrolled user templates.

        Parameters:
        - query_vector: 1D normalized biometric vector from the live camera.
        - enrolled_templates: List of dicts, each with keys 'user_id', 'user_name', 'vector'.
        - threshold: Minimum cosine similarity required to grant access.

        Returns:
        - (best_user_id, best_user_name, best_score, is_match)
        """
        if not enrolled_templates:
            return None, None, 0.0, False

        best_user_id = None
        best_user_name = None
        best_similarity = -1.0

        for t in enrolled_templates:
            enrolled_vec = np.array(t["vector"], dtype=np.float64)
            sim = cls.compute_cosine_similarity(query_vector, enrolled_vec)

            if sim > best_similarity:
                best_similarity = sim
                best_user_id = t["user_id"]
                best_user_name = t.get("user_name", "Unknown")

        is_match = best_similarity >= threshold
        score_pct = cls.similarity_to_percentage(best_similarity)

        return best_user_id, best_user_name, score_pct, is_match
