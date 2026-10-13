"""
Hand Landmark Normalization Module
Provides scale, translation, and rotation-invariant feature vectors from 21 MediaPipe hand landmarks.
Designed for high biometric repeatability and easy explanation in project vivas.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any


class LandmarkNormalizer:
    """
    Normalizes 21 MediaPipe 3D hand landmarks to ensure distance, position,
    and tilt invariance.
    """

    WRIST_IDX = 0
    MIDDLE_MCP_IDX = 9

    # Landmark indices for finger tips and MCP joints
    FINGERTIP_INDICES = [4, 8, 12, 16, 20]     # Thumb, Index, Middle, Ring, Pinky
    MCP_INDICES = [2, 5, 9, 13, 17]

    @staticmethod
    def normalize_landmarks(landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Transforms raw 21 (x, y, z) landmarks into a normalized 63-element feature vector.

        Steps:
        1. Translation Invariance: Shift coordinates so wrist (idx 0) is at origin (0, 0, 0).
        2. Scale Invariance: Divide all coordinates by the palm size (Euclidean distance
           between wrist and middle finger MCP joint).
        3. Planar Rotation Invariance: Rotate in the XY plane so the palm axis (wrist -> middle MCP)
           aligns vertically along the Y-axis.
        """
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")

        pts = np.array(landmarks, dtype=np.float64)  # Shape: (21, 3)

        # Step 1: Center at wrist
        wrist = pts[LandmarkNormalizer.WRIST_IDX].copy()
        centered = pts - wrist

        # Step 2: Calculate palm scale (wrist to middle MCP)
        middle_mcp = centered[LandmarkNormalizer.MIDDLE_MCP_IDX]
        palm_scale = np.linalg.norm(middle_mcp)

        # Fallback to avoid division by zero
        if palm_scale < 1e-6:
            palm_scale = 1.0

        scaled = centered / palm_scale

        # Step 3: Planar rotation alignment (align wrist->middle_mcp with the -Y axis, pointing UP)
        # In image coordinates, Y points downwards, so pointing UP means angle = -pi/2
        mcp_xy = scaled[LandmarkNormalizer.MIDDLE_MCP_IDX, :2]
        angle = np.arctan2(mcp_xy[1], mcp_xy[0])
        target_angle = -np.pi / 2  # Hand pointing straight up
        rotation_angle = target_angle - angle

        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])

        rotated_xy = np.dot(scaled[:, :2], rotation_matrix.T)
        normalized = np.hstack([rotated_xy, scaled[:, 2:3]])  # Keep normalized z intact

        # Flatten into 1D 63-element vector
        return normalized.flatten()

    @staticmethod
    def extract_hand_geometry(landmarks: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Extracts human-interpretable anatomical hand geometry ratios for viva explanation:
        - Finger-to-palm ratios
        - Inter-fingertip distance ratios
        These ratios are unique morphological signatures of individual hands.
        """
        pts = np.array(landmarks, dtype=np.float64)
        wrist = pts[0]
        middle_mcp = pts[9]
        palm_scale = np.linalg.norm(middle_mcp - wrist)
        if palm_scale < 1e-6:
            palm_scale = 1.0

        def dist(i, j):
            return float(np.linalg.norm(pts[i] - pts[j]) / palm_scale)

        geometry = {
            "thumb_length_ratio": dist(1, 4),
            "index_length_ratio": dist(5, 8),
            "middle_length_ratio": dist(9, 12),
            "ring_length_ratio": dist(13, 16),
            "pinky_length_ratio": dist(17, 20),
            "spread_index_middle": dist(8, 12),
            "spread_middle_ring": dist(12, 16),
            "spread_ring_pinky": dist(16, 20),
            "palm_scale": float(palm_scale)
        }
        return geometry

    @classmethod
    def get_composite_biometric_vector(cls, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Combines the 63 normalized coordinates and 8 morphological ratios
        into a unified 71-element invariant biometric signature.
        """
        normalized_coords = cls.normalize_landmarks(landmarks)
        geom = cls.extract_hand_geometry(landmarks)
        geom_vector = np.array([
            geom["thumb_length_ratio"],
            geom["index_length_ratio"],
            geom["middle_length_ratio"],
            geom["ring_length_ratio"],
            geom["pinky_length_ratio"],
            geom["spread_index_middle"],
            geom["spread_middle_ring"],
            geom["spread_ring_pinky"]
        ], dtype=np.float64)

        # Concatenate and L2 normalize
        composite = np.concatenate([normalized_coords, geom_vector])
        norm = np.linalg.norm(composite)
        if norm > 1e-6:
            composite = composite / norm
        return composite
