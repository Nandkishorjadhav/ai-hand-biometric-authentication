"""
Hand Detection Module
Uses Google MediaPipe HandLandmarker Tasks API with 21 3D landmarks,
extracting hand geometry, bounding boxes, and rendering futuristic HUD skeleton overlays.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetectionResult:
    """Encapsulates hand detection output."""
    def __init__(
        self,
        has_hand: bool,
        landmarks: List[Tuple[float, float, float]] = None,
        pixel_landmarks: List[Tuple[int, int]] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        handedness: str = "Hand"
    ):
        self.has_hand = has_hand
        self.landmarks = landmarks or []              # 21 (x, y, z) in normalized [0, 1] space
        self.pixel_landmarks = pixel_landmarks or []      # 21 (x, y) in pixel space
        self.bbox = bbox                              # (x_min, y_min, x_max, y_max)
        self.handedness = handedness


class HandDetector:
    """
    Real-time hand detector using MediaPipe Tasks HandLandmarker.
    Extracts 21 3D landmarks and renders custom biometric HUD overlays.
    """

    # Hand connections based on standard anatomical topology
    CONNECTIONS = [
        # Thumb ray
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger ray
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger ray
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger ray
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky ray
        (0, 17), (17, 18), (18, 19), (19, 20),
        # ── Palm mesh (knuckle arc + thumb metacarpal ring) ──────────────────
        # Knuckle arc across all MCP joints
        (5, 9), (9, 13), (13, 17),
        # Close the palm polygon: pinky MCP → wrist and thumb CMC arc
        (0, 17), (0, 1), (1, 5),
        # Diagonal cross braces for a filled-mesh look
        (5, 13), (9, 17)
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_hand_detection_confidence: float = 0.55,
        min_hand_presence_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55
    ):
        if model_path is None:
            # Resolve relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "hand_landmarker.task")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker task model not found at {model_path}. "
                "Ensure models/hand_landmarker.task is present."
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def find_hand(self, frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, HandDetectionResult]:
        """
        Processes a BGR image frame and detects the primary hand.
        Returns the annotated frame and HandDetectionResult.
        """
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        annotated_frame = frame.copy() if draw else frame

        if not detection_result.hand_landmarks or len(detection_result.hand_landmarks) == 0:
            return annotated_frame, HandDetectionResult(has_hand=False)

        # Primary hand landmarks
        hand_lms = detection_result.hand_landmarks[0]

        handedness = "Hand"
        if detection_result.handedness and len(detection_result.handedness) > 0:
            handedness = detection_result.handedness[0][0].category_name

        raw_landmarks = []
        pixel_landmarks = []
        x_coords, y_coords = [], []

        for lm in hand_lms:
            raw_landmarks.append((float(lm.x), float(lm.y), float(lm.z)))
            px = int(lm.x * w)
            py = int(lm.y * h)
            pixel_landmarks.append((px, py))
            x_coords.append(px)
            y_coords.append(py)

        # Bounding box with margin
        margin = 25
        x_min = max(0, min(x_coords) - margin)
        y_min = max(0, min(y_coords) - margin)
        x_max = min(w, max(x_coords) + margin)
        y_max = min(h, max(y_coords) + margin)
        bbox = (x_min, y_min, x_max, y_max)

        if draw:
            # 1. Draw glowing biometric bone connections (two-pass: glow + sharp)
            for p1_idx, p2_idx in self.CONNECTIONS:
                pt1 = pixel_landmarks[p1_idx]
                pt2 = pixel_landmarks[p2_idx]
                # Soft glow pass (thicker, semi-transparent cyan)
                cv2.line(annotated_frame, pt1, pt2, (0, 180, 220), 4, cv2.LINE_AA)
                # Sharp crisp pass on top
                cv2.line(annotated_frame, pt1, pt2, (0, 235, 255), 1, cv2.LINE_AA)

            # 2. Draw biometric landmark joints
            for i, (px, py) in enumerate(pixel_landmarks):
                # Fingertips highlighted in neon emerald
                if i in [4, 8, 12, 16, 20]:
                    cv2.circle(annotated_frame, (px, py), 7, (0, 255, 128), -1, cv2.LINE_AA)
                    cv2.circle(annotated_frame, (px, py), 11, (0, 255, 128), 1, cv2.LINE_AA)
                elif i == 0:  # Wrist anchor
                    cv2.circle(annotated_frame, (px, py), 8, (255, 165, 0), -1, cv2.LINE_AA)
                    cv2.circle(annotated_frame, (px, py), 12, (255, 165, 0), 1, cv2.LINE_AA)
                else:
                    cv2.circle(annotated_frame, (px, py), 4, (0, 220, 255), -1, cv2.LINE_AA)

            # 3. Draw sleek biometric bracket bounding box
            bracket_color = (100, 220, 255) if handedness == "Left" else (100, 255, 160)
            self._draw_corner_brackets(annotated_frame, bbox, color=bracket_color, length=22, thickness=2)

            # 4. Draw handedness badge (LEFT / RIGHT) near top of bounding box
            self._draw_handedness_badge(annotated_frame, bbox, handedness)

        detection = HandDetectionResult(
            has_hand=True,
            landmarks=raw_landmarks,
            pixel_landmarks=pixel_landmarks,
            bbox=bbox,
            handedness=handedness
        )

        return annotated_frame, detection

    @staticmethod
    def _draw_corner_brackets(img: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int], length: int = 18, thickness: int = 2):
        """Draws high-tech sci-fi corner brackets around the bounding box."""
        x1, y1, x2, y2 = bbox

        # Top-Left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

        # Top-Right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    @staticmethod
    def _draw_handedness_badge(img: np.ndarray, bbox: Tuple[int, int, int, int], handedness: str):
        """
        Renders a styled LEFT / RIGHT hand badge anchored to the top-left corner
        of the bounding box.  Left hand → cyan badge; Right hand → emerald badge.
        """
        x1, y1, x2, y2 = bbox

        is_left = (handedness.lower() == "left")
        # Label text and directional arrow
        label   = f"< LEFT HAND"  if is_left else f"RIGHT HAND >"
        bg_color  = (30, 30, 30)                      # dark background
        bar_color = (0, 220, 255) if is_left else (0, 255, 140)  # cyan / emerald
        txt_color = (255, 255, 255)

        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.60
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Badge position: just above the bounding box top edge
        pad_x, pad_y = 8, 5
        bx1 = x1
        by1 = max(0, y1 - th - pad_y * 2 - 4)
        bx2 = x1 + tw + pad_x * 2
        by2 = max(th + pad_y * 2, y1 - 4)

        # Semi-transparent dark background
        overlay = img.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_color, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Coloured accent bar on left edge of badge
        cv2.rectangle(img, (bx1, by1), (bx1 + 3, by2), bar_color, -1)

        # Badge border
        cv2.rectangle(img, (bx1, by1), (bx2, by2), bar_color, 1)

        # Text
        tx = bx1 + pad_x + 3
        ty = by2 - pad_y - baseline
        cv2.putText(img, label, (tx, ty), font, font_scale, bar_color, thickness, cv2.LINE_AA)

    def close(self):
        """Releases detector resources."""
        try:
            if hasattr(self, "detector") and self.detector:
                self.detector.close()
        except Exception:
            pass
