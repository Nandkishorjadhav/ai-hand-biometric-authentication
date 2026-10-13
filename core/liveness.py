"""
Liveness Detection & Gesture Challenge Engine
Detects live human interaction through dynamic randomized gesture challenges
(e.g., Open Palm -> Fist -> Peace Sign) to defeat static photos and video replay attacks.
"""

import time
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Any


class GestureClassifier:
    """
    Rule-based hand gesture classifier using MediaPipe landmark geometry.
    Fast, reliable, and easily explainable in project presentations.
    """

    @staticmethod
    def get_finger_states(landmarks: List[Tuple[float, float, float]]) -> Dict[str, bool]:
        """
        Determines whether each of the 5 fingers is EXTENDED (True) or CURLED (False).
        Uses Euclidean distance from wrist (landmark 0) to avoid camera angle sensitivity.
        """
        pts = np.array(landmarks, dtype=np.float64)
        wrist = pts[0]

        def dist(idx):
            return np.linalg.norm(pts[idx] - wrist)

        # For index, middle, ring, pinky: compare tip distance with PIP joint distance
        index_extended = dist(8) > dist(6) * 1.05
        middle_extended = dist(12) > dist(10) * 1.05
        ring_extended = dist(16) > dist(14) * 1.05
        pinky_extended = dist(20) > dist(18) * 1.05

        # For thumb: compare distance of thumb tip to pinky MCP joint vs thumb IP to pinky MCP
        pinky_mcp = pts[17]
        thumb_tip_dist = np.linalg.norm(pts[4] - pinky_mcp)
        thumb_ip_dist = np.linalg.norm(pts[3] - pinky_mcp)
        thumb_extended = thumb_tip_dist > thumb_ip_dist * 1.1

        return {
            "thumb": bool(thumb_extended),
            "index": bool(index_extended),
            "middle": bool(middle_extended),
            "ring": bool(ring_extended),
            "pinky": bool(pinky_extended),
        }

    @classmethod
    def classify_gesture(cls, landmarks: List[Tuple[float, float, float]]) -> str:
        """
        Classifies the current hand gesture:
        - 'OPEN_PALM': All or 4 main fingers extended
        - 'FIST': All 4 fingers curled
        - 'PEACE_SIGN': Index and Middle extended, Ring and Pinky curled
        - 'THUMBS_UP': Thumb extended upwards, others curled
        - 'THREE_FINGERS': Index, Middle, Ring extended
        - 'UNKNOWN': Ambiguous or transitional hand posture
        """
        states = cls.get_finger_states(landmarks)
        idx, mid, rng, pnk = states["index"], states["middle"], states["ring"], states["pinky"]
        thb = states["thumb"]

        extended_count = sum([idx, mid, rng, pnk])

        if idx and mid and rng and pnk:
            return "OPEN_PALM"
        elif extended_count == 0 and not thb:
            return "FIST"
        elif idx and mid and not rng and not pnk:
            return "PEACE_SIGN"
        elif thb and not idx and not mid and not rng and not pnk:
            return "THUMBS_UP"
        elif idx and mid and rng and not pnk:
            return "THREE_FINGERS"
        elif extended_count >= 3:
            return "OPEN_PALM"
        elif extended_count <= 1 and not thb:
            return "FIST"

        return "UNKNOWN"


class LivenessChallengeSession:
    """
    Manages an active challenge-response session.
    Randomly requests a sequence of distinct gestures that must be held for consecutive frames
    within a timeout window to guarantee physical human liveness.
    """

    AVAILABLE_GESTURES = ["OPEN_PALM", "FIST", "PEACE_SIGN", "THUMBS_UP"]

    GESTURE_CONFIG = {
        "OPEN_PALM": {
            "icon": "✋",
            "title": "SHOW OPEN PALM",
            "action": "Spread all 5 fingers facing camera",
            "voice": "Open your hand and spread your fingers"
        },
        "FIST": {
            "icon": "✊",
            "title": "MAKE A FIST",
            "action": "Close all fingers into a tight fist",
            "voice": "Close your hand into a fist"
        },
        "PEACE_SIGN": {
            "icon": "✌️",
            "title": "SHOW PEACE SIGN",
            "action": "Hold up 2 fingers (Index & Middle) in a V-shape",
            "voice": "Show two fingers in a peace sign"
        },
        "THUMBS_UP": {
            "icon": "👍",
            "title": "SHOW THUMBS UP",
            "action": "Point thumb upward with other fingers closed",
            "voice": "Give a thumbs up"
        }
    }

    # Backward compatibility
    GESTURE_LABELS = {
        k: f"{v['icon']} {v['title']} ({v['action']})" for k, v in GESTURE_CONFIG.items()
    }

    def __init__(self, steps_count: int = 2, step_timeout_seconds: float = 6.0, required_hold_frames: int = 6):
        self.steps_count = steps_count
        self.step_timeout_seconds = step_timeout_seconds
        self.required_hold_frames = required_hold_frames

        # Pick random distinct gestures
        self.challenges = random.sample(self.AVAILABLE_GESTURES, self.steps_count)
        self.current_step_idx = 0
        self.step_start_time = time.time()
        self.hold_frames_count = 0
        self.is_completed = False
        self.is_failed = False
        self.failure_reason = ""

    def reset(self):
        """Restarts a fresh randomized challenge session."""
        self.challenges = random.sample(self.AVAILABLE_GESTURES, self.steps_count)
        self.current_step_idx = 0
        self.step_start_time = time.time()
        self.hold_frames_count = 0
        self.is_completed = False
        self.is_failed = False
        self.failure_reason = ""

    @property
    def current_target(self) -> Optional[str]:
        if self.current_step_idx < len(self.challenges):
            return self.challenges[self.current_step_idx]
        return None

    @property
    def time_remaining(self) -> float:
        return max(0.0, round(self.step_timeout_seconds - (time.time() - self.step_start_time), 1))

    @property
    def current_instruction(self) -> str:
        if self.is_completed:
            return "✅ LIVENESS VERIFIED! Matching biometric identity..."
        if self.is_failed:
            return f"❌ LIVENESS FAILED: {self.failure_reason}"
        target = self.current_target
        cfg = self.GESTURE_CONFIG.get(target, {"icon": "🖐️", "title": target, "action": ""})
        step_num = self.current_step_idx + 1
        total_steps = len(self.challenges)
        return f"Step {step_num} of {total_steps}: {cfg['icon']} {cfg['title']} ({cfg['action']}) - {self.time_remaining}s left"

    def update(self, detected_gesture: str) -> Dict[str, Any]:
        """
        Processes a single frame's detected gesture against the active challenge step.
        Returns the session state and rich instructions.
        """
        now = time.time()
        time_left = self.time_remaining

        if self.is_completed:
            return {
                "status": "PASSED",
                "step": self.current_step_idx,
                "instruction": self.current_instruction,
                "time_left": 0.0,
                "progress": 1.0
            }

        if self.is_failed:
            return {
                "status": "FAILED",
                "reason": self.failure_reason,
                "instruction": self.current_instruction,
                "time_left": 0.0,
                "progress": 0.0
            }

        # Check for timeout
        elapsed = now - self.step_start_time
        if elapsed > self.step_timeout_seconds:
            self.is_failed = True
            self.failure_reason = f"Timeout: Failed to perform challenge within {self.step_timeout_seconds:.0f}s"
            return {
                "status": "FAILED",
                "reason": self.failure_reason,
                "instruction": self.current_instruction,
                "time_left": 0.0,
                "progress": 0.0
            }

        target = self.current_target
        cfg = self.GESTURE_CONFIG.get(target, {"icon": "🖐️", "title": target, "action": ""})
        step_num = self.current_step_idx + 1
        total_steps = len(self.challenges)

        is_matching = (detected_gesture == target)

        if is_matching:
            self.hold_frames_count += 1
            if self.hold_frames_count >= self.required_hold_frames:
                # Step passed!
                self.current_step_idx += 1
                self.hold_frames_count = 0
                self.step_start_time = now

                if self.current_step_idx >= len(self.challenges):
                    self.is_completed = True
                    return {
                        "status": "PASSED",
                        "step": self.current_step_idx,
                        "instruction": "✅ LIVENESS VERIFIED! Matching biometric identity...",
                        "time_left": 0.0,
                        "progress": 1.0
                    }
                else:
                    next_target = self.current_target
                    next_cfg = self.GESTURE_CONFIG.get(next_target, {"icon": "🖐️", "title": next_target, "action": ""})
                    return {
                        "status": "STEP_PASSED",
                        "current_step": self.current_step_idx + 1,
                        "total_steps": total_steps,
                        "target": next_target,
                        "target_title": next_cfg["title"],
                        "target_icon": next_cfg["icon"],
                        "target_action": next_cfg["action"],
                        "detected": detected_gesture,
                        "progress": 0.0,
                        "time_left": self.step_timeout_seconds,
                        "instruction": f"Great! Step {self.current_step_idx + 1} of {total_steps}: {next_cfg['icon']} {next_cfg['title']} ({next_cfg['action']})"
                    }
        else:
            # Gesture not matching, decay hold frames count
            self.hold_frames_count = max(0, self.hold_frames_count - 1)

        progress = min(1.0, self.hold_frames_count / float(self.required_hold_frames))
        return {
            "status": "IN_PROGRESS",
            "current_step": step_num,
            "total_steps": total_steps,
            "target": target,
            "target_title": cfg["title"],
            "target_icon": cfg["icon"],
            "target_action": cfg["action"],
            "detected": detected_gesture,
            "is_gesture_matching": is_matching,
            "progress": progress,
            "hold_count": self.hold_frames_count,
            "required_hold": self.required_hold_frames,
            "time_left": time_left,
            "instruction": f"Step {step_num} of {total_steps}: {cfg['icon']} {cfg['title']} - {cfg['action']}"
        }

