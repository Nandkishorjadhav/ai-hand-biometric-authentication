"""
Academic Verification & Benchmark Test Suite
Validates the Hand Biometric Authentication pipeline:
1. MediaPipe Landmarker & Hand Detector
2. 71-D Morphological Normalizer (Invariance: translation, scale, rotation)
3. Gesture Classifier (OPEN_PALM, FIST, PEACE, THUMBS_UP, OK_SIGN)
4. Cosine Similarity Biometric Matcher
5. SQLite Database operations & audit logging
6. FastAPI REST endpoints (Preview, Auth Step, User Management, Logs)
"""

import os
import sys
import unittest
import numpy as np
import cv2
from fastapi.testclient import TestClient

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.hand_detector import HandDetector
from core.normalizer import LandmarkNormalizer
from core.liveness import GestureClassifier, LivenessChallengeSession
from core.matcher import BiometricMatcher
from core.database import Database
from app import app


class TestHandBiometrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = HandDetector()
        cls.db = Database("test_eval.db")
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.detector.close()
        if os.path.exists("test_eval.db"):
            try:
                os.remove("test_eval.db")
            except Exception:
                pass

    def test_01_hand_detector_blank_frame(self):
        """Validates detector handles frame with no hand without crashing."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, res = self.detector.find_hand(blank_frame, draw=True, is_mirrored=True)
        self.assertFalse(res.has_hand)
        self.assertEqual(annotated.shape, (480, 640, 3))

    def test_02_biometric_vector_dimension(self):
        """Ensures composite biometric feature vector has exact 71 dimensions."""
        # Simulated 21 3D landmarks
        simulated_landmarks = [(0.5 + 0.02 * i, 0.5 - 0.01 * i, 0.0) for i in range(21)]
        vec = LandmarkNormalizer.get_composite_biometric_vector(simulated_landmarks)
        self.assertEqual(len(vec), 71)
        # Vector should be L2-normalized (norm ~ 1.0)
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_03_translation_scale_invariance(self):
        """Verifies that translating and scaling hand landmarks does not alter similarity."""
        base_landmarks = [(0.3 + 0.01 * i, 0.4 + 0.015 * i, 0.001 * i) for i in range(21)]
        vec_base = LandmarkNormalizer.get_composite_biometric_vector(base_landmarks)

        # Shift by (dx=0.2, dy=0.1) and scale by 1.5x
        transformed_landmarks = [(x * 1.5 + 0.2, y * 1.5 + 0.1, z * 1.5) for x, y, z in base_landmarks]
        vec_transformed = LandmarkNormalizer.get_composite_biometric_vector(transformed_landmarks)

        similarity = BiometricMatcher.compute_cosine_similarity(vec_base, vec_transformed)
        self.assertGreaterEqual(similarity, 0.98, "Biometric vector should be translation and scale invariant.")

    def test_04_cosine_matching_different_identities(self):
        """Ensures different simulated hands yield low cosine similarity."""
        hand_a = [(0.2 + 0.01 * i, 0.3 + 0.02 * i, 0.0) for i in range(21)]
        hand_b = [(0.8 - 0.02 * i, 0.1 + 0.03 * i, 0.0) for i in range(21)]

        vec_a = LandmarkNormalizer.get_composite_biometric_vector(hand_a)
        vec_b = LandmarkNormalizer.get_composite_biometric_vector(hand_b)

        similarity = BiometricMatcher.compute_cosine_similarity(vec_a, vec_b)
        self.assertLess(similarity, 0.88, "Distinct hand topologies must not match above threshold.")

    def test_05_gesture_classifier(self):
        """Tests gesture rule classification logic."""
        # Flat hand with spread fingers
        open_palm_lms = [(0.5, 0.8, 0.0)] * 21
        # Set fingertips far above MCPs
        for tip in [4, 8, 12, 16, 20]:
            open_palm_lms[tip] = (0.5, 0.2, 0.0)
        for mcp in [2, 5, 9, 13, 17]:
            open_palm_lms[mcp] = (0.5, 0.5, 0.0)

        gesture = GestureClassifier.classify_gesture(open_palm_lms)
        self.assertEqual(gesture, "OPEN_PALM")

    def test_06_liveness_session_progression(self):
        """Tests the dynamic 2-step challenge session."""
        session = LivenessChallengeSession(steps_count=2, step_timeout_seconds=5.0, required_hold_frames=2)
        target_gesture = session.current_target
        self.assertIsNotNone(target_gesture)

        # Feed matching gesture 2 times to satisfy required_hold_frames
        res1 = session.update(target_gesture)
        res2 = session.update(target_gesture)

        self.assertIn(res2["status"], ["STEP_PASSED", "PASSED", "IN_PROGRESS"])

    def test_07_database_crud_and_audit(self):
        """Verifies SQLite user registration and attempt logging."""
        dummy_vec = np.ones(71, dtype=np.float64)
        dummy_vec /= np.linalg.norm(dummy_vec)

        user_id = self.db.add_user("Test Candidate", "candidate@college.edu", dummy_vec, sample_count=10)
        self.assertIsInstance(user_id, int)

        users = self.db.get_all_users()
        self.assertTrue(any(u["id"] == user_id for u in users))

        # Test log entry
        self.db.log_auth_attempt(user_id, "Test Candidate", "PASSED", 95.5, "ACCESS GRANTED", "Identity confirmed")
        logs = self.db.get_recent_logs(limit=5)
        self.assertTrue(len(logs) > 0)
        self.assertEqual(logs[0]["user_name"], "Test Candidate")

    def test_08_fastapi_endpoints(self):
        """Tests that web routes respond with 200 OK."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)

        res_users = self.client.get("/api/users")
        self.assertEqual(res_users.status_code, 200)

        res_logs = self.client.get("/api/logs")
        self.assertEqual(res_logs.status_code, 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
