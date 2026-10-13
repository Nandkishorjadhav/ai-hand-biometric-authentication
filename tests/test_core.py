"""
Automated unit tests for core vision, normalization, liveness, matching, and database modules.
"""

import os
import unittest
import numpy as np

from core.normalizer import LandmarkNormalizer
from core.liveness import GestureClassifier, LivenessChallengeSession
from core.matcher import BiometricMatcher
from core.database import Database
from core.hand_detector import HandDetector


class TestCoreModules(unittest.TestCase):

    def setUp(self):
        # Generate synthetic 21 hand landmarks for testing
        # Wrist at (0.5, 0.8, 0.0), Middle MCP at (0.5, 0.5, 0.0)
        self.dummy_landmarks = [(0.5, 0.8, 0.0)]  # 0: wrist
        for i in range(1, 21):
            self.dummy_landmarks.append((0.5 + 0.01 * i, 0.8 - 0.03 * i, 0.001 * i))

    def test_landmark_normalization_invariance(self):
        """Test that translation shift does NOT alter normalized vector."""
        vec1 = LandmarkNormalizer.normalize_landmarks(self.dummy_landmarks)

        # Shift all landmarks by (+0.25, -0.35, +0.10)
        shifted_landmarks = [(x + 0.25, y - 0.35, z + 0.10) for (x, y, z) in self.dummy_landmarks]
        vec2 = LandmarkNormalizer.normalize_landmarks(shifted_landmarks)

        # Vectors should be identical within numerical precision
        diff = np.max(np.abs(vec1 - vec2))
        self.assertLess(diff, 1e-4, f"Translation invariance failed, max diff: {diff}")

    def test_composite_biometric_vector(self):
        """Test composite biometric vector shape and unit L2 norm."""
        comp = LandmarkNormalizer.get_composite_biometric_vector(self.dummy_landmarks)
        self.assertEqual(len(comp), 71)
        norm = np.linalg.norm(comp)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_gesture_classifier_open_palm(self):
        """Test gesture classification for an open palm structure."""
        # Open palm where finger tips are far from wrist
        open_hand = [(0.5, 0.9, 0.0)] * 21
        # Set wrist at (0.5, 0.9, 0.0)
        open_hand[0] = (0.5, 0.9, 0.0)
        # Set tips (4, 8, 12, 16, 20) far away
        for tip in [4, 8, 12, 16, 20]:
            open_hand[tip] = (0.5, 0.1, 0.0)
        # Set PIP joints closer
        for pip in [3, 6, 10, 14, 18]:
            open_hand[pip] = (0.5, 0.6, 0.0)
        # Set pinky MCP
        open_hand[17] = (0.6, 0.7, 0.0)

        gesture = GestureClassifier.classify_gesture(open_hand)
        self.assertEqual(gesture, "OPEN_PALM")

    def test_liveness_challenge_session(self):
        """Test challenge-response session progression."""
        session = LivenessChallengeSession(steps_count=2, required_hold_frames=2)
        self.assertEqual(session.current_step_idx, 0)
        target1 = session.current_target
        self.assertIn(target1, session.AVAILABLE_GESTURES)

        # Send target gesture twice to pass step 1
        res1 = session.update(target1)
        res2 = session.update(target1)
        self.assertEqual(session.current_step_idx, 1)

        # Send step 2 target
        target2 = session.current_target
        session.update(target2)
        res_final = session.update(target2)
        self.assertEqual(res_final["status"], "PASSED")
        self.assertTrue(session.is_completed)

    def test_biometric_matcher_cosine_similarity(self):
        """Test cosine similarity calculation and matching."""
        v1 = LandmarkNormalizer.get_composite_biometric_vector(self.dummy_landmarks)
        # Slightly jittered version
        jittered = [(x + 0.002, y - 0.001, z) for (x, y, z) in self.dummy_landmarks]
        v2 = LandmarkNormalizer.get_composite_biometric_vector(jittered)

        sim = BiometricMatcher.compute_cosine_similarity(v1, v2)
        self.assertGreater(sim, 0.95)

        enrolled = [{"user_id": 1, "user_name": "Test User", "vector": v1}]
        user_id, name, score_pct, is_match = BiometricMatcher.match_hand(v2, enrolled, threshold=0.90)
        self.assertEqual(user_id, 1)
        self.assertEqual(name, "Test User")
        self.assertTrue(is_match)
        self.assertGreater(score_pct, 95.0)

    def test_sqlite_database_operations(self):
        """Test database enrollment, retrieval, and audit logging."""
        test_db_path = "test_hand_auth.db"
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

        db = Database(db_path=test_db_path)
        vec = LandmarkNormalizer.get_composite_biometric_vector(self.dummy_landmarks)
        user_id = db.add_user("Alice", "alice@example.com", vec, sample_count=5)
        self.assertEqual(user_id, 1)

        templates = db.get_all_templates()
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0]["user_name"], "Alice")

        # Test auth log
        log_id = db.log_auth_attempt(user_id, "Alice", "PASSED", 98.5, "ACCESS GRANTED", "Identity match verified")
        self.assertEqual(log_id, 1)

        logs = db.get_recent_logs(limit=10)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["access_result"], "ACCESS GRANTED")

        # Cleanup
        if os.path.exists(test_db_path):
            os.remove(test_db_path)


if __name__ == "__main__":
    unittest.main()
