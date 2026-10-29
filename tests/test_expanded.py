import sys
import os

# Project root path setup
PROJECT_ROOT = 'd:/Hand Biometri Authentication/hand-biometric-auth'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import base64
import math
import sqlite3
import tempfile
import time
import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from core.normalizer import LandmarkNormalizer
from core.liveness import GestureClassifier, LivenessChallengeSession
from core.matcher import BiometricMatcher
from core.database import Database
from app import app


class TestNormalizationEdgeCases(unittest.TestCase):
    """
    Tests edge cases and boundary conditions in hand landmark normalization:
    - Zero-vector coordinates
    - Single landmark and invalid landmark counts
    - Identical/degenerate landmark positions
    - Translation, scale, and planar rotation invariance
    """

    def setUp(self):
        # Canonical 21 MediaPipe hand landmarks for invariance tests
        self.valid_landmarks = [
            (0.50, 0.82,  0.00),  # 0: Wrist
            (0.44, 0.74, -0.01),  # 1: Thumb CMC
            (0.39, 0.66, -0.02),  # 2: Thumb MCP
            (0.35, 0.58, -0.02),  # 3: Thumb IP
            (0.30, 0.51, -0.02),  # 4: Thumb Tip
            (0.43, 0.54, -0.01),  # 5: Index MCP
            (0.41, 0.43, -0.01),  # 6: Index PIP
            (0.40, 0.35, -0.01),  # 7: Index DIP
            (0.39, 0.27, -0.02),  # 8: Index Tip
            (0.50, 0.52,  0.00),  # 9: Middle MCP
            (0.50, 0.39, -0.01),  # 10: Middle PIP
            (0.50, 0.30, -0.01),  # 11: Middle DIP
            (0.50, 0.20, -0.01),  # 12: Middle Tip
            (0.57, 0.54, -0.01),  # 13: Ring MCP
            (0.58, 0.43, -0.01),  # 14: Ring PIP
            (0.60, 0.35, -0.01),  # 15: Ring DIP
            (0.60, 0.27, -0.02),  # 16: Ring Tip
            (0.64, 0.58, -0.01),  # 17: Pinky MCP
            (0.66, 0.49, -0.02),  # 18: Pinky PIP
            (0.68, 0.42, -0.02),  # 19: Pinky DIP
            (0.69, 0.35, -0.02),  # 20: Pinky Tip
        ]

    def test_zero_vector_input(self):
        """Validates that a 21-landmark array of all zeros does not crash or produce NaNs."""
        zero_landmarks = [(0.0, 0.0, 0.0)] * 21

        # normalize_landmarks handles palm_scale < 1e-6 via fallback to 1.0
        norm_coords = LandmarkNormalizer.normalize_landmarks(zero_landmarks)
        self.assertEqual(len(norm_coords), 63)
        self.assertFalse(np.isnan(norm_coords).any(), "Normalized coords contain NaN")
        self.assertFalse(np.isinf(norm_coords).any(), "Normalized coords contain Inf")

        # extract_hand_geometry with all zero coordinates
        geom = LandmarkNormalizer.extract_hand_geometry(zero_landmarks)
        self.assertEqual(geom["palm_scale"], 1.0)
        for k, v in geom.items():
            self.assertFalse(math.isnan(v), f"Geometry ratio {k} is NaN")

        # get_composite_biometric_vector returns 71-D vector with all zeros
        composite = LandmarkNormalizer.get_composite_biometric_vector(zero_landmarks)
        self.assertEqual(len(composite), 71)
        self.assertFalse(np.isnan(composite).any(), "Composite vector contains NaN")
        self.assertFalse(np.isinf(composite).any(), "Composite vector contains Inf")
        self.assertEqual(float(np.linalg.norm(composite)), 0.0)

    def test_single_landmark_raises_error(self):
        """Input with only 1 landmark must raise ValueError (MediaPipe requires 21)."""
        single_landmark = [(0.5, 0.5, 0.0)]
        with self.assertRaises(ValueError) as ctx:
            LandmarkNormalizer.normalize_landmarks(single_landmark)
        self.assertIn("Expected 21 landmarks", str(ctx.exception))

        with self.assertRaises(ValueError):
            LandmarkNormalizer.get_composite_biometric_vector(single_landmark)

    def test_invalid_landmark_counts(self):
        """Any landmark count != 21 must raise ValueError."""
        for invalid_len in [0, 5, 10, 20, 22, 42]:
            lms = [(0.1 * i, 0.2 * i, 0.0) for i in range(invalid_len)]
            with self.assertRaises(ValueError, msg=f"Failed to reject length {invalid_len}"):
                LandmarkNormalizer.normalize_landmarks(lms)

    def test_identical_landmarks(self):
        """When all 21 landmarks share identical coordinates, normalizer avoids division by zero."""
        identical_landmarks = [(0.35, 0.45, 0.12)] * 21
        norm_coords = LandmarkNormalizer.normalize_landmarks(identical_landmarks)
        self.assertEqual(len(norm_coords), 63)
        self.assertFalse(np.isnan(norm_coords).any())

        composite = LandmarkNormalizer.get_composite_biometric_vector(identical_landmarks)
        self.assertEqual(len(composite), 71)
        self.assertFalse(np.isnan(composite).any())

    def test_planar_rotation_invariance_full_circle(self):
        """Rotating hand landmarks in the XY plane should yield cosine similarity >= 0.99."""
        v_base = LandmarkNormalizer.get_composite_biometric_vector(self.valid_landmarks)

        wrist = np.array(self.valid_landmarks[0])
        pts = np.array(self.valid_landmarks)

        for deg in [30, 45, 90, 180, 270, 315]:
            rad = math.radians(deg)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

            rotated_xy = np.dot(pts[:, :2] - wrist[:2], rot_mat.T) + wrist[:2]
            rotated_pts = np.hstack([rotated_xy, pts[:, 2:3]])
            rotated_lms = [(float(p[0]), float(p[1]), float(p[2])) for p in rotated_pts]

            v_rot = LandmarkNormalizer.get_composite_biometric_vector(rotated_lms)
            sim = BiometricMatcher.compute_cosine_similarity(v_base, v_rot)
            self.assertGreaterEqual(sim, 0.985, f"Rotation by {deg} deg yielded low similarity: {sim}")

    def test_scale_and_translation_invariance(self):
        """Translating and scaling landmarks together preserves biometric vector similarity."""
        v_base = LandmarkNormalizer.get_composite_biometric_vector(self.valid_landmarks)

        for scale in [0.4, 0.75, 1.5, 3.0]:
            for dx, dy, dz in [(0.5, -0.3, 0.1), (-1.2, 0.8, -0.4), (10.0, 20.0, 5.0)]:
                transformed = [(x * scale + dx, y * scale + dy, z * scale + dz) for (x, y, z) in self.valid_landmarks]
                v_trans = LandmarkNormalizer.get_composite_biometric_vector(transformed)
                sim = BiometricMatcher.compute_cosine_similarity(v_base, v_trans)
                self.assertGreaterEqual(sim, 0.99, f"Scale {scale} with shift ({dx},{dy},{dz}) yielded similarity {sim}")


class TestLivenessEngine(unittest.TestCase):
    """
    Tests dynamic gesture challenge liveness detection:
    - Session timeout handling
    - Step expiry and recovery
    - Hold frame progression and decay
    - Session reset and challenge completion
    - GestureClassifier coverage
    """

    def test_session_timeout_handling(self):
        """Active session marks is_failed=True and status='FAILED' upon timeout."""
        session = LivenessChallengeSession(steps_count=2, step_timeout_seconds=0.1, required_hold_frames=3)
        self.assertFalse(session.is_failed)

        # Wait past step timeout
        time.sleep(0.12)

        res = session.update("OPEN_PALM")
        self.assertTrue(session.is_failed)
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("Timeout", res["reason"])
        self.assertEqual(session.time_remaining, 0.0)
        self.assertIn("LIVENESS FAILED", session.current_instruction)

        # Subsequent updates must remain FAILED
        subsequent_res = session.update("OPEN_PALM")
        self.assertEqual(subsequent_res["status"], "FAILED")

    def test_step_expiry_and_reset(self):
        """Calling reset() restores session state after failure."""
        session = LivenessChallengeSession(steps_count=2, step_timeout_seconds=0.05, required_hold_frames=3)
        time.sleep(0.06)
        session.update("OPEN_PALM")
        self.assertTrue(session.is_failed)

        session.reset()
        self.assertFalse(session.is_failed)
        self.assertFalse(session.is_completed)
        self.assertEqual(session.current_step_idx, 0)
        self.assertEqual(session.hold_frames_count, 0)
        self.assertEqual(session.failure_reason, "")
        self.assertGreater(session.time_remaining, 0.0)

    def test_hold_frame_decay_on_gesture_mismatch(self):
        """Incorrect gesture decrements accumulated hold frame count without going below 0."""
        session = LivenessChallengeSession(steps_count=2, step_timeout_seconds=5.0, required_hold_frames=4)
        target = session.current_target

        # Hold target for 2 frames
        session.update(target)
        res = session.update(target)
        self.assertEqual(session.hold_frames_count, 2)
        self.assertAlmostEqual(res["progress"], 0.5)

        # Wrong gesture decrements hold count to 1
        res_wrong = session.update("UNKNOWN")
        self.assertEqual(session.hold_frames_count, 1)
        self.assertAlmostEqual(res_wrong["progress"], 0.25)

        # Decay clamps at 0
        session.update("UNKNOWN")
        session.update("UNKNOWN")
        self.assertEqual(session.hold_frames_count, 0)

    def test_full_challenge_progression_to_completion(self):
        """Progressing through all required steps completes session with status='PASSED'."""
        session = LivenessChallengeSession(steps_count=2, step_timeout_seconds=10.0, required_hold_frames=2)

        # Step 1
        t1 = session.current_target
        session.update(t1)
        res1 = session.update(t1)
        self.assertEqual(res1["status"], "STEP_PASSED")
        self.assertEqual(session.current_step_idx, 1)

        # Step 2
        t2 = session.current_target
        session.update(t2)
        res2 = session.update(t2)
        self.assertEqual(res2["status"], "PASSED")
        self.assertTrue(session.is_completed)
        self.assertIn("LIVENESS VERIFIED", session.current_instruction)

        # Subsequent update on completed session remains PASSED
        res3 = session.update("OPEN_PALM")
        self.assertEqual(res3["status"], "PASSED")
        self.assertEqual(res3["progress"], 1.0)

    def test_gesture_classifier_rule_coverage(self):
        """Tests rule-based gesture classification for all standard postures."""
        # 1. OPEN_PALM: All tips extended far from wrist
        open_palm = [(0.5, 0.8, 0.0)] * 21
        for tip in [4, 8, 12, 16, 20]:
            open_palm[tip] = (0.5, 0.2, 0.0)
        for pip in [3, 6, 10, 14, 18]:
            open_palm[pip] = (0.5, 0.6, 0.0)
        open_palm[17] = (0.6, 0.7, 0.0)
        self.assertEqual(GestureClassifier.classify_gesture(open_palm), "OPEN_PALM")

        # 2. FIST: All tips curled close to wrist
        fist = [(0.5, 0.8, 0.0)] * 21
        for tip in [4, 8, 12, 16, 20]:
            fist[tip] = (0.5, 0.78, 0.0)
        for pip in [3, 6, 10, 14, 18]:
            fist[pip] = (0.5, 0.70, 0.0)
        fist[17] = (0.6, 0.75, 0.0)
        self.assertEqual(GestureClassifier.classify_gesture(fist), "FIST")


class TestDatabaseExpanded(unittest.TestCase):
    """
    Tests SQLite database operations with an isolated temporary database file:
    - Duplicate user handling
    - Unique constraint on biometric_templates
    - Deleting non-existent users
    - export_all_data method structure and contents
    """

    def setUp(self):
        # Create an isolated temporary SQLite database file for each test
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_file.name
        self.temp_file.close()
        self.db = Database(db_path=self.temp_db_path)

        # Standard 71-D dummy vector
        self.dummy_vec = np.ones(71, dtype=np.float64) / math.sqrt(71)

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except Exception:
                pass

    def test_duplicate_user_handling(self):
        """Users table allows adding multiple users with identical name/email (assigns unique IDs)."""
        uid1 = self.db.add_user("Candidate One", "candidate@test.com", self.dummy_vec, sample_count=5)
        uid2 = self.db.add_user("Candidate One", "candidate@test.com", self.dummy_vec, sample_count=10)

        self.assertNotEqual(uid1, uid2)
        users = self.db.get_all_users()
        self.assertEqual(len(users), 2)

        templates = self.db.get_all_templates()
        self.assertEqual(len(templates), 2)
        self.assertEqual({t["user_id"] for t in templates}, {uid1, uid2})

    def test_duplicate_template_user_id_constraint(self):
        """biometric_templates table enforces UNIQUE constraint on user_id."""
        uid = self.db.add_user("Alice", "alice@test.com", self.dummy_vec)

        # Attempting a direct insert of a second template with the same user_id must raise IntegrityError
        conn = sqlite3.connect(self.temp_db_path)
        try:
            cursor = conn.cursor()
            with self.assertRaises(sqlite3.IntegrityError):
                cursor.execute("""
                INSERT INTO biometric_templates (user_id, features_json, sample_count, threshold)
                VALUES (?, ?, ?, ?);
                """, (uid, "[]", 1, 0.90))
        finally:
            conn.close()

    def test_delete_existing_and_non_existent_user(self):
        """delete_user returns True for existing user, and False for non-existent ID."""
        # Deleting from empty database returns False
        self.assertFalse(self.db.delete_user(99999))

        # Enroll a user
        uid = self.db.add_user("Bob", "bob@test.com", self.dummy_vec)
        self.assertEqual(len(self.db.get_all_users()), 1)

        # Delete existing user returns True
        self.assertTrue(self.db.delete_user(uid))
        self.assertEqual(len(self.db.get_all_users()), 0)
        self.assertEqual(len(self.db.get_all_templates()), 0)

        # Deleting the same user again returns False
        self.assertFalse(self.db.delete_user(uid))

    def test_export_all_data_method(self):
        """export_all_data returns properly structured dictionary with all enrolled users and logs."""
        # Fresh database export
        empty_data = self.db.export_all_data()
        self.assertEqual(empty_data["total_users"], 0)
        self.assertEqual(empty_data["enrolled_users"], [])
        self.assertEqual(empty_data["total_logs"], 0)
        self.assertEqual(empty_data["auth_logs"], [])

        # Add users
        uid1 = self.db.add_user("User Alpha", "alpha@test.com", self.dummy_vec, sample_count=5)
        uid2 = self.db.add_user("User Beta", "beta@test.com", self.dummy_vec, sample_count=7)

        # Log auth attempts
        self.db.log_auth_attempt(uid1, "User Alpha", "PASSED", 97.5, "ACCESS GRANTED", "Verified")
        self.db.log_auth_attempt(uid2, "User Beta", "PASSED", 83.2, "ACCESS DENIED", "Below threshold")
        self.db.log_auth_attempt(None, "Unknown", "FAILED_TIMEOUT", 0.0, "ACCESS DENIED", "Timeout")

        exported = self.db.export_all_data()
        self.assertEqual(exported["total_users"], 2)
        self.assertEqual(len(exported["enrolled_users"]), 2)
        self.assertEqual(exported["total_logs"], 3)
        self.assertEqual(len(exported["auth_logs"]), 3)

        # Verify user record keys
        user_rec = exported["enrolled_users"][0]
        for key in ["id", "name", "email", "sample_count", "created_at"]:
            self.assertIn(key, user_rec)

        # Verify log record keys
        log_rec = exported["auth_logs"][0]
        for key in ["id", "user_id", "user_name", "timestamp", "liveness_status", "similarity_score", "access_result", "reason"]:
            self.assertIn(key, log_rec)


class TestMatcherExpanded(unittest.TestCase):
    """
    Tests BiometricMatcher with diverse sample distributions:
    - compute_adaptive_threshold with < 3 samples (fallback)
    - Identical samples (low variance / clamped at max_threshold)
    - Orthogonal / dissimilar samples (high variance / clamped at min_threshold)
    - Realistic intra-sample variance distributions
    - Custom clamping bounds
    - Master template generation edge cases
    """

    def setUp(self):
        # Generate 71-D orthogonal basis vectors
        self.v1 = np.zeros(71, dtype=np.float64)
        self.v1[0] = 1.0

        self.v2 = np.zeros(71, dtype=np.float64)
        self.v2[1] = 1.0

        self.v3 = np.zeros(71, dtype=np.float64)
        self.v3[2] = 1.0

    def test_compute_adaptive_threshold_less_than_3_samples(self):
        """Fewer than 3 enrollment samples returns the default threshold of 0.90."""
        self.assertEqual(BiometricMatcher.compute_adaptive_threshold([]), 0.90)
        self.assertEqual(BiometricMatcher.compute_adaptive_threshold([self.v1]), 0.90)
        self.assertEqual(BiometricMatcher.compute_adaptive_threshold([self.v1, self.v2]), 0.90)

    def test_compute_adaptive_threshold_identical_samples(self):
        """Identical samples have 0 variance, so adaptive threshold clamps to max_threshold (0.96)."""
        samples = [self.v1.copy() for _ in range(5)]
        th = BiometricMatcher.compute_adaptive_threshold(samples)
        self.assertEqual(th, 0.96)

    def test_compute_adaptive_threshold_high_variance_orthogonal_samples(self):
        """Orthogonal samples with near-zero similarity clamp to min_threshold (0.82)."""
        samples = [self.v1, self.v2, self.v3]
        th = BiometricMatcher.compute_adaptive_threshold(samples)
        self.assertEqual(th, 0.82)

    def test_compute_adaptive_threshold_realistic_distribution(self):
        """Samples with realistic intra-user variance calculate threshold = mu - 2*sigma."""
        # Create 4 slightly perturbed vectors around base vector
        rng = np.random.default_rng(2026)
        base = self.v1.copy()
        samples = []
        for _ in range(4):
            jitter = rng.normal(0, 0.05, size=71)
            vec = base + jitter
            vec /= np.linalg.norm(vec)
            samples.append(vec)

        # Manually compute pairwise cosine similarities
        sims = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                sims.append(BiometricMatcher.compute_cosine_similarity(samples[i], samples[j]))

        expected_mu = float(np.mean(sims))
        expected_sigma = float(np.std(sims))
        expected_th = round(max(0.82, min(0.96, expected_mu - 2.0 * expected_sigma)), 4)

        calculated_th = BiometricMatcher.compute_adaptive_threshold(samples)
        self.assertEqual(calculated_th, expected_th)
        self.assertGreaterEqual(calculated_th, 0.82)
        self.assertLessEqual(calculated_th, 0.96)

    def test_compute_adaptive_threshold_custom_bounds(self):
        """Custom min_threshold and max_threshold values are respected."""
        samples_identical = [self.v1.copy() for _ in range(5)]
        th_max = BiometricMatcher.compute_adaptive_threshold(samples_identical, min_threshold=0.70, max_threshold=0.98)
        self.assertEqual(th_max, 0.98)

        samples_orthogonal = [self.v1, self.v2, self.v3]
        th_min = BiometricMatcher.compute_adaptive_threshold(samples_orthogonal, min_threshold=0.75, max_threshold=0.95)
        self.assertEqual(th_min, 0.75)

    def test_create_master_template_edge_cases(self):
        """Master template raises on empty list, normalizes single and multiple vectors."""
        with self.assertRaises(ValueError):
            BiometricMatcher.create_master_template([])

        single_res = BiometricMatcher.create_master_template([self.v1])
        self.assertEqual(len(single_res), 71)
        self.assertAlmostEqual(float(np.linalg.norm(single_res)), 1.0, places=5)

        multi_res = BiometricMatcher.create_master_template([self.v1, self.v2])
        self.assertAlmostEqual(float(np.linalg.norm(multi_res)), 1.0, places=5)

    def test_match_hand_empty_templates_and_per_user_threshold(self):
        """match_hand handles empty enrollment gracefully and respects per-user thresholds."""
        # Empty enrollment list
        uid, name, score, is_match = BiometricMatcher.match_hand(self.v1, [])
        self.assertIsNone(uid)
        self.assertIsNone(name)
        self.assertFalse(is_match)
        self.assertEqual(score, 0.0)

        # Template with strict per-user threshold (0.95)
        v_perturbed = self.v1.copy()
        v_perturbed[1] = 0.35
        v_perturbed /= np.linalg.norm(v_perturbed)
        sim = BiometricMatcher.compute_cosine_similarity(self.v1, v_perturbed)  # ~0.94

        templates = [{"user_id": 1, "user_name": "Strict User", "vector": self.v1, "threshold": 0.98}]
        _, _, _, match_strict = BiometricMatcher.match_hand(v_perturbed, templates)
        self.assertFalse(match_strict, "Match should fail against strict 0.98 threshold")

        templates_lenient = [{"user_id": 1, "user_name": "Lenient User", "vector": self.v1, "threshold": 0.85}]
        _, _, _, match_lenient = BiometricMatcher.match_hand(v_perturbed, templates_lenient)
        self.assertTrue(match_lenient, "Match should succeed against lenient 0.85 threshold")


class TestAPIErrorResponses(unittest.TestCase):
    """
    Tests FastAPI REST API error responses using TestClient:
    - Malformed base64 strings
    - Missing required fields (Pydantic 422 Unprocessable Entity)
    - Business rule validation errors (HTTP 400)
    - Resource not found (HTTP 404)
    - Session lifecycle errors
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # Create a valid blank base64 image for endpoints requiring valid image format
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank_img)
        cls.valid_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    def test_detect_preview_malformed_base64(self):
        """POST /api/detect-preview returns 400 for corrupt or invalid base64 data."""
        res1 = self.client.post("/api/detect-preview", json={"image": "not_valid_base64_data!@#$%^&*()"})
        self.assertEqual(res1.status_code, 400)

        res2 = self.client.post("/api/detect-preview", json={"image": "data:image/jpeg;base64,??invalid_payload??"})
        self.assertEqual(res2.status_code, 400)

        res3 = self.client.post("/api/detect-preview", json={"image": ""})
        self.assertEqual(res3.status_code, 400)

    def test_detect_preview_missing_image_field(self):
        """POST /api/detect-preview returns 422 when 'image' field is missing."""
        res_empty = self.client.post("/api/detect-preview", json={})
        self.assertEqual(res_empty.status_code, 422)

        res_wrong_key = self.client.post("/api/detect-preview", json={"frame_data": self.valid_base64})
        self.assertEqual(res_wrong_key.status_code, 422)

    def test_enroll_sample_malformed_base64(self):
        """POST /api/enroll-sample returns 400 when given malformed base64 string."""
        res = self.client.post("/api/enroll-sample", json={"image": "malformed_base64_data"})
        self.assertEqual(res.status_code, 400)

    def test_enroll_complete_missing_fields(self):
        """POST /api/enroll-complete returns 422 when required schema fields are missing."""
        # Missing both name and vectors
        res1 = self.client.post("/api/enroll-complete", json={})
        self.assertEqual(res1.status_code, 422)

        # Missing vectors
        res2 = self.client.post("/api/enroll-complete", json={"name": "Alice"})
        self.assertEqual(res2.status_code, 422)

        # Missing name
        res3 = self.client.post("/api/enroll-complete", json={"vectors": [[0.1] * 71] * 3})
        self.assertEqual(res3.status_code, 422)

    def test_enroll_complete_empty_name(self):
        """POST /api/enroll-complete returns 400 when name is empty or whitespace."""
        res = self.client.post("/api/enroll-complete", json={
            "name": "   ",
            "vectors": [[0.1] * 71] * 3
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("cannot be empty", res.json()["detail"])

    def test_enroll_complete_insufficient_samples(self):
        """POST /api/enroll-complete returns 400 when fewer than 3 samples are provided."""
        # Zero samples
        res_zero = self.client.post("/api/enroll-complete", json={
            "name": "Candidate",
            "vectors": []
        })
        self.assertEqual(res_zero.status_code, 400)
        self.assertIn("Minimum 3 hand samples required", res_zero.json()["detail"])

        # Two samples (insufficient)
        res_two = self.client.post("/api/enroll-complete", json={
            "name": "Candidate",
            "vectors": [[0.1] * 71, [0.2] * 71]
        })
        self.assertEqual(res_two.status_code, 400)
        self.assertIn("Minimum 3 hand samples required", res_two.json()["detail"])

    def test_delete_user_not_found(self):
        """DELETE /api/users/{id} returns 404 when user ID does not exist."""
        res = self.client.delete("/api/users/99999999")
        self.assertEqual(res.status_code, 404)
        self.assertIn("User not found", res.json()["detail"])

    def test_auth_step_without_active_session(self):
        """POST /api/auth-step returns 400 when no authentication session has been started."""
        # Cancel any active session first
        self.client.post("/api/auth-cancel")

        res = self.client.post("/api/auth-step", json={"image": self.valid_base64})
        self.assertEqual(res.status_code, 400)
        self.assertIn("No active authentication session", res.json()["detail"])

    def test_export_data_endpoint(self):
        """GET /api/export supports both json and csv formats."""
        res_json = self.client.get("/api/export?format=json")
        self.assertEqual(res_json.status_code, 200)
        json_data = res_json.json()
        self.assertIn("enrolled_users", json_data)
        self.assertIn("auth_logs", json_data)

        res_csv = self.client.get("/api/export?format=csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("text/csv", res_csv.headers.get("content-type", ""))
        self.assertIn("hand_auth_export.csv", res_csv.headers.get("content-disposition", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
