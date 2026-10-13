"""
Automated API tests for FastAPI application endpoints.
"""

import base64
import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import app, db


class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

        # Create a dummy blank JPEG image for testing endpoints
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank_img)
        self.dummy_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    def test_dashboard_page(self):
        """Test GET / returns HTML dashboard with 200 OK."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Hand Biometric Authentication", response.text)

    def test_preview_detection_endpoint(self):
        """Test POST /api/detect-preview with blank image."""
        response = self.client.post("/api/detect-preview", json={"image": self.dummy_base64})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("has_hand", data)
        self.assertFalse(data["has_hand"])
        self.assertEqual(data["gesture"], "NO HAND")

    def test_enrollment_and_auth_flow(self):
        """Test full enrollment and retrieval API flow."""
        # Enroll test user
        dummy_vectors = [[float(i * 0.01) for i in range(71)] for _ in range(3)]
        enroll_res = self.client.post("/api/enroll-complete", json={
            "name": "API Test User",
            "email": "test@college.edu",
            "vectors": dummy_vectors
        })
        self.assertEqual(enroll_res.status_code, 200)
        enroll_data = enroll_res.json()
        self.assertEqual(enroll_data["status"], "SUCCESS")
        user_id = enroll_data["user_id"]

        # List users
        users_res = self.client.get("/api/users")
        self.assertEqual(users_res.status_code, 200)
        users = users_res.json()
        self.assertTrue(any(u["id"] == user_id for u in users))

        # Start auth session
        auth_start_res = self.client.post("/api/auth-start")
        self.assertEqual(auth_start_res.status_code, 200)

        # Cancel auth session
        cancel_res = self.client.post("/api/auth-cancel")
        self.assertEqual(cancel_res.status_code, 200)

        # Check logs endpoint
        logs_res = self.client.get("/api/logs")
        self.assertEqual(logs_res.status_code, 200)

        # Cleanup: delete test user
        del_res = self.client.delete(f"/api/users/{user_id}")
        self.assertEqual(del_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
