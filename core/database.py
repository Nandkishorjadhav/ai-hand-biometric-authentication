"""
SQLite Database Layer
Stores registered users, biometric feature templates, and authentication audit logs
using Python's built-in sqlite3. Zero setup, zero configuration.
"""

import json
import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class Database:
    """
    Manages SQLite database storage for users, biometric templates, and audit logs.
    """

    def __init__(self, db_path: str = "hand_auth.db"):
        self.db_path = db_path
        self._init_db()

    def _execute(self, query: str, params: tuple = (), fetch: str = "none") -> Any:
        """Executes a query and ensures connection is cleanly closed."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if fetch == "all":
                result = cursor.fetchall()
            elif fetch == "one":
                result = cursor.fetchone()
            elif fetch == "lastrowid":
                conn.commit()
                result = cursor.lastrowid
            elif fetch == "rowcount":
                conn.commit()
                result = cursor.rowcount
            else:
                conn.commit()
                result = None
            return result
        finally:
            conn.close()

    def _init_db(self):
        """Initializes database tables if they do not exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS biometric_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                features_json TEXT NOT NULL,
                sample_count INTEGER DEFAULT 1,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                user_name TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                liveness_status TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                access_result TEXT NOT NULL,
                reason TEXT
            );
            """)
            conn.commit()
        finally:
            conn.close()

    def add_user(self, name: str, email: str, vector: np.ndarray, sample_count: int = 1) -> int:
        """Enrolls a new user with their biometric feature vector."""
        features_json = json.dumps(vector.tolist())
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email) VALUES (?, ?);", (name.strip(), email.strip()))
            user_id = cursor.lastrowid

            cursor.execute("""
            INSERT INTO biometric_templates (user_id, features_json, sample_count)
            VALUES (?, ?, ?);
            """, (user_id, features_json, sample_count))
            conn.commit()
            return user_id
        finally:
            conn.close()

    def get_all_templates(self) -> List[Dict[str, Any]]:
        """Retrieves all registered biometric templates for fast vector matching."""
        rows = self._execute("""
        SELECT u.id as user_id, u.name as user_name, t.features_json
        FROM users u
        JOIN biometric_templates t ON u.id = t.user_id;
        """, fetch="all")

        templates = []
        for r in rows:
            templates.append({
                "user_id": r["user_id"],
                "user_name": r["user_name"],
                "vector": np.array(json.loads(r["features_json"]), dtype=np.float64)
            })
        return templates

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Lists all enrolled users."""
        rows = self._execute("""
        SELECT u.id, u.name, u.email, u.created_at, t.sample_count, t.updated_at
        FROM users u
        LEFT JOIN biometric_templates t ON u.id = t.user_id
        ORDER BY u.id DESC;
        """, fetch="all")
        return [dict(r) for r in rows]

    def delete_user(self, user_id: int) -> bool:
        """Removes a user and their biometric template."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?;", (user_id,))
            cursor.execute("DELETE FROM biometric_templates WHERE user_id = ?;", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def log_auth_attempt(
        self,
        user_id: Optional[int],
        user_name: Optional[str],
        liveness_status: str,
        similarity_score: float,
        access_result: str,
        reason: str
    ) -> int:
        """Records an authentication audit log entry."""
        return self._execute("""
        INSERT INTO auth_logs (user_id, user_name, liveness_status, similarity_score, access_result, reason)
        VALUES (?, ?, ?, ?, ?, ?);
        """, (
            user_id,
            user_name or "Unknown",
            liveness_status,
            float(similarity_score),
            access_result,
            reason
        ), fetch="lastrowid")

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves the most recent authentication attempts for audit inspection."""
        rows = self._execute("""
        SELECT id, user_id, user_name, timestamp, liveness_status, similarity_score, access_result, reason
        FROM auth_logs
        ORDER BY id DESC
        LIMIT ?;
        """, (limit,), fetch="all")
        return [dict(r) for r in rows]

