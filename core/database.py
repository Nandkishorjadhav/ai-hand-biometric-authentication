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
                threshold REAL DEFAULT 0.90,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """)

            # Migration: add threshold column if missing (for existing databases)
            try:
                cursor.execute("ALTER TABLE biometric_templates ADD COLUMN threshold REAL DEFAULT 0.90;")
            except Exception:
                pass  # Column already exists

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

    def add_user(self, name: str, email: str, vector: np.ndarray, sample_count: int = 1, threshold: float = 0.90) -> int:
        """Enrolls a new user with their biometric feature vector and per-user threshold."""
        features_json = json.dumps(vector.tolist())
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email) VALUES (?, ?);", (name.strip(), email.strip()))
            user_id = cursor.lastrowid

            cursor.execute("""
            INSERT INTO biometric_templates (user_id, features_json, sample_count, threshold)
            VALUES (?, ?, ?, ?);
            """, (user_id, features_json, sample_count, threshold))
            conn.commit()
            return user_id
        finally:
            conn.close()

    def get_all_templates(self) -> List[Dict[str, Any]]:
        """Retrieves all registered biometric templates for fast vector matching."""
        rows = self._execute("""
        SELECT u.id as user_id, u.name as user_name, t.features_json, t.threshold
        FROM users u
        JOIN biometric_templates t ON u.id = t.user_id;
        """, fetch="all")

        templates = []
        for r in rows:
            templates.append({
                "user_id": r["user_id"],
                "user_name": r["user_name"],
                "vector": np.array(json.loads(r["features_json"]), dtype=np.float64),
                "threshold": r["threshold"] if r["threshold"] is not None else 0.90
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

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Returns aggregate metrics used by the operations dashboard."""
        summary = self._execute("""
        SELECT
            COUNT(*) AS total_attempts,
            COALESCE(SUM(CASE WHEN access_result = 'ACCESS GRANTED' THEN 1 ELSE 0 END), 0) AS granted_attempts,
            COALESCE(AVG(similarity_score), 0) AS average_similarity,
            COALESCE(SUM(CASE WHEN date(timestamp) = date('now', 'localtime') THEN 1 ELSE 0 END), 0) AS today_attempts
        FROM auth_logs;
        """, fetch="one")

        trend_rows = self._execute("""
        SELECT
            date(timestamp, 'localtime') AS day,
            COUNT(*) AS attempts,
            COALESCE(SUM(CASE WHEN access_result = 'ACCESS GRANTED' THEN 1 ELSE 0 END), 0) AS granted
        FROM auth_logs
        WHERE timestamp >= datetime('now', 'localtime', '-6 days')
        GROUP BY date(timestamp, 'localtime')
        ORDER BY day ASC;
        """, fetch="all")

        total_attempts = int(summary["total_attempts"] or 0)
        granted_attempts = int(summary["granted_attempts"] or 0)
        return {
            "total_users": int(self._execute("SELECT COUNT(*) AS count FROM users;", fetch="one")["count"]),
            "total_attempts": total_attempts,
            "granted_attempts": granted_attempts,
            "denied_attempts": total_attempts - granted_attempts,
            "success_rate": round((granted_attempts / total_attempts) * 100, 1) if total_attempts else 0,
            "average_similarity": round(float(summary["average_similarity"] or 0), 1),
            "today_attempts": int(summary["today_attempts"] or 0),
            "trend": [dict(row) for row in trend_rows],
            "recent_logs": self.get_recent_logs(limit=5)
        }

    def export_all_data(self) -> Dict[str, Any]:
        """Exports all database data for CSV/JSON download."""
        users = self.get_all_users()
        logs = self.get_recent_logs(limit=10000)
        return {
            "enrolled_users": users,
            "total_users": len(users),
            "auth_logs": logs,
            "total_logs": len(logs)
        }
