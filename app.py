"""
FastAPI Web Application Server
Exposes REST APIs for real-time video frame processing, liveness challenge orchestration,
biometric template enrollment, and audit logs.
"""

import base64
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.hand_detector import HandDetector
from core.normalizer import LandmarkNormalizer
from core.liveness import GestureClassifier, LivenessChallengeSession
from core.matcher import BiometricMatcher
from core.database import Database

# Initialize FastAPI application
app = FastAPI(
    title="Hand Biometric Authentication System",
    description="Contactless hand biometric verification with anti-spoofing liveness detection",
    version="1.0.0"
)

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "web", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "web", "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Core singletons
db = Database(os.path.join(BASE_DIR, "hand_auth.db"))
detector = HandDetector()

# In-memory authentication session state
auth_session: Optional[LivenessChallengeSession] = None

# Accumulated biometric vectors sampled during active authentication session.
# Collecting across ALL frames (not just the final one) makes the averaged
# query template far more stable and harder to spoof by mimicry.
auth_biometric_frames: List[np.ndarray] = []

# Minimum frames with a detected hand required before a biometric decision is made.
# Enforces that the person held their hand steadily through the session.
MIN_BIOMETRIC_FRAMES = 5

# Stricter similarity threshold for authentication (higher than enrollment default).
# Raising to 93% makes it significantly harder to impersonate another user.
AUTH_SIMILARITY_THRESHOLD = 0.93


# --- Pydantic Request Models ---
class FramePayload(BaseModel):
    image: str  # Base64 data URL


class EnrollCompletePayload(BaseModel):
    name: str
    email: Optional[str] = ""
    vectors: List[List[float]]


# --- Helper Utilities ---
def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decodes a base64 JPEG/PNG string into an OpenCV BGR numpy array."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode base64 image data")
    return frame


def encode_image_base64(img: np.ndarray) -> str:
    """Encodes an OpenCV image to a base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')


# --- Web Page Route ---
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the single-page biometric authentication dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


# --- API Routes ---
@app.post("/api/detect-preview")
async def detect_preview(payload: FramePayload):
    """
    Real-time frame processing for idle preview mode.
    Detects hand, renders skeleton overlay, and returns gesture tag.
    """
    try:
        frame = decode_base64_image(payload.image)
        annotated_frame, detection = detector.find_hand(frame, draw=True)

        gesture = "NO HAND"
        if detection.has_hand:
            gesture = GestureClassifier.classify_gesture(detection.landmarks)

        return {
            "has_hand": detection.has_hand,
            "gesture": gesture,
            "handedness": detection.handedness,
            "annotated_image": encode_image_base64(annotated_frame)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth-start")
async def start_auth_session():
    """Initializes a new randomized challenge-response liveness session."""
    global auth_session, auth_biometric_frames
    enrolled = db.get_all_templates()
    if not enrolled:
        raise HTTPException(
            status_code=400,
            detail="No users are enrolled in the database yet. Please enroll a user first!"
        )

    auth_session = LivenessChallengeSession(steps_count=2, step_timeout_seconds=6.0, required_hold_frames=5)
    auth_biometric_frames = []  # Reset frame buffer for new session
    return {"status": "STARTED", "instruction": auth_session.current_instruction}


@app.post("/api/auth-cancel")
async def cancel_auth_session():
    """Cancels the active authentication session."""
    global auth_session, auth_biometric_frames
    auth_session = None
    auth_biometric_frames = []
    return {"status": "CANCELLED"}


@app.post("/api/auth-step")
async def process_auth_step(payload: FramePayload):
    """
    Processes an active authentication video frame:
    1. Continuously accumulates biometric vectors from every frame where a hand is detected.
    2. Checks dynamic gesture challenge for liveness.
    3. If liveness PASSES, builds a noise-reduced averaged query template from all
       accumulated frames and matches against the database.
    4. Requires MIN_BIOMETRIC_FRAMES samples and AUTH_SIMILARITY_THRESHOLD to grant access.
       This prevents impersonation via gesture mimicry because a multi-frame average
       demands consistent hand morphology throughout the entire session, not just one frame.
    """
    global auth_session, auth_biometric_frames
    if auth_session is None:
        raise HTTPException(status_code=400, detail="No active authentication session. Call /api/auth-start first.")

    frame = decode_base64_image(payload.image)
    annotated_frame, detection = detector.find_hand(frame, draw=True)

    gesture = "NO HAND"
    if detection.has_hand:
        gesture = GestureClassifier.classify_gesture(detection.landmarks)
        # ── CONTINUOUS BIOMETRIC ACCUMULATION ──────────────────────────────────
        # Collect a biometric vector on EVERY frame with a detected hand.
        # This produces a stable, noise-averaged template matching enrollment quality.
        vec = LandmarkNormalizer.get_composite_biometric_vector(detection.landmarks)
        auth_biometric_frames.append(vec)
        # ───────────────────────────────────────────────────────────────────────

    step_res = auth_session.update(gesture)
    decision = "STANDBY"
    candidate_name = "None"
    similarity_score = 0.0
    reason = ""

    time_left = step_res.get("time_left", auth_session.time_remaining)

    # Render a high-visibility instruction HUD banner directly onto the video frame
    h, w, _ = annotated_frame.shape
    if step_res["status"] in ["IN_PROGRESS", "STEP_PASSED"]:
        # Top banner background
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 85), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)
        cv2.rectangle(annotated_frame, (10, 10), (w - 10, 85), (56, 189, 248), 2)

        title_text = f"STEP {step_res.get('current_step', 1)}/{step_res.get('total_steps', 2)}: {step_res.get('target_title', '')}"
        action_text = f"ACTION: {step_res.get('target_action', '')}"
        timer_text = f"TIMER: {time_left:.1f}s"

        cv2.putText(annotated_frame, title_text, (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, action_text, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, timer_text, (w - 180, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (245, 158, 11), 2, cv2.LINE_AA)

        # Draw hold progress bar inside video frame
        hold_pct = step_res.get("progress", 0.0)
        p_w = int((w - 50) * hold_pct)
        cv2.rectangle(annotated_frame, (25, 78), (25 + p_w, 82), (16, 185, 129), -1)

    if step_res["status"] == "FAILED":
        decision = "ACCESS DENIED"
        reason = step_res.get("reason", "Liveness check failed")
        db.log_auth_attempt(None, "Unknown", "FAILED_TIMEOUT", 0.0, "ACCESS DENIED", reason)
        auth_session = None
        auth_biometric_frames = []

    elif step_res["status"] == "PASSED":
        # ── MULTI-FRAME BIOMETRIC IDENTITY VERIFICATION ────────────────────────
        # We need enough frames to build a stable averaged template.
        if len(auth_biometric_frames) >= MIN_BIOMETRIC_FRAMES:
            # Average all accumulated frames → noise-reduced query template
            # (mirrors the enrollment process which also averages multiple captures)
            stacked = np.array(auth_biometric_frames, dtype=np.float64)
            query_vec = np.mean(stacked, axis=0)
            norm = np.linalg.norm(query_vec)
            if norm > 1e-6:
                query_vec = query_vec / norm

            enrolled = db.get_all_templates()
            user_id, match_name, sim_pct, is_match = BiometricMatcher.match_hand(
                query_vec, enrolled, threshold=AUTH_SIMILARITY_THRESHOLD
            )

            similarity_score = sim_pct
            candidate_name = match_name or "Unknown"
            frames_used = len(auth_biometric_frames)

            if is_match:
                decision = "ACCESS GRANTED"
                reason = f"Identity verified across {frames_used} biometric frames ({similarity_score}% match)"
                db.log_auth_attempt(user_id, candidate_name, "PASSED", similarity_score, "ACCESS GRANTED", reason)
            else:
                decision = "ACCESS DENIED"
                reason = f"Biometric mismatch — similarity {similarity_score}% is below {int(AUTH_SIMILARITY_THRESHOLD*100)}% threshold"
                db.log_auth_attempt(user_id, candidate_name, "PASSED", similarity_score, "ACCESS DENIED", reason)
        else:
            # Not enough frames collected — hand was not consistently present
            decision = "ACCESS DENIED"
            reason = f"Insufficient biometric data: only {len(auth_biometric_frames)} frames captured (minimum {MIN_BIOMETRIC_FRAMES} required)"
            db.log_auth_attempt(None, "Unknown", "PASSED", 0.0, "ACCESS DENIED", reason)
        # ───────────────────────────────────────────────────────────────────────

        auth_session = None
        auth_biometric_frames = []

    return {
        "annotated_image": encode_image_base64(annotated_frame),
        "has_hand": detection.has_hand,
        "gesture": gesture,
        "handedness": detection.handedness,
        "liveness_status": step_res.get("status", "IN_PROGRESS"),
        "instruction": step_res.get("instruction", ""),
        "target_title": step_res.get("target_title", ""),
        "target_icon": step_res.get("target_icon", "🖐️"),
        "target_action": step_res.get("target_action", ""),
        "current_step": step_res.get("current_step", 1),
        "total_steps": step_res.get("total_steps", 2),
        "progress": step_res.get("progress", 0.0),
        "is_matching": step_res.get("is_gesture_matching", False),
        "time_left": time_left,
        "decision": decision,
        "candidate_name": candidate_name,
        "similarity_score": similarity_score,
        "reason": reason
    }



@app.post("/api/enroll-sample")
async def enroll_sample(payload: FramePayload):
    """
    Validates and extracts a single high-quality hand sample during enrollment.
    Requires an OPEN PALM gesture to ensure consistent landmark topology.
    """
    frame = decode_base64_image(payload.image)
    annotated_frame, detection = detector.find_hand(frame, draw=True)

    if not detection.has_hand:
        return {
            "sample_accepted": False,
            "message": "No hand detected",
            "annotated_image": encode_image_base64(annotated_frame)
        }

    gesture = GestureClassifier.classify_gesture(detection.landmarks)
    if gesture != "OPEN_PALM":
        return {
            "sample_accepted": False,
            "message": "Please show an OPEN PALM with fingers spread",
            "annotated_image": encode_image_base64(annotated_frame)
        }

    vec = LandmarkNormalizer.get_composite_biometric_vector(detection.landmarks)
    return {
        "sample_accepted": True,
        "vector": vec.tolist(),
        "annotated_image": encode_image_base64(annotated_frame)
    }


@app.post("/api/enroll-complete")
async def enroll_complete(payload: EnrollCompletePayload):
    """
    Aggregates collected sample vectors into an ultra-stable master template
    and stores user record into SQLite.
    """
    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="User name cannot be empty")

    if not payload.vectors or len(payload.vectors) < 3:
        raise HTTPException(status_code=400, detail="Minimum 3 hand samples required for enrollment")

    sample_vecs = [np.array(v, dtype=np.float64) for v in payload.vectors]
    master_vec = BiometricMatcher.create_master_template(sample_vecs)

    user_id = db.add_user(
        name=payload.name.strip(),
        email=payload.email.strip() if payload.email else f"{payload.name.lower().replace(' ', '')}@college.edu",
        vector=master_vec,
        sample_count=len(payload.vectors)
    )

    return {
        "status": "SUCCESS",
        "user_id": user_id,
        "name": payload.name,
        "samples_enrolled": len(payload.vectors)
    }


@app.get("/api/users")
async def list_users():
    """Lists all registered biometric users."""
    return db.get_all_users()


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int):
    """Deletes a user and their biometric template."""
    success = db.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "DELETED", "user_id": user_id}


@app.get("/api/logs")
async def get_logs(limit: int = 50):
    """Retrieves recent authentication attempts for the audit table."""
    return db.get_recent_logs(limit=limit)


if __name__ == "__main__":
    import uvicorn
    print("\n[START] Hand Biometric Authentication Web Server -> http://localhost:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
