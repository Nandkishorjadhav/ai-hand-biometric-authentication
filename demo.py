"""
Standalone 1-Click Desktop OpenCV Demo
Biometric Hand Authentication with Real-time Liveness Detection
Runs completely offline using OpenCV, MediaPipe Tasks, and SQLite.
No web browser required - perfect for college vivas and classroom projectors.
"""

import cv2
import time
import numpy as np
from core.hand_detector import HandDetector
from core.normalizer import LandmarkNormalizer
from core.liveness import GestureClassifier, LivenessChallengeSession
from core.matcher import BiometricMatcher
from core.database import Database


def draw_hud_header(frame, title: str, status_text: str = "ONLINE", status_color=(0, 255, 128)):
    """Draws top status bar with college project branding."""
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.line(frame, (0, 55), (w, 55), (56, 189, 248), 1)

    # Title
    cv2.putText(frame, title, (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Status indicator
    cv2.circle(frame, (w - 110, 30), 6, status_color, -1, cv2.LINE_AA)
    cv2.putText(frame, status_text, (w - 95, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)


def draw_hud_footer(frame, keys_help: str):
    """Draws bottom banner with hotkey controls."""
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 45), (w, h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.line(frame, (0, h - 45), (w, h - 45), (56, 189, 248), 1)
    cv2.putText(frame, keys_help, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 240), 1, cv2.LINE_AA)


def draw_banner_box(frame, message: str, subtext: str = "", color=(0, 220, 255), y_pos=130):
    """Draws centered glowing message card."""
    h, w, _ = frame.shape
    box_w = 640
    box_h = 90
    x1 = (w - box_w) // 2
    y1 = y_pos
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Text
    cv2.putText(frame, message, (x1 + 25, y1 + 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if subtext:
        cv2.putText(frame, subtext, (x1 + 25, y1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main():
    print("=" * 65)
    print("🖐️  HAND BIOMETRIC AUTHENTICATION SYSTEM (DESKTOP DEMO)")
    print("=" * 65)
    print("Initializing Camera and MediaPipe Hand Landmarker...")

    db = Database("hand_auth.db")
    detector = HandDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam. Make sure camera is connected.")
        return

    # Set desired camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    mode = "IDLE"  # "IDLE", "REGISTER", "AUTH", "RESULT"
    reg_name = ""
    reg_samples = []
    reg_target_count = 10

    liveness_session = None
    result_banner = {"title": "", "sub": "", "color": (0, 255, 0), "expire": 0.0}

    print("\nControls:")
    print(" [R] Register New Hand")
    print(" [A] Authenticate (Liveness + Biometric Match)")
    print(" [L] Show Enrolled Users & Recent Logs")
    print(" [Q] Quit Demo")
    print("=" * 65)

    window_name = "AI Hand Biometric Authentication"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break

        # Mirror frame horizontally for natural user experience
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Run hand detection & skeleton drawing
        annotated_frame, detection = detector.find_hand(frame, draw=True)

        current_gesture = "NO_HAND"
        if detection.has_hand:
            current_gesture = GestureClassifier.classify_gesture(detection.landmarks)

            # Display real-time detected gesture tag near wrist
            if detection.pixel_landmarks:
                wx, wy = detection.pixel_landmarks[0]
                tag = f"{detection.handedness}: {current_gesture}"
                cv2.putText(annotated_frame, tag, (wx - 40, wy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)

        now = time.time()

        # Handle different demo states
        if mode == "IDLE":
            draw_hud_header(annotated_frame, "HAND BIOMETRIC AUTHENTICATION // SYSTEM READY", "READY", (0, 255, 128))
            draw_hud_footer(annotated_frame, "HOTKEYS:  [R] Register User   |   [A] Authenticate   |   [L] List Users   |   [Q] Quit")

        elif mode == "REGISTER":
            draw_hud_header(annotated_frame, f"REGISTRATION // ENROLLING: {reg_name.upper()}", "CAPTURING", (0, 220, 255))
            draw_hud_footer(annotated_frame, "Keep hand open and steady inside the frame... [ESC to Cancel]")

            if detection.has_hand and current_gesture == "OPEN_PALM":
                # Extract biometric vector
                vec = LandmarkNormalizer.get_composite_biometric_vector(detection.landmarks)
                reg_samples.append(vec)

            collected = len(reg_samples)
            progress_msg = f"Collecting Biometric Samples: {collected}/{reg_target_count}"
            sub_msg = "Please hold hand steady with fingers spread (Open Palm)"
            draw_banner_box(annotated_frame, progress_msg, sub_msg, color=(0, 220, 255), y_pos=70)

            # Progress bar
            bar_w = 400
            bar_h = 10
            bx = (w - bar_w) // 2
            by = 175
            cv2.rectangle(annotated_frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
            fill_w = int((collected / float(reg_target_count)) * bar_w)
            cv2.rectangle(annotated_frame, (bx, by), (bx + fill_w, by + bar_h), (0, 255, 128), -1)

            if collected >= reg_target_count:
                # Build master template and save to SQLite
                master_vec = BiometricMatcher.create_master_template(reg_samples)
                user_id = db.add_user(reg_name, f"{reg_name.lower()}@college.edu", master_vec, sample_count=collected)

                print(f"\n✅ User '{reg_name}' enrolled successfully! (ID: {user_id}, Samples: {collected})")
                mode = "RESULT"
                result_banner = {
                    "title": f"ENROLLMENT SUCCESSFUL: {reg_name.upper()}",
                    "sub": f"Biometric template generated from {collected} hand frames and saved to database.",
                    "color": (0, 255, 128),
                    "expire": now + 3.0
                }

        elif mode == "AUTH":
            draw_hud_header(annotated_frame, "AUTHENTICATION // LIVENESS + BIOMETRIC MATCH", "VERIFYING", (56, 189, 248))
            draw_hud_footer(annotated_frame, "Follow on-screen gesture challenge prompts carefully [ESC to Cancel]")

            if liveness_session is not None:
                step_res = liveness_session.update(current_gesture)

                if step_res["status"] == "IN_PROGRESS":
                    target_title = step_res.get("target_title", step_res.get("target", ""))
                    target_action = step_res.get("target_action", "")
                    time_left = step_res.get("time_left", 6.0)

                    prompt_title = f"CHALLENGE {step_res['current_step']}/{step_res['total_steps']}: {target_title} [{time_left:.1f}s left]"
                    prompt_sub = f"Action: {target_action} | Detected: {current_gesture}"
                    draw_banner_box(annotated_frame, prompt_title, prompt_sub, color=(0, 220, 255), y_pos=70)


                    # Draw hold progress bar
                    hold_prog = step_res["progress"]
                    bar_w = 400
                    bar_h = 10
                    bx = (w - bar_w) // 2
                    by = 175
                    cv2.rectangle(annotated_frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
                    fill_w = int(hold_prog * bar_w)
                    cv2.rectangle(annotated_frame, (bx, by), (bx + fill_w, by + bar_h), (0, 255, 128), -1)

                elif step_res["status"] == "FAILED":
                    # Liveness failed (e.g. timeout or spoof suspected)
                    db.log_auth_attempt(None, "Unknown", "FAILED_TIMEOUT", 0.0, "ACCESS DENIED", step_res["reason"])
                    mode = "RESULT"
                    result_banner = {
                        "title": "ACCESS DENIED // SPOOF DETECTED",
                        "sub": f"Reason: {step_res['reason']}",
                        "color": (0, 0, 255),
                        "expire": now + 3.5
                    }

                elif step_res["status"] == "PASSED":
                    # Liveness PASS! Now perform biometric comparison
                    if detection.has_hand:
                        query_vec = LandmarkNormalizer.get_composite_biometric_vector(detection.landmarks)
                        enrolled = db.get_all_templates()

                        user_id, user_name, score_pct, is_match = BiometricMatcher.match_hand(
                            query_vec, enrolled, threshold=0.88
                        )

                        if is_match:
                            db.log_auth_attempt(user_id, user_name, "PASSED", score_pct, "ACCESS GRANTED", "Identity and liveness verified")
                            result_banner = {
                                "title": f"ACCESS GRANTED // {user_name.upper()}",
                                "sub": f"Liveness: PASSED (100%) | Biometric Match Score: {score_pct:.1f}%",
                                "color": (0, 255, 128),
                                "expire": now + 4.0
                            }
                        else:
                            candidate = user_name if user_name else "Unregistered"
                            db.log_auth_attempt(user_id, candidate, "PASSED", score_pct, "ACCESS DENIED", f"Low biometric similarity ({score_pct:.1f}%)")
                            result_banner = {
                                "title": "ACCESS DENIED // UNRECOGNIZED USER",
                                "sub": f"Liveness: PASSED | Best Similarity: {score_pct:.1f}% (Below 88% threshold)",
                                "color": (0, 0, 255),
                                "expire": now + 3.5
                            }
                    else:
                        result_banner = {
                            "title": "ACCESS DENIED // HAND LOST",
                            "sub": "Hand moved out of frame before biometric embedding capture.",
                            "color": (0, 0, 255),
                            "expire": now + 3.0
                        }

                    mode = "RESULT"

        elif mode == "RESULT":
            draw_hud_header(annotated_frame, "AUTHENTICATION RESULT", "COMPLETED", result_banner["color"])
            draw_hud_footer(annotated_frame, "Returning to monitoring in a moment...")

            draw_banner_box(
                annotated_frame,
                result_banner["title"],
                result_banner["sub"],
                color=result_banner["color"],
                y_pos=90
            )

            if now >= result_banner["expire"]:
                mode = "IDLE"

        # Show frame
        cv2.imshow(window_name, annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        elif key == 27:  # ESC to cancel current operation
            mode = "IDLE"
            print("\nOperation cancelled.")

        elif key == ord('r') or key == ord('R'):
            if mode == "IDLE":
                name_input = input("\nEnter User Full Name for enrollment: ").strip()
                if name_input:
                    reg_name = name_input
                    reg_samples = []
                    mode = "REGISTER"
                    print(f"Enrolling '{reg_name}'. Hold your open hand in front of the camera...")
                else:
                    print("Registration cancelled (empty name).")

        elif key == ord('a') or key == ord('A'):
            if mode == "IDLE":
                templates = db.get_all_templates()
                if not templates:
                    print("\n⚠️ No users enrolled yet! Press [R] to register a user first.")
                    mode = "RESULT"
                    result_banner = {
                        "title": "DATABASE EMPTY // NO USERS ENROLLED",
                        "sub": "Please press [R] to register at least one user before authenticating.",
                        "color": (0, 165, 255),
                        "expire": now + 3.0
                    }
                else:
                    liveness_session = LivenessChallengeSession(steps_count=2, step_timeout_seconds=6.0, required_hold_frames=6)
                    mode = "AUTH"
                    print("\nStarting Liveness Challenge & Biometric Verification...")

        elif key == ord('l') or key == ord('L'):
            users = db.get_all_users()
            logs = db.get_recent_logs(limit=5)
            print("\n" + "=" * 60)
            print("📋 ENROLLED USERS:")
            for u in users:
                print(f" - ID: {u['id']} | Name: {u['name']} | Enrolled At: {u['created_at']}")
            if not users:
                print(" (None)")
            print("\n📜 RECENT AUTHENTICATION ATTEMPTS:")
            for l in logs:
                print(f" - [{l['timestamp']}] {l['user_name']}: {l['access_result']} (Match: {l['similarity_score']}%, Liveness: {l['liveness_status']})")
            if not logs:
                print(" (No attempts yet)")
            print("=" * 60 + "\n")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Demo terminated gracefully.")


if __name__ == "__main__":
    main()
