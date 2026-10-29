# 🖐️ AI-Based Hand Biometric Authentication with Liveness Detection

> **College Project Demonstration Edition**  
> Contactless Biometric Authentication using Computer Vision, Landmark Normalization, Anti-Spoofing Liveness Verification, and Cosine Similarity Matching.

---

## 📌 Project Overview

This project implements an **AI-powered biometric authentication system** that verifies a user's identity using their unique **hand biometric features** and ensures physical presence using **anti-spoofing liveness detection**.

Unlike traditional fingerprint scanners that require physical contact, this system is completely **contactless** and runs in real time using a standard webcam.

### 🎯 Key Objectives
1. **Real-Time Hand Landmark Tracking**: Detects 21 anatomical hand landmarks $(x, y, z)$ using MediaPipe.
2. **Scale & Distance Invariance**: Normalizes coordinates so detection works whether the hand is close to or far from the camera.
3. **Anti-Spoofing Liveness Detection**: Dynamic challenge-response verification (e.g. *"Show Open Palm"* &rarr; *"Make a Fist"*) that reliably defeats photos, phone screens, and video replay attacks.
4. **Biometric Similarity Matching**: Compares 71-dimensional invariant biometric feature vectors using Cosine Similarity against enrolled templates.
5. **Zero-Configuration Relational Storage**: Built-in SQLite database (`hand_auth.db`) storing registered users, master templates, and audit logs.
6. **Dual Demonstration Modes**:
   - **Mode A (Web Dashboard)**: Modern dark-theme browser interface with live telemetry and enrollment wizard.
   - **Mode B (Desktop OpenCV Demo)**: 1-click desktop window with hotkeys—ideal as a bulletproof backup if Wi-Fi is unavailable during college presentations.

---

## 🏗️ System Architecture

```text
               ┌───────────────────────────┐
               │    Webcam / Video Feed    │
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │  MediaPipe Hand Detection │
               │   (21 3D Joint Points)    │
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │   Landmark Normalization  │
               │   • Translate: Wrist to 0 │
               │   • Scale: Palm Distance  │
               │   • Planar Rotation Align │
               └─────────────┬─────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌───────────────────────┐   ┌────────────────────────┐
   │   LIVENESS ENGINE     │   │   BIOMETRIC MATCHER    │
   │ Dynamic Challenge:    │   │ 71-D Composite Vector  │
   │ "Open Palm" → "Fist"  │   │ Cosine Similarity vs   │
   │ (Defeats Photos)      │   │ Enrolled Templates     │
   └──────────┬────────────┘   └───────────┬────────────┘
              │                            │
              ▼                            ▼
      Liveness = PASS              Similarity ≥ 90%
              │                            │
              └──────────────┬─────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  FINAL ACCESS DECISION│
                 │   Liveness AND Match  │
                 └───────────┬───────────┘
                             │
             ├── YES ───────► ACCESS GRANTED (Audit Log Saved)
             └── NO  ───────► ACCESS DENIED  (Reason Logged)
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation
Ensure you have **Python 3.10+** installed:

```bash
# Clone or navigate to the project directory
cd "d:/Hand Biometri Authentication/hand-biometric-auth"

# Install lightweight dependencies
pip install -r requirements.txt
```

---

### 2. Running Mode A: Interactive Web Dashboard
Start the FastAPI server:

```bash
python app.py
```
Open your browser at: **`http://localhost:8000`**

- **Live Authentication Tab**: Click *"Start Authentication"*, follow the on-screen gesture challenge, and observe instant biometric identification.
- **Enroll User Tab**: Enter your name and hold an open palm to capture 10 clean training samples.
- **Audit Logs Tab**: Inspect stored user records, similarity scores, and past authentication attempts.

---

### 3. Running Mode B: 1-Click Desktop OpenCV Demo
If you are presenting in a college classroom with a projector and no internet access:

```bash
python demo.py
```

**Hotkeys in Desktop Mode:**
- **`[R]`**: Register a new user (prompts for name in terminal, then captures 10 hand frames).
- **`[A]`**: Authenticate (starts dynamic liveness challenge + biometric template match).
- **`[L]`**: Print enrolled users and recent authentication logs in terminal.
- **`[ESC]`**: Cancel active action.
- **`[Q]`**: Quit demo.

---

## 🧠 How the System Works

### 1. 21 Hand Landmarks Extraction
MediaPipe tracks 21 anatomical landmarks across each finger:
- **Landmark 0**: Wrist anchor
- **Landmarks 1–4**: Thumb (CMC, MCP, IP, Tip)
- **Landmarks 5–8**: Index finger (MCP, PIP, DIP, Tip)
- **Landmarks 9–12**: Middle finger (MCP, PIP, DIP, Tip)
- **Landmarks 13–16**: Ring finger (MCP, PIP, DIP, Tip)
- **Landmarks 17–20**: Pinky finger (MCP, PIP, DIP, Tip)

### 2. Invariant Normalization
If a user stands closer to or farther from the camera, raw pixel values change drastically. To solve this, our `LandmarkNormalizer` applies three mathematical transformations:
1. **Translation Invariance**: Shifts all coordinates so the wrist (Landmark 0) is at $(0, 0, 0)$:
   $$P_{\text{centered}} = P_i - P_{\text{wrist}}$$
2. **Scale Invariance**: Divides all distances by the palm scale $S$ (Euclidean distance from wrist $P_0$ to middle knuckle $P_9$):
   $$S = \|P_9 - P_0\|_2, \quad P_{\text{scaled}} = \frac{P_{\text{centered}}}{S}$$
3. **Planar Rotation Alignment**: Aligns the palm axis with the vertical axis so tilted hands produce identical feature vectors.

### 3. Liveness Challenge Engine
Static photos or phone screen replays cannot physically respond to randomized instruction sequences in real time:
- The system picks 2 random gestures (e.g. `OPEN_PALM` &rarr; `FIST`).
- The user has a 6-second window to complete each challenge.
- The gesture must be held steadily for multiple consecutive frames to filter sensor noise.
- If the challenge times out or wrong gestures are shown &rarr; **Spoofing Alert: ACCESS DENIED**.

### 4. Biometric Feature Matching
Once liveness passes, the system calculates **Cosine Similarity** between the live 71-element biometric vector $\vec{u}$ and enrolled templates $\vec{v}$:
$$\text{Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$
- **Threshold**: $\ge 90\%$ ($0.90$) indicates an authenticated match.
- If similarity is below $90\%$, the user is treated as an unregistered impostor.

---

## 🎓 College Viva Q&A Guide

**Q1: Why use hand biometrics instead of fingerprint or facial recognition?**  
*Answer:* Hand geometry is completely contactless (hygienic), works on standard low-cost webcams without specialized IR depth sensors, and is less invasive than facial recognition.

**Q2: How does the system prevent someone from showing a printed photo of the user's hand?**  
*Answer:* Through dynamic interactive liveness detection. The system commands the user to perform randomized gestures (e.g., *"Make a fist, then show two fingers"*). A printed photograph or pre-recorded video cannot adapt to unpredictable live challenge sequences.

**Q3: What happens if the user moves their hand closer or further from the camera?**  
*Answer:* Our normalization module divides all coordinates by the palm scale (distance between wrist and middle knuckle). This makes all biometric distances scale-independent.

**Q4: What algorithm is used for template matching?**  
*Answer:* Cosine Similarity on L2-normalized composite feature vectors (combining 63 normalized coordinates with 8 morphological finger ratio proportions).

**Q5: Where is user data stored?**  
*Answer:* In an embedded SQLite database (`hand_auth.db`). No raw camera video is saved; only numerical biometric vectors and timestamped audit logs are retained for privacy.

---

## 📁 Project Directory Structure

```text
hand-biometric-auth/
│
├── core/
│   ├── __init__.py
│   ├── hand_detector.py      # MediaPipe Tasks HandLandmarker (21 3D points)
│   ├── normalizer.py         # Scale, translation & rotation normalization
│   ├── liveness.py           # Gesture classifier & dynamic challenge engine
│   ├── matcher.py            # Cosine similarity matcher & template aggregator
│   └── database.py           # SQLite database layer (Users, Templates, Logs)
│
├── web/
│   ├── static/
│   │   ├── css/style.css     # Cyber-styled dark theme
│   │   └── js/app.js         # Camera handling & real-time telemetry
│   └── templates/
│       └── index.html        # Single-page web dashboard
│
├── models/
│   └── hand_landmarker.task  # Google MediaPipe offline landmark model
│
├── tests/
│   ├── test_core.py          # Unit tests for core algorithms
│   └── test_api.py           # Integration tests for FastAPI endpoints
│
├── app.py                    # Web server application (FastAPI)
├── demo.py                   # Desktop OpenCV demo application
├── requirements.txt          # Python dependencies
└── README.md                 # Complete documentation & viva guide
```

---

## 🧪 Testing & Verification

Run the automated test suite:

```bash
# Run core unit tests
python -m unittest tests/test_core.py

# Run API integration tests
python -m unittest tests/test_api.py
```
Both test suites should pass with `OK`.
