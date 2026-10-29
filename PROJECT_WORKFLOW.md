# Hand Biometric Authentication Project: Complete Workflow

This project is a contactless hand-biometric authentication system. It uses a webcam to detect a hand, extract its landmarks, convert those landmarks into a numeric biometric vector, store that vector during enrollment, and later compare a live hand against stored users.

The actual project does not train a CNN. It uses a pre-trained MediaPipe hand landmark model, manually calculated geometric features, rule-based liveness detection, and cosine-similarity matching.

```text
Webcam
   ↓
OpenCV / Browser Camera
   ↓
MediaPipe Hand Landmarker
   ↓
21 hand landmarks
   ↓
Gesture classification for liveness
   ↓
Landmark normalization
   ↓
71-dimensional biometric vector
   ↓
Template averaging
   ↓
Cosine similarity matching
   ↓
Access granted or denied
```

The project has two execution modes:

1. Web application mode:

   ```text
   python app.py
   ```

2. Desktop OpenCV mode:

   ```text
   python demo.py
   ```

Both modes use the same core detection, normalization, liveness, matching, and database modules.

---

## 1. Libraries Used

The dependencies are listed in `requirements.txt`.

### 1.1 OpenCV

Package:

```text
opencv-python
```

Used in:

- `core/hand_detector.py`
- `app.py`
- `demo.py`

OpenCV handles:

- Reading webcam frames in desktop mode
- Converting color formats
- Resizing and flipping frames
- Drawing hand skeletons and bounding boxes
- Drawing interface banners and progress bars
- Encoding and decoding JPEG images
- Displaying the desktop camera window

Example desktop camera flow:

```python
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.flip(frame, 1)
```

The `0` usually means the default webcam.

In web mode, the browser accesses the webcam, not Python directly. The browser sends images to FastAPI as Base64 JPEG data.

### 1.2 MediaPipe

Package:

```text
mediapipe
```

Used in `core/hand_detector.py`.

MediaPipe provides the pre-trained hand landmark model:

```text
models/hand_landmarker.task
```

This model detects one hand and returns 21 landmarks. It is already trained by Google/MediaPipe. This project does not train it.

Each landmark contains:

```text
x, y, z
```

The coordinates represent:

- `x`: horizontal position
- `y`: vertical position
- `z`: relative depth

The 21 landmarks are:

| Index | Location |
|---:|---|
| 0 | Wrist |
| 1-4 | Thumb |
| 5-8 | Index finger |
| 9-12 | Middle finger |
| 13-16 | Ring finger |
| 17-20 | Pinky finger |

Therefore, the raw hand data contains:

```text
21 landmarks × 3 coordinates = 63 values
```

### 1.3 NumPy

Package:

```text
numpy
```

NumPy performs:

- Coordinate conversion
- Vector calculations
- Euclidean distance calculations
- Matrix rotation
- Mean and average calculations
- Vector normalization
- Dot products
- Norm calculations
- Cosine similarity

Nearly all biometric calculations are done with NumPy.

### 1.4 FastAPI

Package:

```text
fastapi
```

Used in `app.py`.

FastAPI provides the web backend and API endpoints:

```text
/api/detect-preview
/api/auth-start
/api/auth-step
/api/auth-cancel
/api/enroll-sample
/api/enroll-complete
/api/users
/api/logs
```

The browser sends camera frames to these endpoints.

### 1.5 Uvicorn

Package:

```text
uvicorn
```

Uvicorn runs the FastAPI server:

```python
uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
```

The application is then available at:

```text
http://localhost:8000
```

### 1.6 Pydantic

Pydantic is included with FastAPI and validates request data.

For example, `app.py` defines:

```python
class FramePayload(BaseModel):
    image: str
```

This ensures that incoming frame requests contain an `image` field.

Enrollment data uses:

```python
class EnrollCompletePayload(BaseModel):
    name: str
    email: Optional[str] = ""
    vectors: List[List[float]]
```

### 1.7 SQLite

SQLite is built into Python through the `sqlite3` module. No separate database server is required.

The database file is:

```text
hand_auth.db
```

Database handling is implemented in `core/database.py`.

SQLite stores:

- User information
- Biometric templates
- Authentication logs

### 1.8 Jinja2

Package:

```text
jinja2
```

Used to serve:

```text
web/templates/index.html
```

FastAPI loads the page through `Jinja2Templates`.

### 1.9 python-multipart

This package provides FastAPI form and multipart request support. The current biometric frame flow primarily sends JSON containing Base64 image data.

---

## 2. Application Startup

When you run:

```text
python app.py
```

the following happens.

### 2.1 FastAPI starts

An application object is created:

```python
app = FastAPI(...)
```

### 2.2 Static files are mounted

CSS and JavaScript files are served from:

```text
web/static/
```

The HTML template is loaded from:

```text
web/templates/
```

### 2.3 Database is initialized

The application creates the database service:

```python
db = Database(os.path.join(BASE_DIR, "hand_auth.db"))
```

The `Database` class creates three tables if they do not exist.

#### `users`

Stores:

```text
id
name
email
created_at
```

#### `biometric_templates`

Stores:

```text
id
user_id
features_json
sample_count
updated_at
```

The biometric vector is converted into JSON before being stored.

#### `auth_logs`

Stores:

```text
id
user_id
user_name
timestamp
liveness_status
similarity_score
access_result
reason
```

### 2.4 MediaPipe detector is initialized

The application creates:

```python
detector = HandDetector()
```

The detector loads:

```text
models/hand_landmarker.task
```

This model remains available for processing camera frames.

---

## 3. Web Camera Processing

The web camera logic is in:

```text
web/static/js/app.js
```

### 3.1 Browser requests camera permission

The browser calls:

```javascript
navigator.mediaDevices.getUserMedia(...)
```

The requested camera resolution is approximately:

```text
640 × 480
```

The video stream is attached to the HTML video element.

### 3.2 Frames are drawn to a canvas

The browser repeatedly executes `renderLoop()`.

Each iteration:

1. Reads the current video frame.
2. Draws it onto a canvas.
3. Approximately every 90 milliseconds, sends one frame to the backend.

This is approximately:

```text
10-11 frames per second
```

### 3.3 Frame converted to Base64 JPEG

The canvas frame is converted using:

```javascript
canvas.toDataURL('image/jpeg', 0.85)
```

The result is sent to FastAPI as JSON:

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

---

## 4. Image Decoding in Python

The backend receives the Base64 image through a `FramePayload`.

In `app.py`, `decode_base64_image()` performs these steps:

1. Removes the `data:image/jpeg;base64,` prefix if present.
2. Decodes the Base64 string into raw bytes.
3. Converts the bytes into a NumPy array:

   ```python
   nparr = np.frombuffer(img_bytes, np.uint8)
   ```

4. Decodes the JPEG using OpenCV:

   ```python
   frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   ```

The final result is an OpenCV BGR image represented as a NumPy array.

---

## 5. Hand Detection

Hand detection is performed by:

```python
annotated_frame, detection = detector.find_hand(frame, draw=True)
```

This method is implemented in `core/hand_detector.py`.

### 5.1 BGR to RGB conversion

OpenCV normally uses BGR images. MediaPipe expects RGB:

```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

### 5.2 MediaPipe image creation

The RGB NumPy image is wrapped as a MediaPipe image:

```python
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_frame
)
```

### 5.3 MediaPipe inference

The hand landmarker processes the image:

```python
detection_result = self.detector.detect(mp_image)
```

MediaPipe returns:

- Hand landmarks
- Handedness
- Detection results

If no hand is found, the method returns:

```python
HandDetectionResult(has_hand=False)
```

If a hand is found, the first hand is selected.

### 5.4 Landmark extraction

For each landmark, the application stores:

```python
(float(lm.x), float(lm.y), float(lm.z))
```

It also calculates pixel coordinates for drawing:

```python
px = int(lm.x * width)
py = int(lm.y * height)
```

Two types of coordinates are maintained:

#### Normalized coordinates

Used for biometric calculations:

```text
(x, y, z)
```

#### Pixel coordinates

Used for drawing:

```text
(pixel_x, pixel_y)
```

### 5.5 Bounding box

The minimum and maximum x/y coordinates are found. A margin of 25 pixels is added around the hand to create a bounding box.

### 5.6 Drawing

If `draw=True`, OpenCV draws:

- Hand bone connections
- Landmark points
- Fingertip highlights
- Wrist marker
- Bounding box brackets
- Left/right hand label

The drawing is only for visual feedback. It is not part of the biometric vector.

---

## 6. Gesture Recognition for Liveness

Gesture recognition is implemented in:

```text
core/liveness.py
```

This is not a CNN. It is a rule-based geometric classifier.

The main function is:

```python
GestureClassifier.classify_gesture(detection.landmarks)
```

### 6.1 Finger state calculation

`get_finger_states()` determines whether each finger is extended.

For the index, middle, ring, and pinky fingers, it compares distances from the wrist.

For example, for the index finger:

```python
distance(landmark_8, wrist)
>
distance(landmark_6, wrist) * 1.05
```

If the fingertip is sufficiently farther from the wrist than the intermediate joint, that finger is considered extended.

The result looks like:

```python
{
    "thumb": True,
    "index": True,
    "middle": True,
    "ring": True,
    "pinky": True
}
```

### 6.2 Thumb calculation

The thumb uses a different comparison. It compares:

- Thumb tip to pinky MCP
- Thumb IP joint to pinky MCP

If the thumb tip is sufficiently farther away, the thumb is considered extended.

### 6.3 Gesture rules

The classifier recognizes the following gestures.

#### Open palm

All four main fingers are extended:

```text
index = true
middle = true
ring = true
pinky = true
```

#### Fist

No main fingers are extended and the thumb is not extended:

```text
index = false
middle = false
ring = false
pinky = false
thumb = false
```

#### Peace sign

Only index and middle fingers are extended:

```text
index = true
middle = true
ring = false
pinky = false
```

#### Thumbs up

The thumb is extended and all other fingers are curled.

#### Three fingers

Index, middle, and ring fingers are extended while the pinky is curled.

#### Unknown

If none of the expected patterns match, the result is:

```text
UNKNOWN
```

---

## 7. Enrollment Process

Enrollment creates the stored biometric template for a user.

There are two enrollment implementations:

- Web enrollment in `app.py`
- Desktop enrollment in `demo.py`

### 7.1 User enters information

The user enters:

```text
Name
Email
```

The browser then resets:

```javascript
enrollSamplesCollected = 0;
enrolledVectors = [];
isEnrolling = true;
```

### 7.2 Browser sends frames

While enrollment is active, the browser calls:

```javascript
processEnrollCycle()
```

Each processed frame is sent to:

```text
POST /api/enroll-sample
```

### 7.3 Backend detects the hand

The backend:

1. Decodes the image.
2. Runs MediaPipe.
3. Checks if a hand exists.
4. Classifies the gesture.

If no hand exists:

```text
sample_accepted = false
```

If the gesture is not `OPEN_PALM`:

```text
sample_accepted = false
```

This ensures that enrollment uses a consistent open-palm pose.

### 7.4 Biometric vector is generated

For a valid open palm:

```python
vec = LandmarkNormalizer.get_composite_biometric_vector(
    detection.landmarks
)
```

The result is returned to the browser. The browser appends it:

```javascript
enrolledVectors.push(data.vector);
```

The web interface normally collects 10 accepted samples.

The backend requires at least 3 samples in `/api/enroll-complete`, but the web interface normally collects 10.

---

## 8. Landmark Normalization

Normalization is implemented in:

```text
core/normalizer.py
```

The purpose is to make the biometric representation less sensitive to:

- Hand position
- Distance from the camera
- Hand size in the frame
- Hand rotation or tilt
- Minor camera movement

The method produces a final 71-element vector.

### 8.1 Input

The input is:

```text
21 landmarks × (x, y, z)
```

Shape:

```text
21 × 3
```

### 8.2 Translation invariance

The wrist is landmark `0`.

The wrist coordinate is subtracted from every landmark:

```python
centered = pts - wrist
```

Mathematically:

```text
Pcentered_i = Pi - Pwrist
```

This makes the wrist become:

```text
(0, 0, 0)
```

If the hand moves from one side of the camera to another, its absolute coordinates change. Subtracting the wrist removes that global position difference.

### 8.3 Scale invariance

The distance between the wrist and middle-finger MCP landmark `9` is used as palm scale:

```python
palm_scale = ||landmark_9 - landmark_0||
```

Every centered coordinate is divided by this scale:

```python
scaled = centered / palm_scale
```

Mathematically:

```text
Pscaled_i = Pcentered_i / S
```

This reduces the effect of the hand being closer to or farther from the camera.

If the scale is almost zero, the code uses `1.0` to avoid division by zero.

### 8.4 Rotation invariance

The direction from the wrist to middle MCP is used to estimate hand angle.

The code calculates:

```python
angle = arctan2(y, x)
```

It then calculates the rotation needed to align that direction to a target vertical angle.

A 2D rotation matrix is created:

```text
[ cos(theta)  -sin(theta) ]
[ sin(theta)   cos(theta) ]
```

The x and y coordinates are rotated using that matrix. The z coordinate is kept unchanged.

This makes a tilted hand more comparable to a straight hand.

### 8.5 Flattening

After normalization, the data still has shape:

```text
21 × 3
```

It is flattened:

```python
normalized.flatten()
```

This produces:

```text
21 × 3 = 63 values
```

---

## 9. Additional Hand Geometry Features

The project adds morphological measurements in:

```python
extract_hand_geometry()
```

These are ratios, so they are less affected by hand size.

The following features are calculated:

1. Thumb length ratio
2. Index finger length ratio
3. Middle finger length ratio
4. Ring finger length ratio
5. Pinky length ratio
6. Index-to-middle fingertip spread
7. Middle-to-ring fingertip spread
8. Ring-to-pinky fingertip spread

Each distance is divided by the palm scale.

For example:

```text
index_length_ratio =
distance(index MCP, index tip) / palm_scale
```

There are 8 geometry values.

Therefore:

```text
63 normalized coordinates
+ 8 geometry ratios
= 71 values
```

---

## 10. Final 71-Dimensional Biometric Vector

The 63 normalized landmark values and 8 geometry values are concatenated:

```python
composite = np.concatenate([
    normalized_coords,
    geom_vector
])
```

The full vector is then L2-normalized:

```python
norm = np.linalg.norm(composite)

if norm > 1e-6:
    composite = composite / norm
```

The final output is:

```text
71 floating-point values
```

Conceptually:

```text
[
  normalized wrist values,
  normalized thumb landmark values,
  normalized index landmark values,
  normalized middle landmark values,
  normalized ring landmark values,
  normalized pinky landmark values,
  thumb length ratio,
  index length ratio,
  middle length ratio,
  ring length ratio,
  pinky length ratio,
  finger spread ratios
]
```

This is the actual biometric representation used by the project. It is not an image embedding generated by a CNN.

---

## 11. Enrollment Template Creation

After the samples are collected, the browser sends them to:

```text
POST /api/enroll-complete
```

The backend converts the submitted vectors into NumPy arrays:

```python
sample_vecs = [
    np.array(v, dtype=np.float64)
    for v in payload.vectors
]
```

Then it calls:

```python
BiometricMatcher.create_master_template(sample_vecs)
```

This is implemented in `core/matcher.py`.

### 11.1 Stack the vectors

If there are 10 vectors of length 71:

```text
10 × 71
```

The vectors are stacked into a NumPy matrix.

### 11.2 Calculate the mean

The vectors are averaged element by element:

```python
mean_vector = np.mean(stacked, axis=0)
```

For each feature:

```text
mean_feature_i =
(feature_1_i + feature_2_i + ... + feature_10_i) / 10
```

This reduces noise caused by:

- Small hand movements
- Camera detection variation
- Landmark jitter
- Slight pose differences

### 11.3 Normalize the mean vector

The averaged vector is normalized again:

```python
mean_vector = mean_vector / norm
```

This becomes the user's master biometric template.

---

## 12. Saving the User to SQLite

The method `Database.add_user()` saves the user.

### 12.1 User record

The name and email are inserted into the `users` table.

### 12.2 Feature vector conversion

NumPy arrays cannot be stored directly in SQLite, so the vector is converted:

```python
features_json = json.dumps(vector.tolist())
```

For example, a vector becomes a JSON array:

```json
[0.01, -0.12, 0.43]
```

The real vector contains 71 values.

### 12.3 Template record

The JSON vector is stored in `biometric_templates`:

```text
user_id
features_json
sample_count
```

The project keeps one master template per user.

At this point, enrollment is complete.

---

## 13. Starting Authentication

The user clicks **Start Authentication**.

The browser calls:

```text
POST /api/auth-start
```

The backend first checks whether any user templates exist. If there are no registered users, authentication is rejected.

If users exist, the backend creates:

```python
LivenessChallengeSession(
    steps_count=2,
    step_timeout_seconds=6.0,
    required_hold_frames=5
)
```

This means:

- Two gestures are selected.
- Each gesture has a six-second time limit.
- Each gesture must be held for five accepted frames.

The gestures are selected randomly from:

```text
OPEN_PALM
FIST
PEACE_SIGN
THUMBS_UP
```

For example:

```text
Step 1: OPEN_PALM
Step 2: FIST
```

The biometric frame buffer is cleared:

```python
auth_biometric_frames = []
```

---

## 14. Authentication Frame Processing

During authentication, the browser repeatedly sends frames to:

```text
POST /api/auth-step
```

The backend performs these operations for every frame.

### Step 1: Decode image

The Base64 image is converted into an OpenCV BGR NumPy image.

### Step 2: Detect hand

MediaPipe extracts the 21 landmarks.

### Step 3: Classify gesture

The landmarks are sent to:

```python
GestureClassifier.classify_gesture(...)
```

The result might be:

```text
OPEN_PALM
FIST
PEACE_SIGN
NO HAND
```

### Step 4: Extract biometric vector

If a hand is detected, the vector is generated even before the liveness challenge finishes:

```python
vec = LandmarkNormalizer.get_composite_biometric_vector(
    detection.landmarks
)
```

It is added to:

```python
auth_biometric_frames
```

This means the system collects multiple biometric observations throughout the complete authentication session.

The constant is:

```python
MIN_BIOMETRIC_FRAMES = 5
```

At least five valid hand frames are needed for identity matching.

---

## 15. Liveness Challenge Calculation

Each detected gesture is passed to:

```python
auth_session.update(gesture)
```

The liveness session checks the current target gesture.

For example, if the target is `OPEN_PALM`:

```python
detected_gesture == "OPEN_PALM"
```

must be true.

### 15.1 Correct gesture

If the gesture matches:

```python
hold_frames_count += 1
```

The progress is:

```text
hold_frames_count / required_hold_frames
```

With five required frames:

```text
1 / 5 = 20%
2 / 5 = 40%
3 / 5 = 60%
4 / 5 = 80%
5 / 5 = 100%
```

Once five matching frames are received, the challenge step passes.

### 15.2 Incorrect gesture

If the gesture does not match:

```python
hold_frames_count = max(0, hold_frames_count - 1)
```

The progress decreases instead of immediately resetting to zero.

### 15.3 Timeout

Each step has a six-second timeout.

If the correct gesture is not held within six seconds:

```text
liveness_status = FAILED
```

Authentication immediately fails.

### 15.4 Two-step completion

After the first gesture passes, the session moves to the second selected gesture.

After both gestures pass:

```text
liveness_status = PASSED
```

The user has responded to random real-time instructions. This makes a static image or simple replay less likely to pass.

---

## 16. Creating the Authentication Query Vector

When liveness passes, the system checks how many biometric frames were collected.

If fewer than five frames were captured:

```text
ACCESS DENIED
```

with a reason such as:

```text
Insufficient biometric data
```

If at least five frames exist, the vectors are averaged.

Suppose the session collected:

```text
20 vectors × 71 features
```

The application calculates:

```python
stacked = np.array(auth_biometric_frames, dtype=np.float64)
query_vec = np.mean(stacked, axis=0)
```

This creates one averaged live query vector.

The query vector is then L2-normalized:

```python
query_vec = query_vec / np.linalg.norm(query_vec)
```

The result is the final biometric representation for this authentication attempt.

---

## 17. Loading Stored Templates

The database loads all enrolled templates using:

```python
db.get_all_templates()
```

Each stored JSON vector is converted back to a NumPy array:

```python
np.array(json.loads(r["features_json"]), dtype=np.float64)
```

The resulting list conceptually looks like:

```python
[
    {
        "user_id": 1,
        "user_name": "Alice",
        "vector": np.array([...])
    },
    {
        "user_id": 2,
        "user_name": "Bob",
        "vector": np.array([...])
    }
]
```

---

## 18. Cosine Similarity Matching

Matching is implemented in `core/matcher.py`.

For the live query vector and every stored user vector, the application calculates cosine similarity:

```python
similarity = dot(v1, v2) / (norm(v1) * norm(v2))
```

Mathematically:

```text
similarity =
       u · v
-------------------
||u|| × ||v||
```

Where:

- `u` is the live query vector.
- `v` is the enrolled template.
- `u · v` is the dot product.
- `||u||` and `||v||` are vector magnitudes.

Cosine similarity ranges from:

```text
-1.0 to 1.0
```

For these normalized hand feature vectors, a higher positive value means the feature directions are more similar.

### Example

Suppose the system compares the live hand against three users:

```text
Alice: 0.96
Bob:   0.72
Carol: 0.84
```

The best candidate is Alice because she has the highest score.

The system selects the highest-scoring user and then checks whether the score passes the threshold.

---

## 19. Similarity Threshold

Web authentication uses:

```python
AUTH_SIMILARITY_THRESHOLD = 0.93
```

The best similarity must be at least:

```text
0.93
```

or:

```text
93%
```

The matching function returns:

```python
(
    best_user_id,
    best_user_name,
    score_percentage,
    is_match
)
```

The percentage is calculated by:

```python
score_pct = max(0.0, similarity) * 100.0
```

It is then rounded to two decimal places.

Example:

```text
similarity = 0.9578
display score = 95.78%
```

### Access granted

If:

```text
best_similarity >= 0.93
```

then:

```text
ACCESS GRANTED
```

### Access denied

If:

```text
best_similarity < 0.93
```

then:

```text
ACCESS DENIED
```

Even if the highest-scoring user is selected, the score must still pass the threshold.

---

## 20. Final Authentication Decision

Authentication requires both checks to pass:

```text
Liveness PASSED
AND
Biometric similarity >= 93%
```

The final decision logic is:

| Liveness | Similarity | Result |
|---|---:|---|
| Failed | Any | Access denied |
| Passed | Less than 93% | Access denied |
| Passed | 93% or higher | Access granted |
| Passed | Fewer than 5 biometric frames | Access denied |

The reason is recorded for each result.

Example successful reason:

```text
Identity verified across 20 biometric frames (95.78% match)
```

Example failed reason:

```text
Biometric mismatch - similarity 81.42% is below 93% threshold
```

---

## 21. Authentication Logging

Every authentication result is written to the `auth_logs` table using:

```python
db.log_auth_attempt(...)
```

The log contains:

```text
User ID
User name
Liveness status
Similarity score
Access result
Reason
Timestamp
```

Examples:

```text
PASSED | 96.32 | ACCESS GRANTED
PASSED | 81.42 | ACCESS DENIED
FAILED_TIMEOUT | 0.0 | ACCESS DENIED
```

The web frontend loads these logs through:

```text
GET /api/logs
```

and displays them in the audit logs table.

---

## 22. Desktop Mode

The desktop workflow is in:

```text
demo.py
```

It uses OpenCV directly instead of a browser.

### Camera loop

The desktop application:

1. Opens camera device `0`.
2. Sets resolution to approximately `960 × 540`.
3. Reads frames continuously.
4. Mirrors the frame horizontally.
5. Sends the frame to `HandDetector`.
6. Displays the skeleton and user interface.

### Keyboard controls

| Key | Function |
|---|---|
| `R` | Register a user |
| `A` | Authenticate |
| `L` | List users and logs |
| `ESC` | Cancel current action |
| `Q` | Quit |

### Desktop enrollment

In registration mode:

1. The user enters a name.
2. The user shows an open palm.
3. The application collects 10 vectors.
4. The vectors are averaged.
5. The master vector is saved to SQLite.

### Desktop authentication

In authentication mode:

1. A random liveness challenge is created.
2. The user follows the gesture instructions.
3. Each gesture must be held for the required number of frames.
4. After liveness passes, the live hand is compared to stored templates.
5. A result banner is shown on the OpenCV window.

The desktop mode uses the same:

- MediaPipe detector
- Gesture classifier
- Landmark normalizer
- Biometric matcher
- SQLite database

as web mode.

---

## 23. Difference Between Detection, Liveness, and Matching

These are three separate functions.

### Detection

Question:

```text
Is there a hand in the image, and where are its landmarks?
```

Technology:

```text
MediaPipe Hand Landmarker
```

Output:

```text
21 × (x, y, z) landmarks
```

### Liveness

Question:

```text
Is the person responding to the requested gesture sequence?
```

Technology:

```text
Rule-based finger geometry + randomized challenge
```

Output:

```text
PASSED or FAILED
```

### Biometric matching

Question:

```text
Which enrolled user's hand geometry is most similar?
```

Technology:

```text
Normalized 71-feature vector + cosine similarity
```

Output:

```text
Candidate user + similarity score
```

The final system combines all three.

---

## 24. What the Project Does Not Use

Although some descriptions in the project documentation mention them, the actual code does not currently use:

```text
PyTorch
TensorFlow
Keras
CNN
Siamese neural network
LSTM
GRU
FAISS
Scikit-learn
PostgreSQL
React
Docker
```

The actual implemented system uses:

```text
OpenCV
MediaPipe
NumPy
FastAPI
Uvicorn
Jinja2
Pydantic
SQLite
JavaScript Canvas/Webcam APIs
```

---

## 25. Complete Example

Assume a user named **Rahul** enrolls.

### Enrollment

The system captures 10 open-palm frames.

Each frame produces:

```text
71-value vector
```

The vectors are averaged:

```text
10 vectors → 1 master vector
```

The master vector is stored in SQLite.

### Authentication

The system randomly requests:

```text
Step 1: Peace sign
Step 2: Fist
```

The user performs both gestures and holds each for five accepted frames.

During this time, the system captures 18 valid hand vectors.

Those 18 vectors are averaged:

```text
18 live vectors → 1 query vector
```

The query vector is compared with Rahul's stored vector:

```text
cosine similarity = 0.9674
```

Converted to a display score:

```text
96.74%
```

Since:

```text
96.74% >= 93%
```

and liveness passed:

```text
ACCESS GRANTED: Rahul
```

If the score had been:

```text
84.10%
```

the result would be:

```text
ACCESS DENIED: Biometric mismatch
```

Even if the user performed the gestures correctly, the identity score would still fail.

---

## Summary

The project works as follows:

```text
1. Start application
2. Initialize SQLite database
3. Load MediaPipe pre-trained hand model
4. Open webcam through browser or OpenCV
5. Capture camera frame
6. Decode image
7. Detect hand with MediaPipe
8. Extract 21 landmarks
9. Classify finger gesture
10. For enrollment:
       - Require open palm
       - Create a 71-value vector
       - Collect 10 vectors
       - Average vectors
       - Store master template
11. For authentication:
       - Select two random gestures
       - Require each gesture for five frames
       - Enforce a six-second timeout
       - Collect live biometric vectors
       - Average them
       - Compare with every enrolled user
       - Select the highest cosine similarity
       - Require at least 93%
12. Require both liveness and identity match
13. Grant or deny access
14. Save the result to SQLite audit logs
```

The project is a **MediaPipe landmark-based biometric matching system with rule-based liveness detection**, not a CNN-based training system.
