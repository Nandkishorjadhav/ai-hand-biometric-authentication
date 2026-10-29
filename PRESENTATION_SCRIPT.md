# Hand Biometric Authentication System
## Presentation Script: Start to End Workflow

## Opening

Good morning everyone.

My project is an **AI-based Hand Biometric Authentication System with Liveness Detection**.

The purpose of this project is to identify a person using the unique geometric structure of their hand and to verify that the hand belongs to a live person instead of a photograph, screenshot, or simple replay.

The system is contactless. It only requires a normal webcam.

The complete flow is:

```text
Webcam
  → Image frame
  → Hand detection
  → 21 hand landmarks
  → Gesture and liveness verification
  → Feature normalization
  → 71-dimensional biometric vector
  → Template comparison
  → Access granted or denied
```

An important point is that this implementation does not train a CNN. It uses a pre-trained MediaPipe hand landmark model, geometric feature calculations, rule-based liveness detection, and cosine similarity matching.

---

## 1. Technologies and Libraries

Before explaining the workflow, I will briefly explain the libraries used.

### OpenCV

OpenCV is used for computer vision and image processing.

In desktop mode, it opens the webcam, reads frames, flips the image for a natural mirror view, draws the hand skeleton, displays status messages, and shows the result window.

It is also used in the web backend to decode incoming JPEG images and draw annotations.

### MediaPipe

MediaPipe provides the pre-trained hand landmark model.

The project uses the file:

```text
models/hand_landmarker.task
```

MediaPipe detects the hand and returns 21 three-dimensional landmarks.

This is the main hand detection model in the project. The project does not train this model.

### NumPy

NumPy is used for numerical calculations.

It handles arrays, distances, rotations, averages, norms, dot products, and cosine similarity.

The biometric calculations are primarily performed using NumPy.

### FastAPI

FastAPI provides the web backend.

It receives camera frames from the browser and exposes endpoints for preview detection, enrollment, authentication, user listing, and logs.

### Uvicorn

Uvicorn runs the FastAPI server.

When we execute:

```text
python app.py
```

Uvicorn starts the application at:

```text
http://localhost:8000
```

### JavaScript and Browser Camera APIs

The frontend uses JavaScript to request webcam permission, capture video frames, draw frames on a canvas, convert them to Base64 JPEG images, and send them to the FastAPI backend.

### SQLite

SQLite is the database used by the project.

It stores user details, biometric templates, and authentication logs in the local file:

```text
hand_auth.db
```

SQLite is built into Python, so no separate database server is required.

### Jinja2

Jinja2 is used by FastAPI to serve the HTML dashboard from:

```text
web/templates/index.html
```

### Pydantic

Pydantic validates the data received by the FastAPI endpoints, such as the image string, user name, email, and biometric vectors.

---

## 2. Starting the Application

Now I will explain the complete workflow from beginning to end.

When the application starts, it creates the FastAPI application, configures the web folders, initializes the SQLite database, and loads the MediaPipe hand detector.

The database creates three main tables:

1. `users`
2. `biometric_templates`
3. `auth_logs`

The `users` table stores the user's name and email.

The `biometric_templates` table stores the user's master biometric vector.

The `auth_logs` table stores every authentication attempt, including the liveness status, similarity score, final result, and reason.

At the same time, the `HandDetector` loads the pre-trained MediaPipe model from:

```text
models/hand_landmarker.task
```

The application is now ready to receive webcam frames.

---

## 3. Capturing a Camera Frame

There are two execution modes.

In web mode, the browser calls:

```javascript
navigator.mediaDevices.getUserMedia(...)
```

This requests access to the user's webcam.

The video stream is displayed in the browser and copied to an HTML canvas.

The frontend captures one frame approximately every 90 milliseconds, which is around 10 to 11 frames per second.

Each frame is converted into a JPEG Base64 string:

```javascript
canvas.toDataURL('image/jpeg', 0.85)
```

The browser sends the image to FastAPI as JSON.

In desktop mode, OpenCV directly performs:

```python
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
```

The frame is mirrored using:

```python
frame = cv2.flip(frame, 1)
```

After this, both modes send the frame to the same hand detection logic.

---

## 4. Decoding the Image

In web mode, the backend receives a Base64 image.

First, it removes the prefix:

```text
data:image/jpeg;base64,
```

Then Python decodes the Base64 data into bytes.

NumPy converts those bytes into an array.

Finally, OpenCV decodes the JPEG:

```python
frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
```

Now the application has a normal OpenCV BGR image that can be processed.

---

## 5. Detecting the Hand with MediaPipe

The backend calls:

```python
detector.find_hand(frame, draw=True)
```

Inside the detector, OpenCV converts the image from BGR to RGB because MediaPipe expects RGB:

```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

The RGB image is converted into a MediaPipe image object.

MediaPipe then runs its pre-trained hand landmark model:

```python
detection_result = self.detector.detect(mp_image)
```

If no hand is found, the system returns:

```text
NO HAND
```

If a hand is found, the system selects the primary hand and extracts 21 landmarks.

Each landmark has:

```text
x coordinate
y coordinate
z coordinate
```

The 21 landmarks are distributed across the wrist, thumb, index finger, middle finger, ring finger, and pinky.

The system also converts the normalized landmark coordinates into pixel coordinates so it can draw the skeleton and bounding box on the video.

The hand detector draws the connections between the joints, highlights fingertips, marks the wrist, draws a bounding box, and displays the handedness.

These drawings are only for user feedback. They are not used as biometric data.

---

## 6. Detecting the Gesture

After the hand is detected, the project identifies the current gesture.

This is handled by:

```python
GestureClassifier.classify_gesture(detection.landmarks)
```

This is a rule-based classifier. It does not use a CNN.

The system calculates distances between finger landmarks and the wrist.

For example, if the fingertip is farther from the wrist than the intermediate finger joint, the finger is considered extended.

The classifier calculates the state of:

```text
Thumb
Index finger
Middle finger
Ring finger
Pinky finger
```

Using these states, it recognizes:

- Open palm
- Fist
- Peace sign
- Thumbs up
- Three fingers
- Unknown gesture

For example, an open palm is detected when the main fingers are extended.

A fist is detected when the fingers are curled.

This gesture result is used mainly for liveness verification and enrollment validation.

---

## 7. User Enrollment

Before a person can authenticate, they must first enroll their hand.

The user enters a name and optionally an email.

The frontend starts sending frames to:

```text
POST /api/enroll-sample
```

For each frame, the backend:

1. Decodes the image.
2. Detects the hand with MediaPipe.
3. Classifies the gesture.
4. Accepts the frame only if the gesture is an open palm.

If no hand is found, the sample is rejected.

If the gesture is not an open palm, the sample is rejected.

If the frame is valid, the system calculates a biometric vector from that frame.

The web interface normally collects 10 valid samples.

The reason for collecting multiple samples is to reduce the effect of small movements, camera noise, and landmark variation.

---

## 8. Calculating the Biometric Features

The biometric calculation is implemented by `LandmarkNormalizer`.

The method receives 21 landmarks, with three coordinates each.

That gives:

```text
21 × 3 = 63 raw coordinate values
```

The system performs three main normalization steps.

### Translation normalization

The wrist is landmark zero.

The wrist coordinate is subtracted from every other landmark:

```python
centered = pts - wrist
```

This moves the wrist to the origin:

```text
(0, 0, 0)
```

This means the hand can move anywhere inside the camera frame without changing the relative hand structure.

### Scale normalization

The distance from the wrist to the middle-finger MCP joint is used as the palm scale.

All coordinates are divided by this scale.

This reduces the effect of the user holding the hand close to or far from the camera.

### Rotation normalization

The system calculates the angle of the wrist-to-middle-finger direction.

It then rotates the x and y coordinates so the hand is aligned to a standard orientation.

This helps make a tilted hand comparable to an upright hand.

After normalization, the 21 landmarks are flattened into:

```text
63 normalized coordinate values
```

---

## 9. Adding Hand Geometry Features

The project also calculates eight additional geometric features.

These include:

1. Thumb length ratio
2. Index finger length ratio
3. Middle finger length ratio
4. Ring finger length ratio
5. Pinky length ratio
6. Index-to-middle fingertip spread
7. Middle-to-ring fingertip spread
8. Ring-to-pinky fingertip spread

Each distance is divided by the palm scale.

This creates size-independent measurements of the user's hand shape.

The final feature count is:

```text
63 normalized landmark values
+ 8 geometry values
= 71 values
```

The system concatenates these values and performs L2 normalization using NumPy.

The output is the final 71-dimensional biometric vector.

This vector represents the geometric structure of the hand.

It is not an image and it is not a CNN embedding.

---

## 10. Creating the Enrollment Template

After 10 valid vectors are collected, the frontend sends them to:

```text
POST /api/enroll-complete
```

The backend converts each list into a NumPy array.

Then `BiometricMatcher.create_master_template()` stacks all vectors and calculates their element-by-element average.

For example:

```text
10 vectors × 71 features
```

becomes:

```text
1 averaged vector × 71 features
```

This average reduces noise and creates a stable master template.

The averaged vector is normalized again.

The master vector is then converted to JSON and saved in SQLite.

The user information is saved in the `users` table.

The biometric template is saved in the `biometric_templates` table.

The sample count is also stored.

Enrollment is now complete.

---

## 11. Starting Authentication

When the user clicks the authentication button, the frontend calls:

```text
POST /api/auth-start
```

The backend first checks whether at least one user has been enrolled.

If no user exists, authentication cannot start.

If users exist, the system creates a liveness session.

The session selects two random gestures from:

```text
Open palm
Fist
Peace sign
Thumbs up
```

Each gesture has:

- A six-second time limit
- A requirement to hold the gesture for five accepted frames

For example, the challenge might be:

```text
Step 1: Show peace sign
Step 2: Make a fist
```

The biometric frame buffer is cleared at the beginning of every authentication session.

---

## 12. Liveness Verification

The browser continues sending frames to:

```text
POST /api/auth-step
```

For every frame, the system:

1. Detects the hand.
2. Classifies the gesture.
3. Creates a biometric vector if a hand is found.
4. Adds that vector to the authentication frame buffer.
5. Checks the detected gesture against the current challenge.

If the detected gesture matches the requested gesture, the hold counter increases.

The progress is calculated as:

```text
hold count / required hold frames
```

With five required frames:

```text
1 frame = 20%
2 frames = 40%
3 frames = 60%
4 frames = 80%
5 frames = 100%
```

After five correct frames, the first step passes and the system moves to the second step.

If the wrong gesture is detected, the hold progress decreases.

If the user fails to complete a step within six seconds, liveness fails.

After both random steps pass:

```text
LIVENESS VERIFIED
```

This liveness process helps reject static photographs and simple non-responsive presentations.

---

## 13. Creating the Live Query Vector

During the liveness process, the application collects a biometric vector from every valid hand frame.

After liveness passes, the system checks whether at least five valid vectors were collected.

If fewer than five vectors exist, access is denied because there is not enough biometric data.

If enough vectors exist, the system averages them:

```python
query_vec = np.mean(stacked, axis=0)
```

For example:

```text
18 live vectors × 71 features
```

becomes:

```text
1 live query vector × 71 features
```

The query vector is normalized again.

This averaged live vector is compared with the stored master vectors.

---

## 14. Matching Against Enrolled Users

The database loads every stored biometric template.

Each JSON vector is converted back to a NumPy array.

The project then calls:

```python
BiometricMatcher.match_hand(...)
```

For every enrolled user, the system calculates cosine similarity between:

```text
Live query vector
and
Stored master template
```

The formula is:

```text
similarity =
       u · v
-------------------
||u|| × ||v||
```

Here:

- `u` is the live query vector.
- `v` is one user's stored template.
- The dot product measures alignment between the vectors.
- The norms measure their magnitudes.

The score is between `-1.0` and `1.0`.

A higher score means the two hand feature vectors are more similar.

Suppose the scores are:

```text
Alice: 0.96
Bob:   0.72
Carol: 0.84
```

Alice is selected as the best candidate because she has the highest score.

---

## 15. Applying the Threshold

The web authentication threshold is:

```python
AUTH_SIMILARITY_THRESHOLD = 0.90
```

This means the best candidate must have at least:

```text
0.90
```

or:

```text
90%
```

If the best score is 0.9674:

```text
0.9674 × 100 = 96.74%
```

Because 96.74% is greater than 90%, the identity check passes.

If the best score is 0.8410:

```text
84.10%
```

the identity check fails.

The system converts the score into a percentage for display, but the actual comparison is performed using the decimal similarity value.

---

## 16. Final Access Decision

The final decision requires both conditions:

```text
Liveness must pass
AND
Similarity must be at least 93%
```

The possible results are:

| Liveness | Similarity | Result |
|---|---:|---|
| Failed | Any value | Access denied |
| Passed | Below 93% | Access denied |
| Passed | At least 93% | Access granted |
| Passed | Fewer than 5 frames | Access denied |

For a successful attempt, the system may return:

```text
ACCESS GRANTED
```

with a message such as:

```text
Identity verified across 18 biometric frames at 96.74% similarity.
```

For a failed identity match, it may return:

```text
ACCESS DENIED
```

with a message such as:

```text
Biometric mismatch. Similarity was below the 93% threshold.
```

For a failed liveness check, the result is:

```text
ACCESS DENIED
```

because the user did not complete the required challenge.

---

## 17. Saving Authentication Logs

Every attempt is saved using:

```python
db.log_auth_attempt(...)
```

The system stores:

- User ID
- User name
- Liveness status
- Similarity score
- Access result
- Failure or success reason
- Timestamp

The frontend retrieves the logs from:

```text
GET /api/logs
```

This allows administrators or evaluators to inspect previous authentication attempts.

---

## 18. Desktop Demonstration Flow

The desktop mode is implemented in `demo.py`.

It uses OpenCV directly and does not require a browser.

The main controls are:

```text
R = Register a new user
A = Authenticate
L = List users and logs
ESC = Cancel
Q = Quit
```

During registration, the user holds an open palm while the system collects 10 samples.

During authentication, the system presents the random gesture challenge, verifies liveness, calculates the live query vector, compares it to stored users, and displays the result in the OpenCV window.

The desktop and web modes use the same:

- MediaPipe detector
- Gesture classifier
- Landmark normalizer
- Biometric matcher
- SQLite database

Only the user interface and frame source are different.

---

## 19. Simple Explanation of the Three Main Stages

The system has three main intelligent stages.

### Hand detection

The question is:

```text
Is there a hand, and where are its joints?
```

MediaPipe answers this by returning 21 landmarks.

### Liveness verification

The question is:

```text
Is the person responding to random gesture instructions?
```

The rule-based gesture classifier and liveness session answer this.

### Biometric matching

The question is:

```text
Which enrolled user's hand shape is most similar?
```

The normalized 71-feature vector and cosine similarity answer this.

The final decision is made only when liveness and identity verification both pass.

---

## 20. Complete Demonstration Example

Now I will describe one complete authentication example.

First, Nandu  enrolls in the system.

He enters his name and holds an open palm.

The system collects 10 valid frames.

Each frame becomes a 71-value vector.

The ten vectors are averaged into one master vector and saved in SQLite.

Later, Nandu starts authentication.

The system randomly asks:

```text
Step 1: Show a peace sign
Step 2: Make a fist
```

Nandu completes both gestures and holds each for five frames.

During the session, the system collects 18 valid hand vectors.

Those 18 vectors are averaged into one live query vector.

The system compares the query vector with Nandu's stored master vector and obtains:

```text
Cosine similarity = 0.9674
```

This becomes:

```text
Similarity score = 96.74%
```

The liveness challenge passed and:

```text
96.74% >= 93%
```

Therefore the system displays:

```text
ACCESS GRANTED: Rahul
```

If the liveness challenge had failed, or if the score had been 84.10%, the result would have been:

```text
ACCESS DENIED
```

---

## Closing

To conclude, this project provides contactless hand authentication using a normal webcam.

OpenCV handles image and camera processing.

MediaPipe detects the 21 hand landmarks.

NumPy performs the feature calculations and vector mathematics.

The landmark normalizer creates a 71-dimensional hand representation.

The gesture classifier and randomized challenge verify liveness.

The biometric matcher uses cosine similarity to compare the live hand with enrolled templates.

SQLite stores users, templates, and authentication logs.

FastAPI and Uvicorn provide the web backend, while JavaScript manages the browser camera and frontend communication.

The final rule is:

```text
Liveness passed
AND
Biometric similarity at least 93%
```

Only when both conditions are satisfied does the system grant access.

Thank you.

---

## Short Viva Answers

### Is this project using a CNN?

No. The current implementation uses a pre-trained MediaPipe landmark model, manually calculated geometric features, rule-based gesture classification, and cosine similarity. It does not contain CNN training code.

### Why are multiple samples collected?

Multiple samples reduce the effect of hand movement, landmark noise, and small changes in camera position. Averaging them creates a more stable biometric template.

### Why is the wrist subtracted?

Subtracting the wrist removes the hand's position in the image. The system then focuses on the relative structure of the hand.

### Why is palm scale used?

Dividing by palm scale reduces the effect of the hand being closer to or farther from the camera.

### Why is rotation normalization used?

It makes the feature vector less sensitive to hand tilt and orientation.

### Why is cosine similarity used?

Cosine similarity compares the direction or pattern of two feature vectors. It works well for comparing normalized biometric representations.

### Why is liveness needed?

Biometric similarity alone should not be the only check. Liveness requires the person to respond to random gestures, helping reject static photographs and non-responsive presentations.

### What happens if the gesture is correct but the similarity is low?

Access is denied. Liveness and identity are separate checks, and both must pass.
