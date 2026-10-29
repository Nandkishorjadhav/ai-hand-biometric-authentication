/**
 * Frontend Application Logic
 * Manages webcam feed, smooth 60 FPS rendering, frame transmission to FastAPI backend,
 * real-time biometric telemetry, HUD skeleton overlays, and database operations.
 */

let videoStream = null;
let isProcessingAuth = false;
let isEnrolling = false;
let isApiBusy = false;
let enrollSamplesCollected = 0;
const ENROLL_TARGET_SAMPLES = 10;
let enrolledVectors = [];

// DOM Elements
const authVideo = document.getElementById('authVideo');
const authCanvas = document.getElementById('authCanvas');
const hudCanvas = document.getElementById('hudCanvas');
const enrollCanvas = document.getElementById('enrollCanvas');
const enrollHudCanvas = document.getElementById('enrollHudCanvas');

const authCtx = authCanvas ? authCanvas.getContext('2d') : null;
const hudCtx = hudCanvas ? hudCanvas.getContext('2d') : null;
const enrollCtx = enrollCanvas ? enrollCanvas.getContext('2d') : null;
const enrollHudCtx = enrollHudCanvas ? enrollHudCanvas.getContext('2d') : null;

// Dedicated Offscreen Canvas for pristine frame capture (isolated from HUD drawings)
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = 640;
offscreenCanvas.height = 480;
const offscreenCtx = offscreenCanvas.getContext('2d');

// Anatomical hand skeleton topology (connections between joints)
const HAND_CONNECTIONS = [
  // Thumb ray
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky finger
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm mesh & knuckle arcs
  [5, 9], [9, 13], [13, 17], [0, 17], [0, 1], [1, 5], [5, 13], [9, 17]
];

// Initialize Camera
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
    });
    videoStream = stream;
    authVideo.srcObject = stream;
    await authVideo.play();

    // Standardize viewport canvas resolutions
    [authCanvas, hudCanvas, enrollCanvas, enrollHudCanvas].forEach(c => {
      if (c) {
        c.width = 640;
        c.height = 480;
      }
    });

    // 1. Silky-smooth 60 FPS live video render loop (never blocked by network)
    requestAnimationFrame(renderLoop);

    // 2. Decoupled asynchronous backend telemetry & liveness detection loop (~10 FPS)
    setInterval(apiTick, 100);
  } catch (err) {
    console.error("Camera access error:", err);
    const statusEl = document.getElementById('systemStatus');
    if (statusEl) {
      statusEl.innerHTML = `<span style="color: var(--accent-red);">❌ CAMERA ERROR</span>`;
    }
    alert("Unable to access camera. Please allow webcam permissions.");
  }
}

// ── Continuous 60 FPS Smooth Video Render Loop ─────────────────────────────
function renderLoop() {
  if (videoStream && authVideo.readyState >= authVideo.HAVE_CURRENT_DATA) {
    // 1. Render mirrored live feed to Auth Viewfinder
    if (authCtx) {
      authCtx.save();
      authCtx.translate(authCanvas.width, 0);
      authCtx.scale(-1, 1);
      authCtx.drawImage(authVideo, 0, 0, authCanvas.width, authCanvas.height);
      authCtx.restore();
    }

    // 2. Render mirrored live feed to Enroll Preview Viewfinder
    if (enrollCtx) {
      enrollCtx.save();
      enrollCtx.translate(enrollCanvas.width, 0);
      enrollCtx.scale(-1, 1);
      enrollCtx.drawImage(authVideo, 0, 0, enrollCanvas.width, enrollCanvas.height);
      enrollCtx.restore();
    }
  }
  requestAnimationFrame(renderLoop);
}

// ── Pristine Frame Capture (Horizontally mirrored to match user perspective) ─
function captureCleanFrame() {
  if (!videoStream || authVideo.readyState < authVideo.HAVE_CURRENT_DATA) return null;
  offscreenCtx.save();
  offscreenCtx.translate(offscreenCanvas.width, 0);
  offscreenCtx.scale(-1, 1);
  offscreenCtx.drawImage(authVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
  offscreenCtx.restore();
  return offscreenCanvas.toDataURL('image/jpeg', 0.80);
}

// ── Render High-Tech Biometric HUD Overlay ──────────────────────────────────
function drawBiometricHud(ctx, landmarks, bbox, handedness) {
  if (!ctx) return;
  ctx.clearRect(0, 0, 640, 480);

  if (!landmarks || landmarks.length === 0) return;

  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // 1. Glowing biometric bone connections (two-pass glow + crisp)
  // Glow pass
  ctx.strokeStyle = 'rgba(0, 220, 255, 0.40)';
  ctx.lineWidth = 4.5;
  for (const [p1, p2] of HAND_CONNECTIONS) {
    if (landmarks[p1] && landmarks[p2]) {
      ctx.beginPath();
      ctx.moveTo(landmarks[p1][0], landmarks[p1][1]);
      ctx.lineTo(landmarks[p2][0], landmarks[p2][1]);
      ctx.stroke();
    }
  }

  // Sharp crisp pass
  ctx.strokeStyle = 'rgba(0, 255, 255, 0.95)';
  ctx.lineWidth = 1.5;
  for (const [p1, p2] of HAND_CONNECTIONS) {
    if (landmarks[p1] && landmarks[p2]) {
      ctx.beginPath();
      ctx.moveTo(landmarks[p1][0], landmarks[p1][1]);
      ctx.lineTo(landmarks[p2][0], landmarks[p2][1]);
      ctx.stroke();
    }
  }

  // 2. Biometric landmark joints
  landmarks.forEach(([x, y], idx) => {
    ctx.beginPath();
    if ([4, 8, 12, 16, 20].includes(idx)) {
      // Fingertips: Emerald green pulse
      ctx.arc(x, y, 6.5, 0, Math.PI * 2);
      ctx.fillStyle = '#00ff88';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else if (idx === 0) {
      // Wrist: Amber anchor
      ctx.arc(x, y, 7.5, 0, Math.PI * 2);
      ctx.fillStyle = '#f59e0b';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      // Internal joints: Cyan
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#22d3ee';
      ctx.fill();
    }
  });

  // 3. Sci-Fi corner brackets around bounding box
  if (bbox && bbox.length === 4) {
    const [x1, y1, x2, y2] = bbox;
    const len = 20;
    const isLeft = (handedness && handedness.toLowerCase() === 'left');
    const color = isLeft ? '#38bdf8' : '#10b981';

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    // Top-left
    ctx.beginPath();
    ctx.moveTo(x1, y1 + len); ctx.lineTo(x1, y1); ctx.lineTo(x1 + len, y1);
    ctx.stroke();
    // Top-right
    ctx.beginPath();
    ctx.moveTo(x2 - len, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + len);
    ctx.stroke();
    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(x1, y2 - len); ctx.lineTo(x1, y2); ctx.lineTo(x1 + len, y2);
    ctx.stroke();
    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(x2 - len, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - len);
    ctx.stroke();

    // Handedness badge pill above bounding box
    const tagText = `${(handedness || 'HAND').toUpperCase()} HAND`;
    ctx.font = 'bold 11px monospace';
    const tw = ctx.measureText(tagText).width;
    const badgeY = Math.max(0, y1 - 22);

    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(x1, badgeY, tw + 16, 19);

    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(x1, badgeY, tw + 16, 19);

    ctx.fillStyle = color;
    ctx.fillText(tagText, x1 + 8, Math.max(13, y1 - 8));
  }
}

// ── Decoupled API Telemetry Loop (~10 FPS) ──────────────────────────────────
async function apiTick() {
  if (isApiBusy) return; // Prevent request piling
  if (!videoStream || authVideo.readyState < authVideo.HAVE_CURRENT_DATA) return;

  isApiBusy = true;
  try {
    if (isProcessingAuth) {
      await processAuthCycle();
    } else if (isEnrolling) {
      await processEnrollCycle();
    } else {
      await processIdleCycle();
    }
  } catch (err) {
    // Graceful non-blocking error handler
    console.warn("API tick notice:", err);
  } finally {
    isApiBusy = false;
  }
}

// ── Idle Mode: Continuous real-time hand detection & gesture telemetry ──────
async function processIdleCycle() {
  const base64Data = captureCleanFrame();
  if (!base64Data) return;

  const response = await fetch('/api/detect-preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: base64Data })
  });
  if (!response.ok) return;

  const data = await response.json();
  if (data.has_hand && data.landmarks && data.landmarks.length > 0) {
    drawBiometricHud(hudCtx, data.landmarks, data.bbox, data.handedness);
  } else {
    if (hudCtx) hudCtx.clearRect(0, 0, 640, 480);
  }

  updateTelemetry(
    data.gesture || "NO HAND",
    data.handedness || "--",
    "IDLE",
    "0.0%",
    "None"
  );
}

// ── Authentication Cycle: Liveness Challenge & Identity Verification ────────
async function processAuthCycle() {
  const base64Data = captureCleanFrame();
  if (!base64Data) return;

  const response = await fetch('/api/auth-step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: base64Data })
  });
  if (!response.ok) return;

  const data = await response.json();

  if (data.has_hand && data.landmarks && data.landmarks.length > 0) {
    drawBiometricHud(hudCtx, data.landmarks, data.bbox, data.handedness);
  } else {
    if (hudCtx) hudCtx.clearRect(0, 0, 640, 480);
  }

  // Guidance HUD Elements
  const stepBadge = document.getElementById('challengeStepBadge');
  const timerBadge = document.getElementById('challengeTimer');
  const iconEl = document.getElementById('challengeIcon');
  const titleEl = document.getElementById('challengeTitle');
  const actionEl = document.getElementById('challengeActionText');
  const feedbackEl = document.getElementById('challengeCameraFeedback');
  const progressEl = document.getElementById('challengeProgress');
  const holdTextEl = document.getElementById('challengeHoldText');

  if (data.liveness_status === "IN_PROGRESS" || data.liveness_status === "STEP_PASSED") {
    if (stepBadge) stepBadge.innerText = `STEP ${data.current_step || 1} OF ${data.total_steps || 2}`;
    if (timerBadge) timerBadge.innerText = `⏱️ ${(data.time_left !== undefined ? data.time_left : 6.0).toFixed(1)}s`;
    if (iconEl) iconEl.innerText = data.target_icon || '🖐️';
    if (titleEl) titleEl.innerText = data.target_title || 'Show Gesture';
    if (actionEl) actionEl.innerText = data.target_action ? `👉 Action: ${data.target_action}` : data.instruction;

    if (feedbackEl) {
      if (!data.has_hand) {
        feedbackEl.innerHTML = `<span style="color: var(--accent-red);">🔍 No hand detected — place hand in view</span>`;
      } else if (data.is_matching) {
        feedbackEl.innerHTML = `<span style="color: var(--accent-green);">✅ Correct gesture detected! Hold steady...</span>`;
      } else {
        feedbackEl.innerHTML = `<span style="color: var(--accent-amber);">⚠️ Camera sees: ${data.gesture} (Waiting for ${data.target_title || 'target'})</span>`;
      }
    }

    const holdPct = Math.round((data.progress || 0) * 100);
    if (progressEl) progressEl.style.width = `${holdPct}%`;
    if (holdTextEl) holdTextEl.innerText = `Hold gesture steady: ${holdPct}%`;
  }

  // Telemetry updates
  updateTelemetry(
    data.gesture || "NO HAND",
    data.handedness || "--",
    data.liveness_status || "CHECKING",
    data.similarity_score ? `${data.similarity_score}%` : "0.0%",
    data.candidate_name || "None"
  );

  // Authentication Outcome
  const banner = document.getElementById('decisionBanner');
  if (data.decision === "ACCESS GRANTED") {
    if (banner) {
      banner.className = "decision-banner decision-granted";
      banner.innerHTML = `✅ ACCESS GRANTED: ${data.candidate_name.toUpperCase()} (${data.similarity_score}%)`;
    }
    if (titleEl) titleEl.innerText = "Liveness Verified!";
    if (actionEl) actionEl.innerText = `Identity confirmed for ${data.candidate_name} with ${data.similarity_score}% confidence.`;
    if (feedbackEl) feedbackEl.innerHTML = `<span style="color: var(--accent-green);">🎉 Authentication successful!</span>`;
    isProcessingAuth = false;
    setTimeout(() => resetDecisionBanner(), 4500);
    loadLogsTable();
  } else if (data.decision === "ACCESS DENIED") {
    if (banner) {
      banner.className = "decision-banner decision-denied";
      banner.innerHTML = `❌ ACCESS DENIED: ${data.reason || "Verification Failed"}`;
    }
    if (titleEl) titleEl.innerText = "Access Denied";
    if (actionEl) actionEl.innerText = data.reason || "Liveness check failed or spoof suspected.";
    if (feedbackEl) feedbackEl.innerHTML = `<span style="color: var(--accent-red);">🚫 Verification failed</span>`;
    isProcessingAuth = false;
    setTimeout(() => resetDecisionBanner(), 4500);
    loadLogsTable();
  }
}

// ── Enrollment Cycle: 10-Frame High-Precision Vector Capture ────────────────
async function processEnrollCycle() {
  const base64Data = captureCleanFrame();
  if (!base64Data) return;

  const response = await fetch('/api/enroll-sample', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: base64Data })
  });
  if (!response.ok) return;

  const data = await response.json();
  if (data.landmarks && data.landmarks.length > 0) {
    drawBiometricHud(enrollHudCtx, data.landmarks, data.bbox, data.handedness || "Right");
  } else {
    if (enrollHudCtx) enrollHudCtx.clearRect(0, 0, 640, 480);
  }

  const feedbackEl = document.getElementById('enrollFeedback');
  const feedbackTextEl = document.getElementById('enrollFeedbackText');

  if (data.sample_accepted) {
    enrolledVectors.push(data.vector);
    enrollSamplesCollected++;

    const countEl = document.getElementById('enrollSampleCount');
    const barEl = document.getElementById('enrollProgressBar');
    if (countEl) countEl.innerText = `${enrollSamplesCollected} / ${ENROLL_TARGET_SAMPLES}`;
    if (barEl) barEl.style.width = `${(enrollSamplesCollected / ENROLL_TARGET_SAMPLES) * 100}%`;

    // Flash green feedback
    if (feedbackEl) {
      feedbackEl.className = 'enroll-feedback feedback-accepted';
      if (feedbackTextEl) feedbackTextEl.innerText = `✅ Sample ${enrollSamplesCollected}/${ENROLL_TARGET_SAMPLES} captured successfully!`;
      setTimeout(() => { feedbackEl.className = 'enroll-feedback'; }, 500);
    }

    if (enrollSamplesCollected >= ENROLL_TARGET_SAMPLES) {
      await finalizeEnrollment();
    }
  } else {
    // Show rejected reason
    if (feedbackEl && data.message) {
      feedbackEl.className = 'enroll-feedback feedback-rejected';
      if (feedbackTextEl) feedbackTextEl.innerText = `⚠️ ${data.message}`;
    }
  }
}

async function finalizeEnrollment() {
  isEnrolling = false;
  const name = document.getElementById('regName').value.trim();
  const email = document.getElementById('regEmail').value.trim();

  try {
    const res = await fetch('/api/enroll-complete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name, email: email, vectors: enrolledVectors })
    });
    const result = await res.json();
    if (res.ok) {
      alert(`✅ User '${name}' successfully enrolled with ${enrollSamplesCollected} biometric hand samples!`);
      document.getElementById('enrollProgressText').innerText = "Enrollment Complete!";
      document.getElementById('startEnrollBtn').disabled = false;
      document.getElementById('regName').value = "";
      document.getElementById('regEmail').value = "";
      loadUsersTable();
      switchTab('authTab');
    } else {
      alert(`Error enrolling: ${result.detail || "Server error"}`);
      document.getElementById('startEnrollBtn').disabled = false;
    }
  } catch (err) {
    alert("Network error finalizing enrollment.");
    document.getElementById('startEnrollBtn').disabled = false;
  }
}

function startEnrollment() {
  const name = document.getElementById('regName').value.trim();
  if (!name) {
    alert("Please enter a User Full Name.");
    return;
  }
  enrollSamplesCollected = 0;
  enrolledVectors = [];
  document.getElementById('enrollSampleCount').innerText = `0 / ${ENROLL_TARGET_SAMPLES}`;
  document.getElementById('enrollProgressBar').style.width = "0%";
  document.getElementById('enrollProgressText').innerText = "Please hold OPEN PALM steady in camera...";
  document.getElementById('startEnrollBtn').disabled = true;
  isEnrolling = true;
}

// ── Session Controls ────────────────────────────────────────────────────────
async function startAuthenticationSession() {
  try {
    const res = await fetch('/api/auth-start', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || "Cannot start authentication");
      return;
    }
    isProcessingAuth = true;
    document.getElementById('startAuthBtn').disabled = true;

    document.getElementById('challengeStepBadge').innerText = "STARTING...";
    document.getElementById('challengeTimer').innerText = "⏱️ 6.0s";
    document.getElementById('challengeTitle').innerText = "Follow On-Screen Challenge";
    document.getElementById('challengeActionText').innerText = "Look at the camera and perform the requested hand gestures.";
    document.getElementById('challengeCameraFeedback').innerText = "Initializing hand tracker...";
    document.getElementById('challengeProgress').style.width = "0%";
    document.getElementById('challengeHoldText').innerText = "Hold gesture steady: 0%";

    const banner = document.getElementById('decisionBanner');
    if (banner) {
      banner.className = "decision-banner decision-idle";
      banner.innerText = "AUTHENTICATION IN PROGRESS...";
    }
  } catch (err) {
    alert("Failed to connect to authentication backend.");
  }
}

function cancelAuthenticationSession() {
  isProcessingAuth = false;
  document.getElementById('startAuthBtn').disabled = false;
  resetDecisionBanner();
  fetch('/api/auth-cancel', { method: 'POST' }).catch(() => {});
}

function resetDecisionBanner() {
  const banner = document.getElementById('decisionBanner');
  if (banner) {
    banner.className = "decision-banner decision-idle";
    banner.innerText = "System Standby";
  }
  const btn = document.getElementById('startAuthBtn');
  if (btn) btn.disabled = false;

  const stepBadge = document.getElementById('challengeStepBadge');
  const timerBadge = document.getElementById('challengeTimer');
  const iconEl = document.getElementById('challengeIcon');
  const titleEl = document.getElementById('challengeTitle');
  const actionEl = document.getElementById('challengeActionText');
  const feedbackEl = document.getElementById('challengeCameraFeedback');
  const progressEl = document.getElementById('challengeProgress');
  const holdTextEl = document.getElementById('challengeHoldText');

  if (stepBadge) stepBadge.innerText = "STANDBY";
  if (timerBadge) timerBadge.innerText = "⏱️ --";
  if (iconEl) iconEl.innerText = "🖐️";
  if (titleEl) titleEl.innerText = "Ready to Authenticate";
  if (actionEl) actionEl.innerText = 'Click "Start Authentication" below to begin the dynamic liveness challenge.';
  if (feedbackEl) feedbackEl.innerHTML = "Waiting to start...";
  if (progressEl) progressEl.style.width = "0%";
  if (holdTextEl) holdTextEl.innerText = "Hold gesture steady: 0%";
}

function updateTelemetry(gesture, hand, liveness, similarity, candidate) {
  const g = document.getElementById('telemetryGesture');
  const h = document.getElementById('telemetryHand');
  const l = document.getElementById('telemetryLiveness');
  const s = document.getElementById('telemetrySimilarity');
  const c = document.getElementById('telemetryCandidate');

  if (g) g.innerText = gesture;
  if (h) h.innerText = hand;
  if (l) l.innerText = liveness;
  if (s) s.innerText = similarity;
  if (c) c.innerText = candidate;

  // Update animated gauge
  const pct = parseFloat(similarity) || 0;
  updateSimilarityGauge(pct);
}

// ── Tab Switching ───────────────────────────────────────────────────────────
function switchTab(tabId) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

  const panel = document.getElementById(tabId);
  if (panel) panel.classList.add('active');

  const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => {
    const oc = b.getAttribute('onclick');
    return oc && oc.includes(tabId);
  });
  if (activeBtn) activeBtn.classList.add('active');

  if (tabId === 'logsTab') {
    loadUsersTable();
    loadLogsTable();
  }
}

// ── Database & Audit Logs ───────────────────────────────────────────────────
async function loadUsersTable() {
  try {
    const res = await fetch('/api/users');
    const users = await res.json();
    const tbody = document.getElementById('usersTableBody');
    if (!tbody) return;
    if (!users || users.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No users enrolled yet. Go to "Enroll User" tab.</td></tr>`;
      return;
    }
    tbody.innerHTML = users.map(u => `
      <tr>
        <td>#${u.id}</td>
        <td style="font-weight: 600; color: #fff;">${u.name}</td>
        <td>${u.email || '--'}</td>
        <td><span class="tag-badge tag-pass">${u.sample_count || 1} samples</span></td>
        <td>${u.created_at}</td>
        <td>
          <button onclick="deleteUser(${u.id})" class="btn btn-danger" style="width: auto; padding: 0.25rem 0.65rem; font-size: 0.75rem;">
            🗑 Delete
          </button>
        </td>
      </tr>
    `).join('');
  } catch (err) {
    console.error("Failed to load users:", err);
  }
}

async function deleteUser(userId) {
  if (!confirm(`Are you sure you want to delete user #${userId}?`)) return;
  try {
    const res = await fetch(`/api/users/${userId}`, { method: 'DELETE' });
    if (res.ok) {
      loadUsersTable();
    } else {
      alert("Failed to delete user.");
    }
  } catch (err) {
    alert("Error communicating with server.");
  }
}

async function loadLogsTable() {
  try {
    const res = await fetch('/api/logs');
    const logs = await res.json();
    const tbody = document.getElementById('logsTableBody');
    if (!tbody) return;
    if (!logs || logs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No authentication attempts logged yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = logs.map(l => {
      const isGranted = l.access_result === "ACCESS GRANTED";
      return `
        <tr>
          <td>#${l.id}</td>
          <td>${l.timestamp}</td>
          <td style="font-weight: 600;">${l.user_name}</td>
          <td>
            <span class="tag-badge ${l.liveness_status === 'PASSED' ? 'tag-pass' : 'tag-fail'}">
              ${l.liveness_status}
            </span>
          </td>
          <td>${l.similarity_score}%</td>
          <td>
            <span class="tag-badge ${isGranted ? 'tag-pass' : 'tag-fail'}">
              ${l.access_result}
            </span>
          </td>
          <td style="font-size: 0.8rem; color: var(--text-muted);">${l.reason || '--'}</td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.error("Failed to load logs:", err);
  }
}

// ── Animated Similarity Arc Gauge ───────────────────────────────────────────
function updateSimilarityGauge(percentage) {
  const gaugeArc = document.getElementById('gaugeArc');
  const gaugeValue = document.getElementById('gaugeValue');
  if (!gaugeArc || !gaugeValue) return;

  // Compute the SVG path total length for stroke-dasharray animation
  const totalLen = gaugeArc.getTotalLength();
  const fillLen = (percentage / 100) * totalLen;

  gaugeArc.style.strokeDasharray = `${fillLen} ${totalLen}`;

  // Color coding based on score
  gaugeArc.classList.remove('gauge-high', 'gauge-medium', 'gauge-low');
  gaugeValue.classList.remove('gauge-high', 'gauge-medium', 'gauge-low');

  if (percentage >= 85) {
    gaugeArc.classList.add('gauge-high');
    gaugeValue.classList.add('gauge-high');
  } else if (percentage >= 60) {
    gaugeArc.classList.add('gauge-medium');
    gaugeValue.classList.add('gauge-medium');
  } else if (percentage > 0) {
    gaugeArc.classList.add('gauge-low');
    gaugeValue.classList.add('gauge-low');
  }

  gaugeValue.innerText = `${percentage.toFixed(1)}%`;
}

// ── Data Export ─────────────────────────────────────────────────────────────
function exportData(format) {
  const url = `/api/export?format=${format}`;
  if (format === 'csv') {
    // Trigger browser download
    const a = document.createElement('a');
    a.href = url;
    a.download = 'hand_auth_export.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } else {
    // Open JSON in new tab
    window.open(url, '_blank');
  }
}

// Initialize on DOM Ready
window.addEventListener('DOMContentLoaded', () => {
  initCamera();
});
