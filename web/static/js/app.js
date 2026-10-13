/**
 * Frontend Application Logic
 * Manages webcam feed, frame transmission to FastAPI backend,
 * real-time biometric telemetry, gesture challenge HUD, and database tables.
 */

let videoStream = null;
let isProcessingAuth = false;
let isEnrolling = false;
let enrollSamplesCollected = 0;
const ENROLL_TARGET_SAMPLES = 10;
let enrolledVectors = [];

const authVideo = document.getElementById('authVideo');
const authCanvas = document.getElementById('authCanvas');
const enrollCanvas = document.getElementById('enrollCanvas');
const authCtx = authCanvas.getContext('2d');
const enrollCtx = enrollCanvas.getContext('2d');

// Initialize Camera
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
    });
    videoStream = stream;
    authVideo.srcObject = stream;
    await authVideo.play();

    authCanvas.width = 640;
    authCanvas.height = 480;
    enrollCanvas.width = 640;
    enrollCanvas.height = 480;

    // Start background render loop
    requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error("Camera access error:", err);
    document.getElementById('systemStatus').innerHTML = `
      <span style="color: var(--accent-red);">❌ CAMERA ERROR</span>
    `;
    alert("Unable to access camera. Please allow webcam permissions.");
  }
}

// Continuous Frame Processing Loop
let lastFrameTime = 0;
async function renderLoop(timestamp) {
  if (videoStream && authVideo.readyState === authVideo.HAVE_ENOUGH_DATA) {
    // Draw raw camera frame to both canvases
    authCtx.drawImage(authVideo, 0, 0, authCanvas.width, authCanvas.height);
    enrollCtx.drawImage(authVideo, 0, 0, enrollCanvas.width, enrollCanvas.height);

    // Throttle backend API calls to ~10 FPS (every 100ms) to ensure smooth performance
    if (timestamp - lastFrameTime > 90) {
      lastFrameTime = timestamp;

      if (isProcessingAuth) {
        await processAuthCycle();
      } else if (isEnrolling) {
        await processEnrollCycle();
      } else {
        // Idle preview detection
        await processIdleCycle();
      }
    }
  }
  requestAnimationFrame(renderLoop);
}

// Helper: capture frame as base64 JPEG
function captureFrameBase64(canvas) {
  return canvas.toDataURL('image/jpeg', 0.85);
}

// Idle Mode: just detect hand & gesture for real-time responsiveness
async function processIdleCycle() {
  try {
    const base64Data = captureFrameBase64(authCanvas);
    const response = await fetch('/api/detect-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Data })
    });
    if (!response.ok) return;

    const data = await response.json();
    if (data.annotated_image) {
      renderAnnotatedImage(authCanvas, authCtx, data.annotated_image);
    }
    updateTelemetry(data.gesture || "NO HAND", data.handedness || "--", "IDLE", "0.0%", "None");
  } catch (err) {
    // Silent fail in idle loop
  }
}

// Authentication Cycle
async function processAuthCycle() {
  try {
    const base64Data = captureFrameBase64(authCanvas);
    const response = await fetch('/api/auth-step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Data })
    });
    if (!response.ok) return;

    const data = await response.json();

    if (data.annotated_image) {
      renderAnnotatedImage(authCanvas, authCtx, data.annotated_image);
    }

    // High-Visibility Guidance HUD Elements
    const stepBadge = document.getElementById('challengeStepBadge');
    const timerBadge = document.getElementById('challengeTimer');
    const iconEl = document.getElementById('challengeIcon');
    const titleEl = document.getElementById('challengeTitle');
    const actionEl = document.getElementById('challengeActionText');
    const feedbackEl = document.getElementById('challengeCameraFeedback');
    const progressEl = document.getElementById('challengeProgress');
    const holdTextEl = document.getElementById('challengeHoldText');

    if (data.liveness_status === "IN_PROGRESS" || data.liveness_status === "STEP_PASSED") {
      stepBadge.innerText = `STEP ${data.current_step || 1} OF ${data.total_steps || 2}`;
      timerBadge.innerText = `⏱️ ${(data.time_left !== undefined ? data.time_left : 6.0).toFixed(1)}s`;
      iconEl.innerText = data.target_icon || '🖐️';
      titleEl.innerText = data.target_title || 'Show Gesture';
      actionEl.innerText = data.target_action ? `👉 Action: ${data.target_action}` : data.instruction;

      // Real-time camera feedback
      if (!data.has_hand) {
        feedbackEl.innerHTML = `<span style="color: var(--accent-red);">🔍 No hand detected — place hand in view</span>`;
      } else if (data.is_matching) {
        feedbackEl.innerHTML = `<span style="color: var(--accent-green);">✅ Correct gesture detected! Hold steady...</span>`;
      } else {
        feedbackEl.innerHTML = `<span style="color: var(--accent-amber);">⚠️ Camera sees: ${data.gesture} (Waiting for ${data.target_title || 'target'})</span>`;
      }

      const holdPct = Math.round((data.progress || 0) * 100);
      progressEl.style.width = `${holdPct}%`;
      holdTextEl.innerText = `Hold gesture steady: ${holdPct}%`;
    }

    // Update telemetry metrics
    updateTelemetry(
      data.gesture || "NO HAND",
      data.handedness || "--",
      data.liveness_status || "CHECKING",
      data.similarity_score ? `${data.similarity_score}%` : "0.0%",
      data.candidate_name || "None"
    );

    // Check completion
    const banner = document.getElementById('decisionBanner');
    if (data.decision === "ACCESS GRANTED") {
      banner.className = "decision-banner decision-granted";
      banner.innerHTML = `✅ ACCESS GRANTED: ${data.candidate_name.toUpperCase()} (${data.similarity_score}%)`;
      titleEl.innerText = "Liveness Verified!";
      actionEl.innerText = `Identity confirmed for ${data.candidate_name} with ${data.similarity_score}% confidence.`;
      feedbackEl.innerHTML = `<span style="color: var(--accent-green);">🎉 Authentication successful!</span>`;
      isProcessingAuth = false;
      setTimeout(() => resetDecisionBanner(), 4500);
      loadLogsTable();
    } else if (data.decision === "ACCESS DENIED") {
      banner.className = "decision-banner decision-denied";
      banner.innerHTML = `❌ ACCESS DENIED: ${data.reason || "Verification Failed"}`;
      titleEl.innerText = "Access Denied";
      actionEl.innerText = data.reason || "Liveness check failed or spoof suspected.";
      feedbackEl.innerHTML = `<span style="color: var(--accent-red);">🚫 Verification failed</span>`;
      isProcessingAuth = false;
      setTimeout(() => resetDecisionBanner(), 4500);
      loadLogsTable();
    }
  } catch (err) {
    console.error("Auth cycle error:", err);
  }
}


// Enrollment Cycle
async function processEnrollCycle() {
  try {
    const base64Data = captureFrameBase64(enrollCanvas);
    const response = await fetch('/api/enroll-sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Data })
    });
    if (!response.ok) return;

    const data = await response.json();
    if (data.annotated_image) {
      renderAnnotatedImage(enrollCanvas, enrollCtx, data.annotated_image);
    }

    if (data.sample_accepted) {
      enrolledVectors.push(data.vector);
      enrollSamplesCollected++;

      document.getElementById('enrollSampleCount').innerText = `${enrollSamplesCollected}/${ENROLL_TARGET_SAMPLES}`;
      document.getElementById('enrollProgressBar').style.width = `${(enrollSamplesCollected / ENROLL_TARGET_SAMPLES) * 100}%`;

      if (enrollSamplesCollected >= ENROLL_TARGET_SAMPLES) {
        // Complete registration
        await finalizeEnrollment();
      }
    }
  } catch (err) {
    console.error("Enroll cycle error:", err);
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
      alert(`✅ User '${name}' successfully enrolled with ${enrollSamplesCollected} hand samples!`);
      document.getElementById('enrollProgressText').innerText = "Enrollment Complete!";
      document.getElementById('startEnrollBtn').disabled = false;
      document.getElementById('regName').value = "";
      document.getElementById('regEmail').value = "";
      loadUsersTable();
      switchTab('authTab');
    } else {
      alert(`Error enrolling: ${result.detail || "Server error"}`);
    }
  } catch (err) {
    alert("Network error finalizing enrollment.");
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
  document.getElementById('enrollSampleCount').innerText = `0/${ENROLL_TARGET_SAMPLES}`;
  document.getElementById('enrollProgressBar').style.width = "0%";
  document.getElementById('enrollProgressText').innerText = "Please hold OPEN PALM steady in camera...";
  document.getElementById('startEnrollBtn').disabled = true;
  isEnrolling = true;
}

// Start Authentication
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

    // Immediately display active challenge guidance
    document.getElementById('challengeStepBadge').innerText = "STARTING...";
    document.getElementById('challengeTimer').innerText = "⏱️ 6.0s";
    document.getElementById('challengeTitle').innerText = "Follow On-Screen Challenge";
    document.getElementById('challengeActionText').innerText = "Look at the camera and perform the requested hand gestures.";
    document.getElementById('challengeCameraFeedback').innerText = "Initializing hand tracker...";
    document.getElementById('challengeProgress').style.width = "0%";
    document.getElementById('challengeHoldText').innerText = "Hold gesture steady: 0%";

    const banner = document.getElementById('decisionBanner');
    banner.className = "decision-banner decision-idle";
    banner.innerText = "AUTHENTICATION IN PROGRESS...";
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
  banner.className = "decision-banner decision-idle";
  banner.innerText = "System Standby";
  document.getElementById('startAuthBtn').disabled = false;

  document.getElementById('challengeStepBadge').innerText = "STANDBY";
  document.getElementById('challengeTimer').innerText = "⏱️ --";
  document.getElementById('challengeIcon').innerText = "🖐️";
  document.getElementById('challengeTitle').innerText = "Ready to Authenticate";
  document.getElementById('challengeActionText').innerText = 'Click "Start Authentication" below to begin the dynamic liveness challenge.';
  document.getElementById('challengeCameraFeedback').innerText = "Waiting to start...";
  document.getElementById('challengeProgress').style.width = "0%";
  document.getElementById('challengeHoldText').innerText = "Hold gesture steady: 0%";
}

function renderAnnotatedImage(canvas, ctx, base64Str) {
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.src = "data:image/jpeg;base64," + base64Str;
}

function updateTelemetry(gesture, hand, liveness, similarity, candidate) {
  document.getElementById('telemetryGesture').innerText = gesture;
  document.getElementById('telemetryHand').innerText = hand;
  document.getElementById('telemetryLiveness').innerText = liveness;
  document.getElementById('telemetrySimilarity').innerText = similarity;
  document.getElementById('telemetryCandidate').innerText = candidate;
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

  document.getElementById(tabId).classList.add('active');
  const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
  if (activeBtn) activeBtn.classList.add('active');

  if (tabId === 'logsTab') {
    loadUsersTable();
    loadLogsTable();
  }
}

// Fetch Users & Logs Tables
async function loadUsersTable() {
  try {
    const res = await fetch('/api/users');
    const users = await res.json();
    const tbody = document.getElementById('usersTableBody');
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
          <button onclick="deleteUser(${u.id})" class="btn btn-secondary" style="width: auto; padding: 0.2rem 0.5rem; font-size: 0.75rem; color: var(--accent-red);">
            Delete
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

// Initialize on DOM Load
window.addEventListener('DOMContentLoaded', () => {
  initCamera();
});
