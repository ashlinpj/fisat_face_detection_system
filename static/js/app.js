/* ===================================================================
   College Canteen Face Detection System – Web GUI front-end logic
   =================================================================== */

// ---- helpers ----
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function toast(msg, ms = 3000) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => (el.hidden = true), 300);
  }, ms);
}

async function api(url, opts = {}) {
  try {
    const res = await fetch(url, opts);
    return await res.json();
  } catch (e) {
    console.error(url, e);
    toast("Request failed – check console");
    return null;
  }
}

// ---- tabs ----
$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    $$(".tab-content").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");

    // Lazy-load data when switching tabs
    const tab = btn.dataset.tab;
    if (tab === "students") loadStudents();
    if (tab === "logs") loadLogs();
    if (tab === "stats") loadStats();
  });
});

// ==================================================================
// DETECTION TAB
// ==================================================================
let detectionRunning = false;
let pollTimer = null;

$("#btn-toggle").addEventListener("click", async () => {
  if (!detectionRunning) {
    const r = await api("/api/detection/start", { method: "POST" });
    if (r && r.ok) {
      detectionRunning = true;
      $("#btn-toggle").textContent = "⏹ Stop Detection";
      startPolling();
      toast("Detection started");
    } else {
      toast(r?.error || "Could not start detection");
    }
  } else {
    await api("/api/detection/stop", { method: "POST" });
    detectionRunning = false;
    $("#btn-toggle").textContent = "▶ Start Detection";
    stopPolling();
    toast("Detection stopped");
  }
});

function startPolling() {
  stopPolling();
  pollTimer = setInterval(pollDetection, 1000);
}
function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

async function pollDetection() {
  const r = await api("/api/detection/info");
  if (!r) return;

  // Status badge
  const badge = $("#system-status");
  const s = r.status;
  badge.className = "status-badge status-" + s;
  const labels = {
    initializing: "⚪ Initializing…",
    ready: "🟢 System Ready",
    running: "🟢 Detection Running",
    stopped: "⚪ Detection Stopped",
    error: "🔴 Error",
  };
  badge.textContent = labels[s] || s;

  // Detection info
  if (r.info) {
    $("#det-faces").textContent = r.info.faces;
    $("#det-known").textContent = r.info.known;
    $("#det-unknown").textContent = r.info.unknown;
    $("#det-fps").textContent = r.info.fps;
  }

  // Recent detections
  const ul = $("#recent-list");
  ul.innerHTML = "";
  (r.recent || []).forEach((d) => {
    const li = document.createElement("li");
    li.textContent = `${d.time} – ${d.name}`;
    ul.appendChild(li);
  });

  // Today summary
  if (r.today) {
    $("#today-visits").textContent = r.today.total_visits;
    $("#today-unique").textContent = r.today.unique_visitors;
  }
}

// Poll status even before starting detection (get initial status)
setInterval(async () => {
  const r = await api("/api/detection/info");
  if (!r) return;
  const badge = $("#system-status");
  const s = r.status;
  badge.className = "status-badge status-" + s;
  const labels = {
    initializing: "⚪ Initializing…",
    ready: "🟢 System Ready",
    running: "🟢 Detection Running",
    stopped: "⚪ Detection Stopped",
    error: "🔴 Error",
  };
  badge.textContent = labels[s] || s;
}, 3000);

// Screenshot
$("#btn-screenshot").addEventListener("click", async () => {
  const r = await api("/api/screenshot", { method: "POST" });
  if (r && r.ok) toast(`Screenshot saved: ${r.filename}`);
  else toast(r?.error || "No frame to capture");
});

// ==================================================================
// REGISTER – from camera capture (multi-pose guided flow)
// ==================================================================
let poseScript = [];
let poseIndex = 0;
let captureStudentData = {};

$("#btn-register-capture").addEventListener("click", () => {
  if (!detectionRunning) {
    toast("Start detection first!");
    return;
  }
  // Reset phases
  $("#capture-phase-info").hidden = false;
  $("#capture-phase-poses").hidden = true;
  $("#capture-phase-done").hidden = true;
  $("#modal-register-capture").hidden = false;
});

// Phase 1 -> Phase 2: start pose capture
$("#form-capture-info").addEventListener("submit", async (e) => {
  e.preventDefault();
  const sid = $("#cap-student-id").value.trim();
  const name = $("#cap-name").value.trim();
  if (!sid || !name) { toast("Student ID and Name are required"); return; }

  captureStudentData = {
    student_id: sid,
    name: name,
    department: $("#cap-department").value.trim(),
    year: parseInt($("#cap-year").value),
  };

  const glasses = $("#cap-glasses").checked ? "1" : "0";
  poseScript = await api(`/api/pose_script?glasses=${glasses}`);
  if (!poseScript || !poseScript.length) { toast("Could not load pose script"); return; }

  // Start session on server
  await api("/api/capture_session/start", { method: "POST" });

  poseIndex = 0;
  showPoseStep();
  $("#capture-phase-info").hidden = true;
  $("#capture-phase-poses").hidden = false;
});

function showPoseStep() {
  const total = poseScript.length;
  const pose = poseScript[poseIndex];
  $("#pose-step").textContent = `Step ${poseIndex + 1}/${total}`;
  $("#pose-bar").style.width = `${((poseIndex) / total) * 100}%`;
  $("#pose-title").textContent = pose.title;
  $("#pose-tip").textContent = pose.tip;
  $("#pose-status").textContent = "";
}

async function snapAndAdvance() {
  $("#pose-status").textContent = "Capturing…";
  // Small delay to let user settle
  await new Promise((r) => setTimeout(r, 400));

  const r = await api("/api/capture_session/snap", { method: "POST" });
  if (r && r.ok) {
    $("#pose-status").textContent = `✓ Captured (${r.captured} total)`;
  } else {
    $("#pose-status").textContent = "⚠ Could not capture – skipped";
  }

  poseIndex++;
  if (poseIndex < poseScript.length) {
    setTimeout(showPoseStep, 600);
  } else {
    // All poses done -> submit registration
    await submitMultiCapture();
  }
}

$("#btn-pose-snap").addEventListener("click", snapAndAdvance);
$("#btn-pose-skip").addEventListener("click", () => {
  $("#pose-status").textContent = "Skipped";
  poseIndex++;
  if (poseIndex < poseScript.length) {
    setTimeout(showPoseStep, 300);
  } else {
    submitMultiCapture();
  }
});

async function submitMultiCapture() {
  $("#capture-phase-poses").hidden = true;
  $("#capture-phase-done").hidden = false;

  const r = await api("/api/students/register_capture", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(captureStudentData),
  });

  if (r && r.ok) {
    toast(`Student registered with ${r.samples} samples!`);
    $("#modal-register-capture").hidden = true;
    // Reset form
    $("#cap-student-id").value = "";
    $("#cap-name").value = "";
    $("#cap-department").value = "";
    $("#cap-year").value = "1";
    loadStudents();
  } else {
    toast(r?.error || "Registration failed");
    // Go back to pose phase so user can retry
    $("#capture-phase-done").hidden = true;
    $("#capture-phase-info").hidden = false;
  }
}

// ==================================================================
// REGISTER – from image upload
// ==================================================================
$("#btn-register-image").addEventListener("click", () => {
  $("#modal-register-image").hidden = false;
});

$("#form-register-image").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const r = await api("/api/students/register_image", {
    method: "POST",
    body: fd,
  });
  if (r && r.ok) {
    toast("Student registered!");
    $("#modal-register-image").hidden = true;
    e.target.reset();
    loadStudents();
  } else {
    toast(r?.error || "Registration failed");
  }
});

// Modal close buttons
$$(".modal-close").forEach((btn) => {
  btn.addEventListener("click", () => {
    btn.closest(".modal-overlay").hidden = true;
  });
});
// Close modals on overlay click
$$(".modal-overlay").forEach((overlay) => {
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.hidden = true;
  });
});

// ==================================================================
// STUDENTS TAB
// ==================================================================
let selectedStudentId = null;

async function loadStudents() {
  const search = ($("#student-search").value || "").trim();
  const qs = search ? `?search=${encodeURIComponent(search)}` : "";
  const data = await api(`/api/students${qs}`);
  if (!data) return;

  const tbody = $("#students-table tbody");
  tbody.innerHTML = "";
  selectedStudentId = null;

  data.forEach((s) => {
    const tr = document.createElement("tr");
    tr.dataset.sid = s.student_id;
    tr.innerHTML = `<td>${s.id}</td><td>${s.student_id}</td><td>${s.name}</td><td>${s.department}</td><td>${s.year}</td><td>${s.created_at}</td>`;
    tr.addEventListener("click", () => {
      $$("#students-table tbody tr").forEach((r) => r.classList.remove("selected"));
      tr.classList.add("selected");
      selectedStudentId = s.student_id;
    });
    tbody.appendChild(tr);
  });
}

$("#btn-refresh-students").addEventListener("click", loadStudents);

let searchDebounce;
$("#student-search").addEventListener("input", () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(loadStudents, 300);
});

$("#btn-delete-student").addEventListener("click", async () => {
  if (!selectedStudentId) {
    toast("Select a student first");
    return;
  }
  if (!confirm(`Delete student ${selectedStudentId}?`)) return;
  await api(`/api/students/${selectedStudentId}`, { method: "DELETE" });
  toast("Student deleted");
  loadStudents();
});

// ==================================================================
// LOGS TAB
// ==================================================================
// Set today as default
$("#log-date").value = new Date().toISOString().slice(0, 10);

async function loadLogs() {
  const date = $("#log-date").value || "";
  const sid = ($("#log-student").value || "").trim();
  const params = new URLSearchParams();
  if (date) params.set("date", date);
  if (sid) params.set("student_id", sid);

  const data = await api(`/api/logs?${params}`);
  if (!data) return;

  const tbody = $("#logs-table tbody");
  tbody.innerHTML = "";
  data.forEach((l) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${l.id}</td><td>${l.date}</td><td>${l.entry_time}</td><td>${l.student_id}</td><td>${l.student_name}</td><td>${l.status}</td><td>${l.duration}</td>`;
    tbody.appendChild(tr);
  });
}

$("#btn-filter-logs").addEventListener("click", loadLogs);
$("#btn-clear-logs").addEventListener("click", () => {
  $("#log-date").value = new Date().toISOString().slice(0, 10);
  $("#log-student").value = "";
  loadLogs();
});
$("#btn-export-csv").addEventListener("click", () => {
  const date = $("#log-date").value || "";
  const sid = ($("#log-student").value || "").trim();
  const params = new URLSearchParams();
  if (date) params.set("date", date);
  if (sid) params.set("student_id", sid);
  window.location.href = `/api/logs/export?${params}`;
});

// ==================================================================
// STATISTICS TAB
// ==================================================================
async function loadStats() {
  const r = await api("/api/statistics");
  if (!r) return;

  $("#stat-students").textContent = r.total_students;
  $("#stat-today").textContent = r.today.total_visits;
  $("#stat-unique").textContent = r.today.unique_visitors;
  $("#stat-unknown").textContent = r.today.unknown_visitors;

  const now = new Date().toLocaleString();
  $("#stats-detail").textContent =
    `╔════════════════════════════════════════════════════════════╗\n` +
    `║       CANTEEN FACE DETECTION – STATISTICS REPORT          ║\n` +
    `╠════════════════════════════════════════════════════════════╣\n` +
    `║  Report Generated: ${now.padEnd(39)}║\n` +
    `╠════════════════════════════════════════════════════════════╣\n` +
    `║  TODAY'S SUMMARY (${r.today.date})                          ║\n` +
    `╠════════════════════════════════════════════════════════════╣\n` +
    `║  • Total Visits:        ${String(r.today.total_visits).padEnd(35)}║\n` +
    `║  • Unique Visitors:     ${String(r.today.unique_visitors).padEnd(35)}║\n` +
    `║  • Unknown Visitors:    ${String(r.today.unknown_visitors).padEnd(35)}║\n` +
    `║  • Avg Duration:        ${String(r.today.average_duration_minutes).padEnd(32)}min ║\n` +
    `╠════════════════════════════════════════════════════════════╣\n` +
    `║  REGISTERED STUDENTS                                      ║\n` +
    `╠════════════════════════════════════════════════════════════╣\n` +
    `║  • Total Registered:    ${String(r.total_students).padEnd(35)}║\n` +
    `╚════════════════════════════════════════════════════════════╝`;
}

$("#btn-refresh-stats").addEventListener("click", loadStats);

// ==================================================================
// Initial data load
// ==================================================================
loadStudents();
loadLogs();
loadStats();
