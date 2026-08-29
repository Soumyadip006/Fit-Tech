// ============================================================
// FORMIQ — FRONTEND JAVASCRIPT APP LOGIC
// ============================================================

const API_BASE = 'http://127.0.0.1:5000/api';

// DOM Elements
const serverStatusDot = document.getElementById('serverStatusDot');
const serverStatusText = document.getElementById('serverStatusText');

const samplePills = document.getElementById('samplePills');
const dropzone = document.getElementById('dropzone');
const videoFileInput = document.getElementById('videoFileInput');
const browseBtn = document.querySelector('.browse-btn');

const videoPreviewCard = document.getElementById('videoPreviewCard');
const videoFileName = document.getElementById('videoFileName');
const videoPlayer = document.getElementById('videoPlayer');
const changeVideoBtn = document.getElementById('changeVideoBtn');

const analyzeBtn = document.getElementById('analyzeBtn');

const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const resultsContent = document.getElementById('resultsContent');

// State
let selectedFile = null;
let selectedSampleName = null;

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  fetchSampleVideos();
  setupEventListeners();
});

// 1. API Health Check
async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      serverStatusDot.classList.add('pulse');
      serverStatusText.textContent = 'Backend Online';
      serverStatusText.style.color = '#10B981';
    } else {
      throw new Error();
    }
  } catch (err) {
    serverStatusDot.classList.remove('pulse');
    serverStatusDot.style.backgroundColor = '#EF4444';
    serverStatusText.textContent = 'Backend Offline';
    serverStatusText.style.color = '#EF4444';
  }
}

// 2. Fetch Sample Videos
async function fetchSampleVideos() {
  try {
    const res = await fetch(`${API_BASE}/samples`);
    if (!res.ok) return;
    const data = await res.json();
    
    samplePills.innerHTML = '';
    if (data.samples && data.samples.length > 0) {
      data.samples.forEach(sample => {
        const btn = document.createElement('button');
        btn.className = 'pill';
        btn.textContent = sample;
        btn.onclick = () => selectSampleVideo(sample, btn);
        samplePills.appendChild(btn);
      });
    } else {
      samplePills.innerHTML = '<span style="font-size:0.8rem;color:#64748B;">No sample videos found</span>';
    }
  } catch (err) {
    samplePills.innerHTML = '<span style="font-size:0.8rem;color:#64748B;">Could not load samples</span>';
  }
}

// 3. Setup Drag & Drop and File Events
function setupEventListeners() {
  // Dropzone dragover & leave
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    }, false);
  });

  // Handle drop
  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('video/')) {
      handleFileSelection(files[0]);
    }
  });

  // Handle file input change
  videoFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    videoFileInput.click();
  });

  changeVideoBtn.addEventListener('click', () => {
    resetVideoSelection();
  });

  // Analyze Button click
  analyzeBtn.addEventListener('click', () => {
    runBiomechanicalAnalysis();
  });
}

// Handle User Uploaded File Selection
function handleFileSelection(file) {
  selectedFile = file;
  selectedSampleName = null;

  // Clear pill highlights
  document.querySelectorAll('.sample-pills .pill').forEach(p => p.classList.remove('active'));

  // Update UI
  videoFileName.textContent = file.name;
  const objectUrl = URL.createObjectURL(file);
  videoPlayer.src = objectUrl;

  dropzone.classList.add('hidden');
  videoPreviewCard.classList.remove('hidden');
  analyzeBtn.disabled = false;
}

// Handle Sample Video Selection
function selectSampleVideo(sampleName, pillElement) {
  selectedFile = null;
  selectedSampleName = sampleName;

  // Highlight pill
  document.querySelectorAll('.sample-pills .pill').forEach(p => p.classList.remove('active'));
  pillElement.classList.add('active');

  // Update UI
  videoFileName.textContent = `[Sample] ${sampleName}`;
  videoPlayer.src = `${API_BASE}/sample_video/${sampleName}`;

  dropzone.classList.add('hidden');
  videoPreviewCard.classList.remove('hidden');
  analyzeBtn.disabled = false;
}

// Reset Selection
function resetVideoSelection() {
  selectedFile = null;
  selectedSampleName = null;
  videoFileInput.value = '';

  document.querySelectorAll('.sample-pills .pill').forEach(p => p.classList.remove('active'));

  videoPlayer.pause();
  videoPlayer.src = '';

  videoPreviewCard.classList.add('hidden');
  dropzone.classList.remove('hidden');
  analyzeBtn.disabled = true;

  // Reset Results view
  resultsContent.classList.add('hidden');
  loadingState.classList.add('hidden');
  emptyState.classList.remove('hidden');
}

// 4. Run Analysis via Flask REST API
async function runBiomechanicalAnalysis() {
  if (!selectedFile && !selectedSampleName) return;

  // UI state transition to Loading
  emptyState.classList.add('hidden');
  resultsContent.classList.add('hidden');
  loadingState.classList.remove('hidden');
  analyzeBtn.disabled = true;

  const formData = new FormData();
  if (selectedFile) {
    formData.append('video', selectedFile);
  } else if (selectedSampleName) {
    formData.append('sample_name', selectedSampleName);
  }

  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (response.ok && !data.error) {
      renderAnalysisResults(data);
    } else {
      alert(`Analysis Error: ${data.error || 'Could not analyze video'}`);
      loadingState.classList.add('hidden');
      emptyState.classList.remove('hidden');
    }
  } catch (err) {
    alert('Server connection error. Make sure backend is running on http://127.0.0.1:5000');
    loadingState.classList.add('hidden');
    emptyState.classList.remove('hidden');
  } finally {
    analyzeBtn.disabled = false;
  }
}

// 5. Render Analysis Results Payload
function renderAnalysisResults(data) {
  loadingState.classList.add('hidden');
  resultsContent.classList.remove('hidden');

  // Exercise Banner
  document.getElementById('detectedExerciseTitle').textContent = data.detected_exercise;
  document.getElementById('framesAnalyzedTag').textContent = `${data.total_frames} Frames Analyzed`;
  document.getElementById('confidenceScoreVal').textContent = data.confidence_score;

  // Score Gauge Color
  const scoreGaugeCircle = document.querySelector('.gauge-circle');
  if (data.confidence_score >= 80) {
    scoreGaugeCircle.style.borderColor = '#10B981';
    scoreGaugeCircle.style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.4)';
  } else if (data.confidence_score >= 50) {
    scoreGaugeCircle.style.borderColor = '#F59E0B';
    scoreGaugeCircle.style.boxShadow = '0 0 20px rgba(245, 158, 11, 0.4)';
  } else {
    scoreGaugeCircle.style.borderColor = '#EF4444';
    scoreGaugeCircle.style.boxShadow = '0 0 20px rgba(239, 68, 68, 0.4)';
  }

  // Verdict Card
  const verdictCard = document.getElementById('verdictCard');
  const verdictBadge = document.getElementById('verdictBadge');
  const verdictHeadline = document.getElementById('verdictHeadline');
  const verdictBody = document.getElementById('verdictBody');
  const evaluationsGrid = document.getElementById('evaluationsGrid');

  if (data.is_poor_form) {
    verdictCard.classList.add('poor-form');
    verdictBadge.textContent = '⚠️ POOR FORM DETECTED';
    verdictHeadline.textContent = 'Spine Rounding / Technique Defect';
  } else {
    verdictCard.classList.remove('poor-form');
    verdictBadge.textContent = '✓ FORM VERDICT';
    verdictHeadline.textContent = 'Mechanics Evaluation';
  }

  verdictBody.textContent = data.form_feedback.verdict || 'Analysis completed.';

  // Evaluations Grid
  evaluationsGrid.innerHTML = '';
  if (data.form_feedback.evaluations) {
    data.form_feedback.evaluations.forEach(ev => {
      const div = document.createElement('div');
      div.className = 'eval-item';
      div.innerHTML = `<strong>${ev.name}</strong>: ${ev.status}`;
      evaluationsGrid.appendChild(div);
    });
  }

  // Key Measurements Grid
  const m = data.key_measurements;

  document.getElementById('mKneeAngle').textContent = `${m.knee_angle.avg}°`;
  document.getElementById('mKneeSub').textContent = `Min: ${m.knee_angle.min}° | Range: ${m.knee_angle.range}°`;

  document.getElementById('mKneeAsymmetry').textContent = `${m.knee_asymmetry}°`;

  document.getElementById('mHipAngle').textContent = `${m.hip_angle.avg}°`;
  document.getElementById('mHipSub').textContent = `Min: ${m.hip_angle.min}° | Range: ${m.hip_angle.range}°`;

  document.getElementById('mElbowAngle').textContent = `${m.elbow_angle.avg}°`;
  document.getElementById('mElbowSub').textContent = `Min: ${m.elbow_angle.min}° | Range: ${m.elbow_angle.range}°`;

  document.getElementById('mSpineAngle').textContent = `${m.spine_alignment.avg}°`;
  const mSpineStatus = document.getElementById('mSpineStatus');
  mSpineStatus.textContent = m.spine_alignment.status;

  const spineMetricCard = document.getElementById('spineMetricCard');
  if (m.spine_alignment.avg < 162) {
    mSpineStatus.style.color = '#EF4444';
    spineMetricCard.style.borderColor = 'rgba(239, 68, 68, 0.5)';
    spineMetricCard.style.background = 'rgba(239, 68, 68, 0.1)';
  } else {
    mSpineStatus.style.color = '#10B981';
    spineMetricCard.style.borderColor = 'rgba(99, 102, 241, 0.3)';
    spineMetricCard.style.background = 'rgba(99, 102, 241, 0.1)';
  }

  document.getElementById('mTorsoLean').textContent = `${m.torso_position.lean}° Lean`;
  document.getElementById('mTorsoVert').textContent = `Vertical Offset: ${m.torso_position.vertical}`;

  document.getElementById('mHipKneeRatio').textContent = `${m.hip_knee_rom_ratio}`;
  document.getElementById('mHipKneeSub').textContent = m.hip_knee_rom_ratio > 1.2 ? '>1.2 Hip-Dominant' : (m.hip_knee_rom_ratio < 0.8 ? '<0.8 Knee-Dominant' : 'Balanced');

  document.getElementById('mWristAbove').textContent = `${m.wrist_above_pct}%`;

  // Exercise Scores Chart
  const chartBars = document.getElementById('chartBars');
  chartBars.innerHTML = '';

  if (data.all_scores) {
    data.all_scores.forEach(([exName, score]) => {
      const row = document.createElement('div');
      const isWinner = exName === data.detected_exercise;
      row.className = `bar-row ${isWinner ? 'winner' : ''}`;

      const pct = Math.max(0, Math.min(100, score));

      row.innerHTML = `
        <span class="bar-label">${exName}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width: 0%"></div>
        </div>
        <span class="bar-score">${score}</span>
      `;

      chartBars.appendChild(row);

      // Animate bar width
      setTimeout(() => {
        row.querySelector('.bar-fill').style.width = `${pct}%`;
      }, 50);
    });
  }
}
