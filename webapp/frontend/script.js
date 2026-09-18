const API_BASE = window.location.origin;

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const previewArea = document.getElementById('preview-area');
const previewImage = document.getElementById('preview-image');
const resetBtn = document.getElementById('reset-btn');
const resultCard = document.getElementById('result-card');
const resultRing = document.getElementById('result-ring');
const confidenceValue = document.getElementById('confidence-value');
const resultBadge = document.getElementById('result-badge');
const resultDesc = document.getElementById('result-desc');
const statusLine = document.getElementById('status-line');

// --- Upload interactions -----------------------------------------------
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

['dragenter', 'dragover'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });
});

['dragleave', 'drop'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
  });
});

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

resetBtn.addEventListener('click', () => {
  fileInput.value = '';
  previewArea.hidden = true;
  dropzone.hidden = false;
  resultCard.hidden = true;
  statusLine.hidden = true;
});

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    dropzone.hidden = true;
    previewArea.hidden = false;
    resultCard.hidden = true;
  };
  reader.readAsDataURL(file);

  predict(file);
}

async function predict(file) {
  statusLine.hidden = false;
  statusLine.textContent = 'Analyzing panel...';

  const formData = new FormData();
  formData.append('image', file);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      statusLine.textContent = data.error || 'Something went wrong.';
      return;
    }

    statusLine.hidden = true;
    showResult(data.label, data.confidence);
  } catch (err) {
    statusLine.textContent = 'Could not reach the server. Is server.py running?';
  }
}

function showResult(label, confidence) {
  resultCard.hidden = false;

  const isClean = label === 'clean';
  const ringColor = isClean ? 'var(--clean)' : 'var(--dusty)';

  resultRing.style.setProperty('--pct', confidence);
  resultRing.style.setProperty('--ring-color', ringColor);
  confidenceValue.textContent = `${confidence}%`;

  resultBadge.textContent = isClean ? 'Clean' : 'Dusty';
  resultBadge.className = `result-badge ${isClean ? 'clean' : 'dusty'}`;

  resultDesc.textContent = isClean
    ? 'This panel looks clear — no cleaning needed right now.'
    : 'Soiling detected — scheduling a cleaning could recover lost output.';
}

// --- Results / metrics ---------------------------------------------------
async function loadResults() {
  try {
    const res = await fetch(`${API_BASE}/api/results`);
    const data = await res.json();

    if (data.metrics) {
      const m = data.metrics;
      document.getElementById('metric-accuracy').textContent =
        `${(m.accuracy * 100).toFixed(1)}%`;
      document.getElementById('metric-clean-precision').textContent =
        `${(m.clean.precision * 100).toFixed(1)}%`;
      document.getElementById('metric-dusty-precision').textContent =
        `${(m.dusty.precision * 100).toFixed(1)}%`;
      document.getElementById('metric-f1').textContent =
        `${(m['macro avg']['f1-score'] * 100).toFixed(1)}%`;
    }

    const imageMap = {
      training_curve_initial: 'plot-initial',
      training_curve_finetune: 'plot-finetune',
      confusion_matrix: 'plot-confusion',
    };

    let anyImages = false;
    for (const [key, elId] of Object.entries(imageMap)) {
      const el = document.getElementById(elId);
      if (data.images[key]) {
        el.src = `${API_BASE}${data.images[key]}`;
        el.closest('.plot-panel').hidden = false;
        anyImages = true;
      } else {
        el.closest('.plot-panel').hidden = true;
      }
    }

    document.getElementById('results-empty').hidden = anyImages || !!data.metrics;
  } catch (err) {
    document.getElementById('results-empty').hidden = false;
  }
}

loadResults();
