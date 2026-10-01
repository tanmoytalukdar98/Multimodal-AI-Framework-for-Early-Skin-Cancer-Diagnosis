const DISPLAY = {
  akiec: "Actinic Keratoses / Intraepithelial Carcinoma",
  bcc: "Basal Cell Carcinoma",
  bkl: "Benign Keratosis",
  df: "Dermatofibroma",
  mel: "Melanoma",
  nv: "Melanocytic Nevus",
  vasc: "Vascular Lesion",
};

const CATEGORY = {
  akiec: "Suspicious / Precancerous",
  bcc: "Malignant",
  bkl: "Benign",
  df: "Benign",
  mel: "Malignant",
  nv: "Benign",
  vasc: "Benign",
};

const STAGES = [
  "Image preprocessing",
  "Visual feature analysis",
  "Clinical feature analysis",
  "Multimodal fusion",
  "Classification",
  "Confidence calibration",
];

const els = {
  home: document.getElementById("view-home"),
  analysis: document.getElementById("view-analysis"),
  analyzing: document.getElementById("view-analyzing"),
  results: document.getElementById("view-results"),
  startBtn: document.getElementById("start-btn"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  dropEmpty: document.getElementById("drop-empty"),
  dropPreview: document.getElementById("drop-preview"),
  previewImg: document.getElementById("preview-img"),
  fileMeta: document.getElementById("file-meta"),
  imageStatus: document.getElementById("image-status"),
  replaceBtn: document.getElementById("replace-btn"),
  removeBtn: document.getElementById("remove-btn"),
  age: document.getElementById("age"),
  sex: document.getElementById("sex"),
  location: document.getElementById("location"),
  analyzeBtn: document.getElementById("analyze-btn"),
  resetBtn: document.getElementById("reset-btn"),
  diagnosis: document.getElementById("diagnosis"),
  confidence: document.getElementById("confidence"),
  categoryValue: document.getElementById("category-value"),
  categoryNote: document.getElementById("category-note"),
  categoryBadge: document.getElementById("category-badge"),
  classCode: document.getElementById("class-code"),
  clinicalSummary: document.getElementById("clinical-summary"),
  probList: document.getElementById("prob-list"),
  stageCopy: document.getElementById("stage-copy"),
  stageFlow: document.getElementById("stage-flow"),
  progressBar: document.getElementById("progress-bar"),
  modelStatus: document.getElementById("model-status"),
  themeToggle: document.getElementById("theme-toggle"),
  toast: document.getElementById("toast"),
};

let selectedFile = null;
let objectUrl = null;
let analyzing = false;

function showView(name) {
  els.home.hidden = name !== "home";
  els.analysis.hidden = name !== "analysis";
  els.analyzing.hidden = name !== "analyzing";
  els.results.hidden = name !== "results";
}

function showToast(message) {
  els.toast.hidden = false;
  els.toast.textContent = message;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    els.toast.hidden = true;
  }, 4200);
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("dermai-theme", theme);
  els.themeToggle.querySelector(".theme-toggle-label").textContent =
    theme === "dark" ? "Light mode" : "Dark mode";
}

function fillSelect(select, items) {
  const current = select.value;
  select.innerHTML = '<option value="">Select</option>';
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    select.appendChild(option);
  });
  if ([...select.options].some((opt) => opt.value === current)) {
    select.value = current;
  }
}

async function loadOptions() {
  try {
    const response = await fetch("/options");
    if (!response.ok) throw new Error("unavailable");
    const data = await response.json();
    fillSelect(els.sex, data.sex || []);
    fillSelect(els.location, data.location || []);
    const device = data.device ? String(data.device).toUpperCase() : "CPU";
    els.modelStatus.textContent = `Model ready · ${device}`;
    els.modelStatus.classList.add("ready");
    els.modelStatus.classList.remove("down");
  } catch (_error) {
    els.modelStatus.textContent = "Backend unavailable";
    els.modelStatus.classList.add("down");
    els.modelStatus.classList.remove("ready");
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function clearImage() {
  selectedFile = null;
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  els.previewImg.removeAttribute("src");
  els.dropPreview.hidden = true;
  els.dropEmpty.hidden = false;
  els.fileMeta.textContent = "";
  els.imageStatus.textContent = "No image";
  els.replaceBtn.disabled = true;
  els.removeBtn.disabled = true;
  els.fileInput.value = "";
}

function setImage(file) {
  if (!/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) {
    showToast("Unsupported image type. Please upload JPG, JPEG, PNG, WEBP, or BMP.");
    return;
  }
  selectedFile = file;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  els.previewImg.src = objectUrl;
  els.dropPreview.hidden = false;
  els.dropEmpty.hidden = true;
  els.replaceBtn.disabled = false;
  els.removeBtn.disabled = false;
  els.imageStatus.textContent = "Image loaded";
  const probe = new Image();
  probe.onload = () => {
    els.fileMeta.textContent = `${file.name} · ${probe.width} × ${probe.height} px · ${formatBytes(file.size)}`;
  };
  probe.src = objectUrl;
}

function validate() {
  if (!selectedFile) {
    showToast("Please upload a skin lesion image first.");
    return false;
  }
  const ageText = els.age.value.trim();
  if (!ageText) {
    showToast("Please enter the patient’s age.");
    return false;
  }
  const age = Number(ageText);
  if (!Number.isFinite(age) || age < 0 || age > 120) {
    showToast("Please enter a valid age between 0 and 120.");
    return false;
  }
  if (!els.sex.value) {
    showToast("Please select the patient’s sex.");
    return false;
  }
  if (!els.location.value) {
    showToast("Please select the lesion location.");
    return false;
  }
  return true;
}

function renderStages(activeIndex) {
  els.stageFlow.innerHTML = "";
  STAGES.forEach((stage, index) => {
    const item = document.createElement("li");
    item.textContent = stage;
    if (index < activeIndex) item.classList.add("done");
    if (index === activeIndex) item.classList.add("active");
    els.stageFlow.appendChild(item);
  });
  const current = STAGES[Math.min(activeIndex, STAGES.length - 1)];
  els.stageCopy.textContent = current;
  els.progressBar.style.width = `${((activeIndex + 1) / STAGES.length) * 100}%`;
}

function categoryClass(category) {
  const value = (category || "").toLowerCase();
  if (value.includes("malignant")) return "malignant";
  if (value.includes("suspicious")) return "suspicious";
  return "benign";
}

function renderResult(data) {
  const id = String(data.prediction || "").toLowerCase();
  const diagnosis = DISPLAY[id] || data.diagnosis;
  const category = CATEGORY[id] || data.category;
  els.diagnosis.textContent = diagnosis;
  els.confidence.textContent = `${(data.confidence * 100).toFixed(2)}%`;
  els.categoryValue.textContent = category.split(" / ")[0];
  els.categoryNote.textContent = category;
  els.categoryBadge.textContent = category;
  els.categoryBadge.className = `badge ${categoryClass(category)}`;
  els.classCode.textContent = id.toUpperCase();
  const sexLabel = els.sex.options[els.sex.selectedIndex]?.text || els.sex.value;
  const locLabel = els.location.options[els.location.selectedIndex]?.text || els.location.value;
  els.clinicalSummary.textContent = `${els.age.value}y · ${sexLabel} · ${locLabel}`;

  const rows = (data.classes || Object.entries(data.probabilities || {}).map(([key, probability]) => ({
    id: key,
    label: DISPLAY[key] || key,
    probability,
  }))).map((row) => ({
    ...row,
    label: DISPLAY[row.id] || row.label,
  }));

  els.probList.innerHTML = "";
  rows.forEach((row) => {
    const isTop = row.id === id;
    const wrap = document.createElement("div");
    wrap.className = `prob-row${isTop ? " top" : ""}`;
    wrap.innerHTML = `
      <span class="name">${row.id.toUpperCase()}<small>${row.label}</small></span>
      <span class="pct">${(row.probability * 100).toFixed(2)}%</span>
      <div class="bar"><span></span></div>
    `;
    els.probList.appendChild(wrap);
    requestAnimationFrame(() => {
      wrap.querySelector(".bar > span").style.width = `${Math.max(row.probability * 100, 0.7)}%`;
    });
  });

  showView("results");
}

function parseError(payload, fallback) {
  if (payload && typeof payload.message === "string") return payload.message;
  if (payload && payload.detail && typeof payload.detail.message === "string") {
    return payload.detail.message;
  }
  return fallback;
}

async function analyze() {
  if (analyzing || !validate()) return;
  analyzing = true;
  els.analyzeBtn.disabled = true;
  showView("analyzing");

  let stage = 0;
  renderStages(0);
  const timer = setInterval(() => {
    if (stage < STAGES.length - 1) {
      stage += 1;
      renderStages(stage);
    }
  }, 480);

  const form = new FormData();
  form.append("image", selectedFile, selectedFile.name);
  form.append("age", els.age.value.trim());
  form.append("sex", els.sex.value);
  form.append("location", els.location.value);

  try {
    const response = await fetch("/predict", { method: "POST", body: form });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(parseError(payload, "Analysis could not be completed. Please try again."));
    }
    clearInterval(timer);
    renderStages(STAGES.length - 1);
    await new Promise((resolve) => setTimeout(resolve, 280));
    renderResult(payload);
  } catch (error) {
    clearInterval(timer);
    showView("analysis");
    const offline = error instanceof TypeError;
    showToast(
      offline
        ? "The analysis service is unavailable. Please start the backend and try again."
        : error.message
    );
  } finally {
    analyzing = false;
    els.analyzeBtn.disabled = false;
  }
}

function resetWorkflow() {
  clearImage();
  els.age.value = "";
  els.sex.value = "";
  els.location.value = "";
  showView("analysis");
}

function bindUpload() {
  els.dropzone.addEventListener("click", () => els.fileInput.click());
  els.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      els.fileInput.click();
    }
  });
  els.fileInput.addEventListener("change", () => {
    if (els.fileInput.files[0]) setImage(els.fileInput.files[0]);
  });
  ["dragenter", "dragover"].forEach((type) => {
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((type) => {
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropzone.classList.remove("dragover");
    });
  });
  els.dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) setImage(file);
  });
  els.replaceBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    els.fileInput.click();
  });
  els.removeBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    clearImage();
  });
}

els.startBtn.addEventListener("click", () => {
  showView("analysis");
  loadOptions();
});
els.analyzeBtn.addEventListener("click", analyze);
els.resetBtn.addEventListener("click", resetWorkflow);
els.themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  setTheme(next);
});

setTheme(localStorage.getItem("dermai-theme") || "light");
bindUpload();
loadOptions();
