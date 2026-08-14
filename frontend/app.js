// Front-end del reconocedor de cartas.
// Flujo: elegir imagen (archivo o camara) -> POST /api/identify -> mostrar top-5.
// En modo camara, la captura se RECORTE al marco de guia (aspecto de carta)
// para que la consulta llegue con poco fondo de mesa.

// --- Referencias a los elementos de la pagina ---
const tabUpload = document.getElementById("tab-upload");
const tabCamera = document.getElementById("tab-camera");
const uploadMode = document.getElementById("upload-mode");
const cameraMode = document.getElementById("camera-mode");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("capture-btn");
const previewBox = document.getElementById("preview-box");
const preview = document.getElementById("preview");
const identifyBtn = document.getElementById("identify-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

// Relacion de aspecto ancho/alto de una carta Yu-Gi-Oh (igual que CARD_ASPECT
// en dev/transforms.py).
const CARD_ASPECT = 268 / 391;

// Imagen seleccionada (Blob) lista para enviar al back-end.
let selectedBlob = null;
// Stream de la camara, para poder apagarla al cambiar de pestana.
let cameraStream = null;

// --- Cambio de pestanas archivo <-> camara ---
tabUpload.addEventListener("click", () => switchTab("upload"));
tabCamera.addEventListener("click", () => switchTab("camera"));

function switchTab(mode) {
  const isUpload = mode === "upload";
  tabUpload.classList.toggle("active", isUpload);
  tabCamera.classList.toggle("active", !isUpload);
  uploadMode.hidden = !isUpload;
  cameraMode.hidden = isUpload;
  if (isUpload) {
    stopCamera();
  } else {
    startCamera();
  }
}

// --- Camara del navegador (getUserMedia) ---
async function startCamera() {
  try {
    // facingMode "environment" pide la camara trasera en celulares
    // (la ideal para escanear una carta sobre la mesa).
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
    });
    video.srcObject = cameraStream;
  } catch (err) {
    setStatus("No se pudo acceder a la camara: " + err.message, "warn");
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
    video.srcObject = null;
  }
}

/**
 * Calcula el rectangulo del marco de guia en coordenadas del video nativo.
 * El CSS centra un rectangulo con aspect-ratio de carta y altura 78% del
 * contenedor; aca se replica esa geometria sobre videoWidth x videoHeight.
 */
function guideRectInVideo() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const guideH = vh * 0.78;
  const guideW = guideH * CARD_ASPECT;
  const x = (vw - guideW) / 2;
  const y = (vh - guideH) / 2;
  return {
    x: Math.max(0, Math.round(x)),
    y: Math.max(0, Math.round(y)),
    w: Math.min(vw, Math.round(guideW)),
    h: Math.min(vh, Math.round(guideH)),
  };
}

// Capturar: dibuja solo la zona del marco (recorte) y la exporta a JPEG.
captureBtn.addEventListener("click", () => {
  if (!video.videoWidth) {
    setStatus("La camara aun no esta lista.", "warn");
    return;
  }
  const { x, y, w, h } = guideRectInVideo();
  canvas.width = w;
  canvas.height = h;
  // drawImage con sx,sy,sw,sh -> dx,dy,dw,dh: recorta al marco de la carta.
  canvas.getContext("2d").drawImage(video, x, y, w, h, 0, 0, w, h);
  canvas.toBlob((blob) => setSelectedImage(blob), "image/jpeg", 0.92);
});

// --- Seleccion de archivo (clic o arrastrar y soltar) ---
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) setSelectedImage(fileInput.files[0]);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) setSelectedImage(e.dataTransfer.files[0]);
});

// Guarda la imagen elegida y muestra la vista previa + boton de identificar.
function setSelectedImage(blob) {
  selectedBlob = blob;
  preview.src = URL.createObjectURL(blob);
  previewBox.hidden = false;
  setStatus("Imagen lista. Presiona 'Identificar carta'.");
  resultsEl.innerHTML = "";
}

// --- Llamada al back-end ---
identifyBtn.addEventListener("click", async () => {
  if (!selectedBlob) return;

  identifyBtn.disabled = true;
  setStatus("Calculando embedding y buscando en la galeria...");

  try {
    // La imagen viaja como multipart/form-data, el formato que espera
    // el parametro UploadFile de FastAPI en backend/main.py.
    const formData = new FormData();
    formData.append("file", selectedBlob, "query.jpg");

    const resp = await fetch("/api/identify", { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Error HTTP ${resp.status}`);
    }

    const data = await resp.json();
    renderResults(data);
  } catch (err) {
    setStatus("Error: " + err.message, "warn");
  } finally {
    identifyBtn.disabled = false;
  }
});

// --- Renderizado de resultados ---
function renderResults(data) {
  if (data.recognized) {
    setStatus(`Carta reconocida: ${data.matches[0].name}`, "ok");
  } else {
    setStatus(
      "Ninguna carta de la galeria esta lo suficientemente cerca " +
        "(posible carta desconocida / no es una carta). " +
        "Se muestran los candidatos mas parecidos.",
      "warn"
    );
  }

  resultsEl.innerHTML = "";
  data.matches.forEach((m, i) => {
    // La distancia L2 entre embeddings unitarios va de 0 (identica) a 2
    // (opuesta); se convierte en una barra de similitud 0-100%.
    const similarity = Math.max(0, 1 - m.distance / 2) * 100;

    const div = document.createElement("div");
    div.className = "match" + (i === 0 ? " best" : "");
    div.innerHTML = `
      <span class="rank">#${i + 1}</span>
      <img src="${m.image_url}" alt="${m.name}" loading="lazy" />
      <div class="info">
        <div class="name"></div>
        <div class="distance">distancia: ${m.distance.toFixed(4)}</div>
        <div class="bar"><div style="width:${similarity.toFixed(1)}%"></div></div>
      </div>`;
    // El nombre se asigna con textContent para escapar caracteres especiales.
    div.querySelector(".name").textContent = m.name;
    resultsEl.appendChild(div);
  });
}

function setStatus(message, kind) {
  statusEl.textContent = message;
  statusEl.className = "status" + (kind ? " " + kind : "");
}
