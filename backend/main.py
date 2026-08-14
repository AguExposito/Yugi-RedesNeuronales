"""Back-end web del trabajo integrador (FastAPI).

Expone el modelo entrenado como servicio HTTP y sirve el front-end estatico.
Corresponde a la etapa de "deploy" del curso (FCC 09, model deployment): el
modelo deja de vivir en un notebook y pasa a responder peticiones reales.

Endpoints:
- GET  /api/health    -> estado del servicio y tamano de la galeria.
- POST /api/identify  -> recibe una foto (multipart) y devuelve el top-k de
                         cartas mas parecidas con sus distancias.
- GET  /cards/{id}.jpg -> imagen de una carta de la galeria (para mostrarla
                          en los resultados del front-end).
- GET  /              -> front-end (frontend/index.html).

Ejecucion:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.inference import CardIdentifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_DIR / "frontend"
IMAGES_DIR = PROJECT_DIR / "data" / "yugioh_card_images"

app = FastAPI(title="Reconocedor de cartas Yu-Gi-Oh!",
              description="Red siamesa + Triplet Loss + retrieval por embeddings")

# El modelo y la galeria se cargan UNA sola vez al arrancar el servidor
# (cargar ~45 MB de ResNet en cada peticion seria inviable).
identifier = CardIdentifier()


@app.get("/api/health")
def health() -> dict:
    """Chequeo rapido: confirma que el modelo y la galeria estan cargados."""
    return {
        "status": "ok",
        "device": str(identifier.device),
        "gallery_size": len(identifier.names),
    }


@app.post("/api/identify")
async def identify(file: UploadFile = File(...), top_k: int = 5) -> dict:
    """Identifica la carta de la foto subida.

    Pipeline completo (GuiaEstudio/06_YuGiOh_Triplet.md, "Idea central"):
        Imagen -> ResNet -> vector 128-D -> distancia -> nombre de carta
    """
    # Validacion en el borde del sistema: el archivo debe ser una imagen decodificable.
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen valida.")

    # Toda la logica de ML vive en inference.py; la API solo traduce HTTP <-> modelo.
    result = identifier.identify(image, top_k=top_k)

    # Se agrega la URL local de la imagen de cada candidata para que el
    # front-end pueda mostrarla junto al nombre y la distancia.
    for match in result["matches"]:
        match["image_url"] = f"/cards/{match['image_id']}.jpg"

    return result


# /cards/{id}.jpg -> las imagenes descargadas por data/cards_downloader.py.
app.mount("/cards", StaticFiles(directory=IMAGES_DIR), name="cards")

# La raiz sirve el front-end completo (html=True hace que "/" devuelva index.html).
# Se monta al final para que no tape las rutas /api/* y /cards/*.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
