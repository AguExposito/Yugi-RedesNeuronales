"""Motor de inferencia: identifica una carta buscando su vecino mas cercano.

Encapsula lo que en el prototipo Colab era la funcion ``identify()``:
    foto -> transform -> EmbeddingNet -> vector 128-D -> torch.cdist contra la
    galeria -> top-k cartas mas cercanas.

Mejoras:
- Transform DETERMINISTA de inferencia (compartido con build_gallery).
- Top-k + umbral de rechazo CALIBRADO (guardado en gallery.pt por
  build_gallery.py). La constante de abajo es solo fallback si el archivo
  es antiguo y no trae el campo.

El retrieval por vecino mas cercano se explica en
GuiaEstudio/06_YuGiOh_Triplet.md (concepto 6, "Nearest neighbor:
torch.cdist -> argmin").
"""

import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
# Se agrega dev/ al path para reutilizar model.py y transforms.py sin duplicar codigo.
sys.path.insert(0, str(PROJECT_DIR / "dev"))

from model import EmbeddingNet          # noqa: E402
from transforms import inference_transform  # noqa: E402

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

# Fallback si gallery.pt no trae distance_threshold (galeria vieja).
# El valor real se calibra en build_gallery.py (U1/9 Seleccion de Modelos).
FALLBACK_THRESHOLD = 0.75


class CardIdentifier:
    """Carga modelo + galeria UNA sola vez y responde consultas."""

    def __init__(self):
        # En un servidor web la inferencia suele correr en CPU; si hay GPU se usa.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Modelo entrenado (dev/train.py) ---
        checkpoint = torch.load(ARTIFACTS_DIR / "model.pt", map_location=self.device,
                                weights_only=False)
        # pretrained=False: load_state_dict pisa los pesos con los entrenados.
        self.model = EmbeddingNet(embedding_dim=checkpoint["embedding_dim"],
                                  pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        # eval(): BatchNorm en modo inferencia (U2/2 Redes_Convolucionales.ipynb).
        self.model.eval()

        # --- Galeria de referencia (dev/build_gallery.py) ---
        gallery = torch.load(ARTIFACTS_DIR / "gallery.pt", map_location="cpu",
                             weights_only=False)
        self.embeddings = gallery["embeddings"].to(self.device)  # [N, 128]
        self.ids = gallery["ids"]
        self.names = gallery["names"]
        # Umbral calibrado; fallback si el .pt es de una version anterior.
        self.threshold = float(gallery.get("distance_threshold", FALLBACK_THRESHOLD))

    def identify(self, image: Image.Image, top_k: int = 5) -> dict:
        """Devuelve las top_k cartas mas parecidas a la imagen consultada."""
        # Mismo preprocesamiento que la galeria: Resize + ToTensor + Normalize
        # (dev/transforms.py). unsqueeze(0) agrega la dimension de lote:
        # (3,H,W) -> (1,3,H,W), porque la red siempre espera lotes.
        query = inference_transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        # no_grad: solo inferencia, no se necesitan gradientes.
        with torch.no_grad():
            query_emb = self.model(query)  # [1, 128], L2-normalizado

        # torch.cdist: matriz de distancias L2 entre la consulta y las N cartas
        # de la galeria -> [1, N] -> squeeze -> [N].
        dists = torch.cdist(query_emb, self.embeddings).squeeze(0)

        # topk con largest=False = los k vecinos MAS CERCANOS (menor distancia).
        k = min(top_k, len(self.names))
        top_dists, top_idx = torch.topk(dists, k=k, largest=False)

        matches = [
            {
                "name": self.names[i],
                "image_id": self.ids[i],
                # La distancia se informa al usuario como medida de confianza:
                # ~0 = practicamente identica, cerca de 2 = nada que ver.
                "distance": round(d.item(), 4),
            }
            for d, i in zip(top_dists, top_idx)
        ]

        # Umbral calibrado: si ni el mejor candidato esta cerca -> no reconocida.
        recognized = matches[0]["distance"] <= self.threshold

        return {
            "recognized": recognized,
            "threshold": round(self.threshold, 4),
            "matches": matches,
        }
