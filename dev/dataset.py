"""Datasets del proyecto: imagenes individuales y pares para la red siamesa.

Un ``Dataset`` de PyTorch es una clase con ``__len__`` y ``__getitem__`` que el
``DataLoader`` recorre en lotes. El patron Dataset custom se estudia en
``U1/Copia de L1.ipynb`` Parte 1 (clase ``PlatesDataSet``) y en FCC 04
(custom datasets); aca se aplica a las cartas descargadas por
``data/cards_downloader.py``.

Hay dos datasets:
- ``CardImageDataset``: una muestra = una imagen de carta (con su id y nombre).
  Se usa para construir la galeria de embeddings.
- ``TripletYugiDataset``: una muestra = (vista1, vista2, etiqueta_de_clase).
  Dos augmentaciones de la MISMA carta; la negativa NO se sortea aca, se elige
  dentro del lote en ``dev/train.py`` (batch-hard mining). Asi la Triplet Loss
  deja de ser trivialmente 0 (DocumentacionRN_Yugi, "Ramas Gemelas" +
  GuiaEstudio/06_YuGiOh_Triplet.md).
"""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

# Rutas por defecto: las que genera data/cards_downloader.py.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_DIR / "data" / "yugioh_card_images"
DEFAULT_CARDS_JSON = PROJECT_DIR / "data" / "cards.json"


def load_id_to_name(cards_json: Path) -> dict[str, str]:
    """Construye el mapa image_id -> nombre de carta desde cards.json.

    Cada carta del JSON puede tener varias imagenes (artes alternativas) en
    ``card_images``; todas apuntan al mismo nombre. Asi, dos archivos distintos
    de la misma carta comparten etiqueta.
    """
    with open(cards_json, encoding="utf-8") as f:
        cards = json.load(f)

    id_to_name = {}
    for card in cards:
        # El id principal de la carta tambien se registra por robustez.
        id_to_name[str(card["id"])] = card["name"]
        for image_info in card.get("card_images", []):
            id_to_name[str(image_info["id"])] = card["name"]
    return id_to_name


class CardImageDataset(Dataset):
    """Una muestra = (imagen transformada, image_id, nombre de carta).

    Equivalente al ``CustomImageDataset`` del prototipo Colab. Estructura
    identica a ``PlatesDataSet`` de U1/Copia de L1.ipynb Parte 1:
    __init__ arma la lista de archivos, __getitem__ abre y transforma una imagen.
    """

    def __init__(self, image_dir=DEFAULT_IMAGE_DIR, cards_json=DEFAULT_CARDS_JSON,
                 transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.id_to_name = load_id_to_name(Path(cards_json))

        # Solo se listan archivos cuyo id figura en el JSON: evita que una
        # imagen sin etiqueta rompa el mapeo (en el prototipo este filtro solo
        # existia en el dataset de tripletas y los dos datasets podian diferir).
        self.files = sorted(
            p for p in self.image_dir.glob("*.jpg")
            if p.stem in self.id_to_name
        )
        if not self.files:
            raise FileNotFoundError(
                f"No hay imagenes en {self.image_dir}. "
                "Ejecutar primero data/cards_downloader.py"
            )

    def __len__(self) -> int:
        # Cantidad total de muestras; el DataLoader lo usa para saber cuando
        # termina una epoca (ver U1/1 Regresion_Lineal_1.ipynb).
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        # .convert("RGB") garantiza 3 canales aunque el JPG venga en otro modo.
        image = Image.open(path).convert("RGB")
        if self.transform:
            # El transform convierte PIL -> tensor normalizado (dev/transforms.py).
            image = self.transform(image)
        card_id = path.stem
        return image, card_id, self.id_to_name[card_id]


class TripletYugiDataset(Dataset):
    """Una muestra = (vista1, vista2, etiqueta) para mining de negativas en lote.

    Antes se devolvia (ancla, positiva, negativa) con negativa ALEATORIA. Con
    ResNet preentrenada esa negativa casi siempre esta ya muy lejos ->
    Triplet Loss ~0 y no hay gradiente (GuiaEstudio/06 checklist 5).

    Ahora se devuelven DOS vistas augmentadas de la misma carta + un indice
    entero de clase. En ``dev/train.py`` se elige, dentro del lote, la
    negativa MAS CERCANA (batch-hard): esa SI viola el margen y genera
    aprendizaje. Las "Ramas Gemelas" de DocumentacionRN_Yugi siguen siendo
    la misma red aplicada a ambas vistas (pesos compartidos).
    """

    def __init__(self, image_dir=DEFAULT_IMAGE_DIR, cards_json=DEFAULT_CARDS_JSON,
                 transform=None):
        self.transform = transform
        # Reutiliza CardImageDataset para el listado de archivos y etiquetas.
        base = CardImageDataset(image_dir, cards_json, transform=None)
        self.files = base.files
        self.names = [base.id_to_name[p.stem] for p in self.files]

        # Mapa nombre -> indice entero estable (0..C-1) para el mining en lote.
        unique = sorted(set(self.names))
        self.name_to_label = {n: i for i, n in enumerate(unique)}
        self.labels = [self.name_to_label[n] for n in self.names]

        if len(unique) < 2:
            raise ValueError("Se necesitan al menos 2 cartas distintas para armar tripletas.")

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, idx: int):
        """Abre y transforma la imagen idx (cada llamada re-augmenta distinto)."""
        image = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx: int):
        # Dos llamadas a _load con el MISMO idx: el transform estocastico
        # (mesa, perspectiva, blur...) genera dos vistas distintas.
        # Si hay artes alternativas de la misma carta, se podria mezclar indices
        # de la misma etiqueta; con 1 imagen/carta la augmentacion alcanza.
        view1 = self._load(idx)
        view2 = self._load(idx)
        # Etiqueta entera: identifica a que carta pertenecen ambas vistas.
        label = self.labels[idx]
        return view1, view2, label
