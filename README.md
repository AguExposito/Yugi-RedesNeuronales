# Trabajo Integrador — Reconocimiento de Cartas Yu-Gi-Oh!

Sistema completo (back-end + front-end) que identifica cartas de Yu-Gi-Oh! a partir de una
foto, usando una **red siamesa con Triplet Loss** que aprende *embeddings* de cartas y las
reconoce por **búsqueda por similitud** (vecino más cercano), tal como se describe en la
propuesta `../DocumentacionRN_Yugi` y en la guía `../GuiaEstudio/06_YuGiOh_Triplet.md`.

```text
Imagen → ResNet-18 (backbone) → vector 128-D (L2-normalizado) → distancia → nombre de carta
```

## ¿Por qué embeddings y no un softmax de 10.000+ clases?

Con más de 10.000 cartas, una capa softmax clásica (como la de `U1/3 Regresion_Softmax_1.ipynb`)
necesitaría una neurona por carta y reentrenarse cada vez que sale una carta nueva.
Con **aprendizaje métrico** (Triplet Loss) el modelo aprende un espacio donde la misma carta
queda cerca y cartas distintas quedan lejos; agregar una carta nueva es solo agregar su
embedding a la galería. Ver `../GuiaEstudio/06_YuGiOh_Triplet.md`, sección
"Por qué no softmax de 10k+ clases".

## Estructura del repositorio

Sigue la organización propuesta en `../DocumentacionRN_Yugi` (sección "Organización del
Repositorio"), extendida con back-end y front-end:

```text
TrabajoIntegrador/
├── README.md
├── requirements.txt
├── data/
│   ├── cards_downloader.py    # descarga el dataset desde la API YGOPRODeck
│   ├── cards.json             # (generado) metadatos id → nombre de carta
│   └── yugioh_card_images/    # (generado) imágenes {image_id}.jpg
├── dev/
│   ├── transforms.py          # transforms de entrenamiento e inferencia (unificados)
│   ├── dataset.py             # CustomImageDataset + TripletYugiDataset
│   ├── model.py               # EmbeddingNet: ResNet-18 → embedding 128-D
│   ├── train.py               # entrenamiento con TripletMarginLoss → artifacts/model.pt
│   └── build_gallery.py       # embeddings de todas las cartas → artifacts/gallery.pt
├── artifacts/                 # (generado) model.pt y gallery.pt
├── backend/
│   ├── inference.py           # carga modelo + galería; identify() con top-k y umbral
│   └── main.py                # API FastAPI (POST /api/identify) + sirve el front-end
└── frontend/
    ├── index.html             # subir foto o escanear con la cámara
    ├── styles.css
    └── app.js
```

## Mapa de conceptos → dónde se estudian en el curso

| Pieza del proyecto | Concepto | Dónde se explica |
|---|---|---|
| `dev/dataset.py` | Dataset custom + DataLoader | `U1/Copia de L1.ipynb` (Parte 1, `PlatesDataSet`), FCC 04 |
| `dev/transforms.py` | Augmentación / Normalize ImageNet | `U2/2 Redes_Convolucionales.ipynb` ("Image Augmentation"), `U3/2_FineTuning.ipynb` |
| `dev/model.py` | ResNet / convoluciones | `U2/1 Capas_Convolucionales.ipynb`, `U2/2 Redes_Convolucionales.ipynb` |
| `dev/model.py` | Transfer learning, reemplazo de `fc`, congelar backbone | `U3/2_FineTuning.ipynb`, FCC 06 |
| `dev/model.py` | Embedding + red siamesa | `../DocumentacionRN_Yugi` ("La Red Siamesa"), `../GuiaEstudio/06_YuGiOh_Triplet.md` |
| `dev/train.py` | Loop de entrenamiento (zero_grad→backward→step) | `U1/1 Regresion_Lineal_1.ipynb`, `U1/3 Regresion_Softmax_1.ipynb`; autograd en `U1/5  MLP4.ipynb` |
| `dev/train.py` | Adam, learning rate | `U1/11  Optimización.ipynb` |
| `dev/train.py` | weight_decay (regularización L2) | `U1/10 Técnicas para Evitar el Overfitting.ipynb` |
| `dev/train.py` | Triplet Loss | `../DocumentacionRN_Yugi` ("Función de Entrenamiento"), `../GuiaEstudio/06_YuGiOh_Triplet.md` |
| `backend/inference.py` | Retrieval por vecino más cercano (`torch.cdist`) | `../GuiaEstudio/06_YuGiOh_Triplet.md`, prototipo Colab |
| GPU / `device` | Entrenamiento en GPU | uso práctico en `U1/Copia de L1.ipynb` y `U3/2_FineTuning.ipynb` |

## Instalación y ejecución

Requiere Python 3.10+.

> **Nota (Windows):** instalar PyTorch dentro de una carpeta muy anidada puede fallar con
> `WinError 206` (ruta demasiado larga). Conviene crear el entorno virtual en una ruta corta,
> por ejemplo `python -m venv %USERPROFILE%\venv-yugi`, y activarlo antes de instalar.

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar el dataset (subset de 800 cartas por defecto; --max-cards 0 = todas)
python data/cards_downloader.py --max-cards 800

# 3. Entrenar el modelo (guarda artifacts/model.pt)
#    Usa batch-hard mining + augmentación tipo foto (mesa/perspectiva/blur).
#    En el log deben aparecer "tripletas activas" > 0% (si quedan en 0%, no aprende).
python dev/train.py --epochs 12 --batch-size 32 --workers 0

# 4. Construir la galería + calibrar umbral de rechazo (guarda artifacts/gallery.pt)
python dev/build_gallery.py

# 5. (Opcional) Evaluar top-1 / top-5 y rechazo de no-cartas
python dev/evaluate.py

# 6. Levantar la aplicación web
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Luego abrir `http://localhost:8000` en el navegador: se puede **subir una foto** de una
carta o **escanearla con la cámara** (con marco de encuadre), y el sistema responde con
las 5 cartas más parecidas y su distancia en el espacio de embeddings.

## Cómo funciona (resumen técnico)

1. **Entrenamiento** (`dev/train.py`): por cada carta se generan dos vistas augmentadas
   (ancla y positiva). La negativa se elige dentro del lote como la carta **más cercana
   equivocada** (batch-hard mining), para que la Triplet Loss no quede en ~0. La
   augmentación incluye fondo de mesa falso, perspectiva, blur y cambios de color
   (simula una foto real). Margen por defecto: 0.3.
2. **Galería** (`dev/build_gallery.py`): se calcula el embedding de cada carta con el
   transform determinista de inferencia y se calibra el umbral de rechazo con distancias
   genuinas vs impostoras (se guarda en `gallery.pt`).
3. **Inferencia** (`backend/inference.py`): la foto del usuario se transforma en un
   embedding y se compara contra toda la galería con `torch.cdist` (distancia L2).
   Se devuelven los top-5 vecinos; si el mejor supera el umbral calibrado, la respuesta
   es "carta no reconocida" (`recognized: false`).

## Mejoras respecto del prototipo Colab

- **Transforms unificados**: el prototipo entrenaba con `Normalize` de ImageNet pero armaba
  la galería sin normalizar, y la función `identify()` aplicaba transforms aleatorios en
  inferencia. Acá el transform de inferencia es determinista y compartido por galería y API.
- **Persistencia**: el modelo (`model.pt`) y la galería (`gallery.pt`) se guardan a disco;
  el prototipo perdía todo al cerrar la sesión.
- **Top-k + umbral de rechazo calibrado**: el prototipo devolvía siempre el top-1; acá se
  devuelven los 5 mejores candidatos y se rechazan consultas demasiado lejanas / no-cartas.
- **Batch-hard + augmentación fuerte**: evita que la Triplet Loss colapse a 0 con
  negativas aleatorias y scans perfectos.
- **Estructura modular** `data/` + `dev/` como pide la propuesta (y FCC 05 "Going Modular").
