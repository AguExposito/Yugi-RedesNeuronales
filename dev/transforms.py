"""Transforms de imagen unificados para entrenamiento e inferencia.

Este modulo corrige el bug principal del prototipo Colab: alli el entrenamiento
usaba ``Normalize`` de ImageNet pero la galeria se construia SIN normalizar, y la
funcion ``identify()`` aplicaba transforms ALEATORIOS (RandomResizedCrop,
ColorJitter) a la consulta, dando resultados no deterministas. Aca hay una unica
fuente de verdad:

- ``train_transform``: con augmentacion fuerte tipo "foto real" (solo para entrenar).
- ``inference_transform``: determinista (para galeria Y para la API).

Referencias del curso:
- Augmentacion de imagenes (flip, crop, color) como tecnica para ampliar el
  dataset: ``U2/2 Redes_Convolucionales.ipynb``, seccion "Image Augmentation".
- Augmentacion como regularizacion contra el overfitting:
  ``U1/10 Tecnicas para Evitar el Overfitting.ipynb`` (misma idea que dropout /
  weight decay: evitar que la red memorice cada pixel).
- Normalize con las estadisticas de ImageNet cuando se usa una red preentrenada:
  ``U3/2_FineTuning.ipynb`` (alli se normaliza igual porque la ResNet fue
  entrenada con esas medias/desvios) y FCC 06.
"""

import random

from PIL import Image
from torchvision import transforms

# Tamano de entrada del modelo en formato (ALTO, ANCHO), que es el orden que
# espera transforms.Resize. Las cartas 'small' de YGOPRODeck miden 268 de ancho
# x 391 de alto; antes estaba invertido como (268, 391) y todas las cartas se
# aplastaban a formato apaisado. ResNet acepta cualquier tamano gracias a su
# AdaptiveAvgPool final (ver U2/2 Redes_Convolucionales.ipynb, seccion ResNet).
IMG_SIZE = (391, 268)  # (alto, ancho)

# Relacion de aspecto ancho/alto de una carta (la usa tambien el front-end
# para dibujar la guia de encuadre de la camara).
CARD_ASPECT = 268 / 391

# Estadisticas de ImageNet: la ResNet-18 preentrenada espera entradas
# normalizadas con estas medias y desvios por canal RGB.
# Es exactamente el mismo Normalize que usa U3/2_FineTuning.ipynb.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomTableBackground:
    """Pega la carta sobre un fondo aleatorio que simula una mesa (padding falso).

    Motivo: la galeria se construye con scans "perfectos" (la carta ocupa el
    100% de la imagen), pero una foto real siempre incluye mesa/fondo alrededor.
    Si la red nunca vio fondo en entrenamiento, el embedding de una foto real
    cae lejos del scan y el retrieval falla. Este transform cierra esa brecha:
    es augmentacion como regularizacion (U1/10 Tecnicas para Evitar el
    Overfitting.ipynb) llevada al dominio especifico del problema.

    Ademas hace que la "positiva" de la tripleta sea genuinamente distinta del
    ancla, evitando que la Triplet Loss colapse a ~0 (advertencia de
    GuiaEstudio/06_YuGiOh_Triplet.md, seccion "Triplet Loss").
    """

    def __init__(self, p=0.7, card_scale=(0.55, 0.92), max_angle=8):
        # p: probabilidad de aplicar el fondo (a veces la foto SI viene bien
        # recortada, asi que no siempre se agrega mesa).
        self.p = p
        # card_scale: fraccion del lienzo que ocupa la carta (0.55 = carta chica
        # con mucha mesa alrededor; 0.92 = casi sin margen).
        self.card_scale = card_scale
        # max_angle: rotacion de la carta sobre la mesa, en grados.
        self.max_angle = max_angle

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        # Rotar la carta con canal alfa: las esquinas que "sobran" quedan
        # transparentes y dejan ver la mesa (en vez de rellenarse de negro).
        angle = random.uniform(-self.max_angle, self.max_angle)
        card = img.convert("RGBA").rotate(angle, expand=True,
                                          resample=Image.BICUBIC)

        # El lienzo (la mesa) se dimensiona a partir de la carta ya rotada,
        # para que siempre entre completa.
        scale = random.uniform(*self.card_scale)
        canvas_w = int(card.width / scale)
        canvas_h = int(card.height / scale)

        # Fondo: color solido aleatorio (madera, mantel, escritorio...).
        base_color = tuple(random.randint(30, 220) for _ in range(3))
        background = Image.new("RGB", (canvas_w, canvas_h), base_color)
        # La mitad de las veces se mezcla con ruido para simular textura
        # (una mesa real nunca es un color perfectamente uniforme).
        if random.random() < 0.5:
            noise = Image.effect_noise((canvas_w, canvas_h),
                                       random.uniform(10, 40)).convert("L")
            texture = Image.merge("RGB", (noise, noise, noise))
            background = Image.blend(background, texture, alpha=0.25)

        # Posicion aleatoria de la carta sobre la mesa (encuadre imperfecto).
        max_x = max(0, canvas_w - card.width)
        max_y = max(0, canvas_h - card.height)
        position = (random.randint(0, max_x), random.randint(0, max_y))
        # El tercer argumento (mascara alfa) hace que solo se pegue la carta,
        # no las esquinas transparentes de la rotacion.
        background.paste(card, position, card)
        return background


# ---------------------------------------------------------------------------
# Transform de ENTRENAMIENTO: augmentacion fuerte que simula una foto real.
#
# Es clave para la Triplet Loss: como el dataset tiene 1 imagen por carta, la
# "positiva" es el ancla augmentada. Si la positiva fuera casi identica al
# ancla, la perdida seria ~0 desde la primera epoca y la red no aprenderia nada
# (exactamente lo que paso en el primer entrenamiento: loss promedio 0.0012 en
# la epoca 1; ver GuiaEstudio/06_YuGiOh_Triplet.md, checklist 5).
# ---------------------------------------------------------------------------
train_transform = transforms.Compose([
    # Fondo de mesa falso (ver clase RandomTableBackground arriba).
    RandomTableBackground(p=0.7),
    # Se agranda la imagen para que el crop aleatorio tenga margen.
    transforms.Resize((440, 300)),
    # Recorte aleatorio que conserva 70-100% del area con relacion de aspecto
    # cercana a la de una carta: simula encuadres imperfectos
    # (augmentacion, U2/2 "Image Augmentation").
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.6, 0.8)),
    # Distorsion de perspectiva: la camara nunca esta perfectamente
    # perpendicular a la mesa; la carta se ve como un trapecio.
    transforms.RandomPerspective(distortion_scale=0.25, p=0.5),
    # Variaciones de brillo/contraste/saturacion/tono: distintas condiciones de
    # iluminacion (luz calida, sombra, flash) al fotografiar la carta.
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.03),
    # Rotaciones y traslaciones: la carta nunca queda perfectamente alineada.
    transforms.RandomAffine(degrees=12, translate=(0.05, 0.05)),
    # Desenfoque ocasional: fotos movidas o fuera de foco.
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    # Convierte PIL [0,255] a tensor float [0,1] con forma (C,H,W)
    # (mismo paso que en U1/Copia de L1.ipynb Parte 1 y FCC 04).
    transforms.ToTensor(),
    # Normalizacion ImageNet (ver comentario del modulo).
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Transform de INFERENCIA: 100% determinista.
#
# Lo usan tanto build_gallery.py (para calcular el embedding de cada carta de
# referencia) como backend/inference.py (para la foto del usuario). Que ambos
# usen EXACTAMENTE el mismo preprocesamiento es lo que hace comparables las
# distancias en el espacio de embeddings.
# ---------------------------------------------------------------------------
inference_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
