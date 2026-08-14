"""Evaluacion del reconocedor: top-1 / top-5 retrieval + test de rechazo.

Metricas recomendadas por GuiaEstudio/06_YuGiOh_Triplet.md
("Agregar top-1 / top-5 retrieval accuracy"):
- Top-1: la carta correcta es el vecino mas cercano.
- Top-5: la carta correcta esta entre los 5 vecinos mas cercanos.

Las consultas se generan con ``train_transform`` (mesa, perspectiva, blur)
para simular fotos reales contra la galeria de scans limpios.

El test de rechazo crea imagenes que NO son cartas (ruido, color plano,
textura) y verifica que ``recognized`` sea False (umbral calibrado).

Evaluacion / train vs test: idea de U1/8 Evaluacion de los Modelos.ipynb.

Uso:
    python dev/evaluate.py
"""

import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from dataset import CardImageDataset
from model import EmbeddingNet
from transforms import inference_transform, train_transform

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


def load_model_and_gallery(device):
    checkpoint = torch.load(ARTIFACTS_DIR / "model.pt", map_location=device,
                            weights_only=False)
    model = EmbeddingNet(embedding_dim=checkpoint["embedding_dim"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    gallery = torch.load(ARTIFACTS_DIR / "gallery.pt", map_location="cpu",
                         weights_only=False)
    embeddings = gallery["embeddings"].to(device)
    names = gallery["names"]
    threshold = float(gallery.get("distance_threshold", 0.75))
    return model, embeddings, names, threshold


def evaluate_retrieval(model, embeddings, names, dataset, device, sample_size=200):
    """Top-1 / top-5 con consultas augmentadas (simulan fotos)."""
    n = len(dataset)
    k = min(sample_size, n)
    indices = random.sample(range(n), k)

    top1 = top5 = 0
    with torch.no_grad():
        for idx in indices:
            path = dataset.files[idx]
            true_name = names[idx]
            pil = Image.open(path).convert("RGB")
            # Consulta "foto real": misma augmentacion que en entrenamiento.
            query = train_transform(pil).unsqueeze(0).to(device)
            emb = model(query)
            dists = torch.cdist(emb, embeddings).squeeze(0)
            # Indices de los 5 vecinos mas cercanos.
            nearest = dists.argsort()[:5]
            pred_names = [names[i] for i in nearest.tolist()]
            if pred_names[0] == true_name:
                top1 += 1
            if true_name in pred_names:
                top5 += 1

    print(f"Retrieval (consultas augmentadas, n={k}):")
    print(f"  Top-1 accuracy: {100.0 * top1 / k:.1f}% ({top1}/{k})")
    print(f"  Top-5 accuracy: {100.0 * top5 / k:.1f}% ({top5}/{k})")
    return top1 / k, top5 / k


def make_non_card_images(n=20, size=(268, 391)):
    """Genera imagenes que no son cartas: ruido, color plano, textura."""
    images = []
    for i in range(n):
        kind = i % 3
        if kind == 0:
            # Color plano aleatorio (una mesa / pared sin carta).
            color = tuple(random.randint(0, 255) for _ in range(3))
            img = Image.new("RGB", size, color)
        elif kind == 1:
            # Ruido gaussiano (PIL effect_noise).
            noise = Image.effect_noise(size, random.uniform(20, 80)).convert("L")
            img = Image.merge("RGB", (noise, noise, noise))
        else:
            # Gradiente / formas geometricas sin parecer una carta.
            img = Image.new("RGB", size, (40, 40, 60))
            draw = ImageDraw.Draw(img)
            for _ in range(8):
                x0, y0 = random.randint(0, size[0]), random.randint(0, size[1])
                x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
                c = tuple(random.randint(0, 255) for _ in range(3))
                # PIL exige x0<=x1 e y0<=y1: se ordenan las esquinas.
                box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
                draw.rectangle(box, fill=c)
        images.append(img)
    return images


def evaluate_rejection(model, embeddings, threshold, device, n=20):
    """Verifica que imagenes no-carta den recognized=False."""
    images = make_non_card_images(n)
    rejected = 0
    distances = []
    with torch.no_grad():
        for img in images:
            query = inference_transform(img).unsqueeze(0).to(device)
            emb = model(query)
            dist = torch.cdist(emb, embeddings).squeeze(0).min().item()
            distances.append(dist)
            if dist > threshold:
                rejected += 1

    print(f"Rechazo de no-cartas (n={n}, threshold={threshold:.4f}):")
    print(f"  Rechazadas: {100.0 * rejected / n:.1f}% ({rejected}/{n})")
    print(f"  Distancia media al vecino: {sum(distances) / len(distances):.4f}")
    return rejected / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando en: {device}")

    model, embeddings, names, threshold = load_model_and_gallery(device)
    # Dataset solo para listar archivos; la consulta usa train_transform.
    dataset = CardImageDataset(transform=None)
    print(f"Galeria: {len(names)} cartas | umbral={threshold:.4f}")

    evaluate_retrieval(model, embeddings, names, dataset, device)
    evaluate_rejection(model, embeddings, threshold, device)


if __name__ == "__main__":
    main()
