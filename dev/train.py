"""Entrenamiento de la red siamesa con Triplet Loss + batch-hard mining.

Implementa la "Funcion de Entrenamiento" descripta en DocumentacionRN_Yugi:
iterar por epocas, procesar lotes, forward pass, calcular la perdida y
backward pass + optimizacion. El esqueleto del loop
(zero_grad -> backward -> step) es el mismo que se ensena en
``U1/1 Regresion_Lineal_1.ipynb`` y ``U1/3 Regresion_Softmax_1.ipynb``.

Cambio clave respecto del primer entrenamiento (loss ~0 desde epoca 1):
en vez de una negativa ALEATORIA (casi siempre "facil" para ResNet
preentrenada), se elige dentro del lote la negativa MAS CERCANA al ancla
(batch-hard mining). Asi la desigualdad del margen SI se viola y hay
gradiente. El mining no esta en U1-U4; se explica en
GuiaEstudio/06_YuGiOh_Triplet.md (tripletas que realmente ensenan).

Uso:
    python dev/train.py --epochs 12 --batch-size 32
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TripletYugiDataset
from model import EmbeddingNet
from transforms import train_transform

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


def batch_hard_triplet_loss(emb_a, emb_p, labels, margin):
    """Triplet Loss con negativa mas dificil del lote (batch-hard).

    emb_a / emb_p: embeddings L2-normalizados de las dos vistas [B, D].
    labels: etiqueta entera de cada carta [B].

    Para cada ancla i se toma:
      - positiva = emb_p[i] (la otra vista de la misma carta),
      - negativa = la emb_p[j] de OTRA carta con menor distancia a emb_a[i].

    Asi max(0, d(A,P) - d(A,N) + margen) deja de ser 0 casi siempre.
    torch.cdist es la misma herramienta de retrieval de la guia 06.
    """
    # Distancias L2 ancla -> todas las positivas del lote: [B, B].
    dist_ap_matrix = torch.cdist(emb_a, emb_p, p=2)
    # Distancia a la positiva propia: diagonal.
    dist_ap = dist_ap_matrix.diag()

    # Mascara True donde labels[i] != labels[j]: candidatas negativas.
    # (Si un lote trae dos veces la misma carta, no se usan como negativa.)
    neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)  # [B, B]

    # Pone infinito donde NO es negativa, asi el min elige solo otras cartas.
    dist_an_candidates = dist_ap_matrix.clone()
    dist_an_candidates[~neg_mask] = float("inf")
    # Negativa mas cercana (= mas dificil) por cada ancla.
    dist_an, _ = dist_an_candidates.min(dim=1)

    # Si un lote no tiene otras cartas (caso extremo), dist_an queda inf:
    # se descartan esas filas para no romper el promedio.
    valid = torch.isfinite(dist_an)
    if not valid.any():
        # Lote degenerado: devolver 0 sin romper el grafo.
        return emb_a.new_tensor(0.0), 0.0

    # TripletMarginLoss clasica: max(0, d(A,P) - d(A,N) + margen).
    # (GuiaEstudio/06_YuGiOh_Triplet.md, formula de la perdida).
    per_sample = torch.clamp(dist_ap[valid] - dist_an[valid] + margin, min=0.0)
    # Porcentaje de tripletas "activas" (loss > 0): senal de que hay aprendizaje.
    active_pct = (per_sample > 1e-8).float().mean().item() * 100.0
    return per_sample.mean(), active_pct


def train(model, loader, optimizer, device, epochs, margin):
    """Loop de entrenamiento (DocumentacionRN_Yugi, 'Funcion de Entrenamiento')."""
    # model.train() activa el modo entrenamiento (afecta a BatchNorm/Dropout;
    # BatchNorm se explica en U2/2 Redes_Convolucionales.ipynb, Dropout en
    # U1/10 Tecnicas para Evitar el Overfitting.ipynb).
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_active = 0.0
        n_batches = 0
        for batch_idx, (view1, view2, labels) in enumerate(loader):
            # Mover los tensores al mismo device que el modelo (GPU si hay);
            # mismo patron .to(device) que en U1/Copia de L1.ipynb y
            # U3/2_FineTuning.ipynb.
            view1 = view1.to(device)
            view2 = view2.to(device)
            labels = labels.to(device)

            # FORWARD PASS x2 con la MISMA red y los MISMOS pesos: red siamesa
            # (DocumentacionRN_Yugi, punto 3: "dos o mas entradas son
            # procesadas por la misma sub-red").
            emb_a = model(view1)
            emb_p = model(view2)

            # Batch-hard: negativa mas dificil del lote (ver funcion arriba).
            loss, active_pct = batch_hard_triplet_loss(emb_a, emb_p, labels, margin)

            # zero_grad: limpia los gradientes acumulados del paso anterior
            # (PyTorch los ACUMULA por defecto; ver U1/1 Regresion_Lineal_1.ipynb).
            optimizer.zero_grad()
            # backward: autograd calcula dLoss/dPeso (U1/5  MLP4.ipynb).
            loss.backward()
            # step: Adam actualiza los pesos (U1/11  Optimizacion.ipynb).
            optimizer.step()

            running_loss += loss.item()
            running_active += active_pct
            n_batches += 1
            if (batch_idx + 1) % 20 == 0:
                print(f"  epoca {epoch} | lote {batch_idx + 1}/{len(loader)} "
                      f"| loss {loss.item():.4f} | activas {active_pct:.0f}%",
                      flush=True)

        # Si la loss arranca ~0 y las activas ~0%, la red no esta aprendiendo
        # (misma senal de alarma de GuiaEstudio/06 checklist 5).
        avg = running_loss / max(n_batches, 1)
        avg_active = running_active / max(n_batches, 1)
        print(f"Epoca {epoch}/{epochs} terminada | loss promedio: {avg:.4f} "
              f"| tripletas activas: {avg_active:.0f}%", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Entrena la red siamesa Yu-Gi-Oh!")
    parser.add_argument("--epochs", type=int, default=12)
    # Batch >= 16 recomendado: el mining necesita varias cartas distintas
    # en el mismo lote para encontrar negativas dificiles.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Margen de Triplet Loss (antes 0.2; 0.3 fuerza mas separacion)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Congela la ResNet como en la propuesta original")
    parser.add_argument("--workers", type=int, default=2,
                        help="Procesos paralelos del DataLoader")
    args = parser.parse_args()

    # Usar GPU si esta disponible (device-agnostic code, igual que en
    # U1/Copia de L1.ipynb y todos los notebooks del curso).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}", flush=True)

    # Dataset de pares con el transform de AUGMENTACION fuerte (mesa, blur...).
    dataset = TripletYugiDataset(transform=train_transform)
    print(f"Dataset: {len(dataset)} imagenes", flush=True)

    # DataLoader: agrupa las muestras en lotes y las mezcla (shuffle) en cada
    # epoca; el entrenamiento por mini-lotes se usa desde
    # U1/1 Regresion_Lineal_1.ipynb. drop_last=True evita lotes de 1 carta
    # donde el mining no tiene negativa posible.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        pin_memory=(device.type == "cuda"))

    # Modelo: ResNet-18 preentrenada + capa de embedding (ver dev/model.py).
    model = EmbeddingNet(embedding_dim=args.embedding_dim, pretrained=True,
                         freeze_backbone=args.freeze_backbone).to(device)

    # Adam: optimizador adaptativo (U1/11  Optimizacion.ipynb).
    # lr=1e-4 bajo a proposito (fine-tuning, U3/2_FineTuning.ipynb).
    # weight_decay=1e-5: regularizacion L2 (U1/10 Tecnicas para Evitar el
    # Overfitting.ipynb).
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )

    train(model, loader, optimizer, device, args.epochs, args.margin)

    # Persistir el checkpoint: state_dict() contiene todos los pesos.
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    checkpoint_path = ARTIFACTS_DIR / "model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "embedding_dim": args.embedding_dim,
        "freeze_backbone": args.freeze_backbone,
        "margin": args.margin,
    }, checkpoint_path)
    print(f"Modelo guardado en {checkpoint_path}")


if __name__ == "__main__":
    main()
