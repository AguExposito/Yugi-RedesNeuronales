"""Construye la galeria de embeddings y calibra el umbral de rechazo.

La "galeria" es la base de datos contra la que se compara toda consulta:
un tensor [N, 128] con el embedding de cada carta conocida, mas sus ids y
nombres. En inferencia, identificar una carta = buscar su vecino mas cercano
en esta galeria (GuiaEstudio/06_YuGiOh_Triplet.md).

Ademas calibra ``distance_threshold`` con datos (idea de elegir hiperparametros
con validacion: ``U1/9 Seleccion de Modelos.ipynb``):
- genuinas: distancia entre el scan de la galeria y una vista "foto real"
  (train_transform) de la misma carta;
- impostoras: distancia al vecino mas cercano EQUIVOCADO en la galeria.
El umbral = percentil 95 de las genuinas, acotado por el percentil 5 de las
impostoras, para que fotos validas pasen y no-cartas / cartas ajenas fallen.

Uso:
    python dev/build_gallery.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CardImageDataset
from model import EmbeddingNet
from transforms import inference_transform, train_transform

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

# Fallback si no se puede calibrar (galeria muy chica).
DEFAULT_THRESHOLD = 0.75


def calibrate_threshold(model, dataset, gallery_emb, gallery_names, device,
                        sample_size=200):
    """Estima el umbral de rechazo a partir de genuinas e impostoras."""
    n = len(dataset)
    # Indices a muestrear (sin reemplazo si hay suficientes).
    k = min(sample_size, n)
    indices = torch.randperm(n)[:k].tolist()

    genuine = []
    impostor = []

    # model.eval() + no_grad: modo evaluacion (U1/8 Evaluacion de los Modelos).
    model.eval()
    with torch.no_grad():
        for idx in indices:
            # Scan limpio (mismo transform que la galeria).
            path = dataset.files[idx]
            from PIL import Image
            pil = Image.open(path).convert("RGB")
            name = gallery_names[idx]

            # Vista "foto real": train_transform (mesa, perspectiva, blur...).
            aug = train_transform(pil).unsqueeze(0).to(device)
            emb_aug = model(aug)  # [1, D]

            # Distancia genuina: consulta augmentada vs su propia fila de galeria.
            emb_self = gallery_emb[idx:idx + 1].to(device)
            d_gen = torch.cdist(emb_aug, emb_self).item()
            genuine.append(d_gen)

            # Distancia impostora: vecino mas cercano de OTRA carta.
            dists = torch.cdist(emb_aug, gallery_emb.to(device)).squeeze(0)
            # Enmascara la propia carta (todas las filas con el mismo nombre).
            same = torch.tensor([n == name for n in gallery_names], device=device)
            dists_other = dists.clone()
            dists_other[same] = float("inf")
            d_imp = dists_other.min().item()
            if torch.isfinite(torch.tensor(d_imp)):
                impostor.append(d_imp)

    if not genuine:
        return DEFAULT_THRESHOLD

    gen_t = torch.tensor(genuine)
    # Percentil 95 de genuinas: casi todas las fotos validas quedan debajo.
    thr_gen = torch.quantile(gen_t, 0.95).item()

    if impostor:
        imp_t = torch.tensor(impostor)
        # Percentil 5 de impostoras: casi ninguna carta equivocada queda debajo.
        thr_imp = torch.quantile(imp_t, 0.05).item()
        # Punto intermedio: maximiza el margen entre ambas distribuciones.
        # Si se solapan, prioriza no rechazar fotos validas (thr_gen).
        threshold = min(thr_gen, 0.5 * (thr_gen + thr_imp))
        # Acota a un rango razonable sobre la esfera unitaria (dist max = 2).
        threshold = float(max(0.35, min(threshold, 1.2)))
        print(f"Calibracion umbral: p95 genuinas={thr_gen:.4f}, "
              f"p5 impostoras={thr_imp:.4f} -> threshold={threshold:.4f}")
    else:
        threshold = float(max(0.35, min(thr_gen, 1.2)))
        print(f"Calibracion umbral (solo genuinas): {threshold:.4f}")

    return threshold


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar el checkpoint entrenado por dev/train.py.
    checkpoint = torch.load(ARTIFACTS_DIR / "model.pt", map_location=device,
                            weights_only=False)
    model = EmbeddingNet(embedding_dim=checkpoint["embedding_dim"], pretrained=False)
    # load_state_dict restaura exactamente los pesos aprendidos.
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    # model.eval(): BatchNorm usa estadisticas globales
    # (U2/2 Redes_Convolucionales.ipynb) y Dropout se desactiva.
    model.eval()

    # Dataset de imagenes individuales con el transform DETERMINISTA.
    dataset = CardImageDataset(transform=inference_transform)
    # workers=0 en Windows evita problemas con spawn al calibrar despues.
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Calculando embeddings de {len(dataset)} cartas en {device}...")

    embeddings, ids, names = [], [], []
    # torch.no_grad(): desactiva autograd (patron de U1/Copia de L1.ipynb).
    with torch.no_grad():
        for batch_imgs, batch_ids, batch_names in loader:
            emb = model(batch_imgs.to(device))
            embeddings.append(emb.cpu())
            ids.extend(batch_ids)
            names.extend(batch_names)

    # Tensor final [N, 128]: una fila por carta, listo para torch.cdist.
    gallery = torch.cat(embeddings, dim=0)

    # Calibrar umbral de rechazo con datos (U1/9 Seleccion de Modelos.ipynb).
    threshold = calibrate_threshold(model, dataset, gallery, names, device)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out_path = ARTIFACTS_DIR / "gallery.pt"
    torch.save({
        "embeddings": gallery,          # [N, embedding_dim]
        "ids": ids,                     # image_id de cada fila
        "names": names,                 # nombre de la carta de cada fila
        "distance_threshold": threshold,  # umbral calibrado para recognized
    }, out_path)
    print(f"Galeria guardada en {out_path} ({gallery.shape[0]} cartas, "
          f"dim {gallery.shape[1]}, threshold={threshold:.4f})")


if __name__ == "__main__":
    main()
