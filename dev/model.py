"""EmbeddingNet: la red que convierte una imagen de carta en un vector 128-D.

Arquitectura (DocumentacionRN_Yugi, seccion "La Red Siamesa"):

    imagen (3,H,W) --ResNet-18 preentrenada--> features --Linear--> 128-D --L2 norm--> embedding

1. **Extractor de caracteristicas**: una ResNet preentrenada en ImageNet.
   - Que es una convolucion, pooling, padding y stride:
     ``U2/1 Capas_Convolucionales.ipynb``.
   - La arquitectura ResNet (bloques residuales, BatchNorm):
     ``U2/2 Redes_Convolucionales.ipynb``, seccion ResNet.
   - Transfer learning (reusar pesos de ImageNet en otra tarea):
     ``U3/2_FineTuning.ipynb`` y FCC 06.
2. **Capa de embedding**: se reemplaza la capa final ``fc`` (que en ImageNet
   clasificaba 1000 clases) por una ``nn.Linear`` que proyecta a 128
   dimensiones. Es el mismo truco de U3/2_FineTuning.ipynb, donde se reemplaza
   ``fc`` para las clases nuevas; aca la salida no son clases sino un embedding.
3. **Normalizacion L2**: el vector se lleva a norma 1, asi todas las cartas
   viven en la esfera unitaria y las distancias L2 son comparables entre si
   (GuiaEstudio/06_YuGiOh_Triplet.md, checklist pregunta 1).

La "red siamesa" NO es una clase aparte: es esta misma red aplicada 3 veces
(ancla, positiva, negativa) compartiendo pesos, ver dev/train.py.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True,
                 freeze_backbone: bool = False):
        # super().__init__() registra la clase como modulo de PyTorch para que
        # .parameters(), .to(device), etc. funcionen (patron nn.Module visto
        # desde U1/4  MLP2.ipynb en adelante).
        super().__init__()

        # ResNet-18 con pesos preentrenados en ImageNet ("DEFAULT" = los mejores
        # disponibles). Empezar con una red preentrenada en vez de pesos
        # aleatorios es transfer learning: U3/2_FineTuning.ipynb muestra que
        # converge mas rapido y generaliza mejor con pocos datos.
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # in_features = 512 en ResNet-18: dimension del vector que sale del
        # average pooling global, justo antes del clasificador original.
        in_features = self.backbone.fc.in_features

        # Se REEMPLAZA la cabeza clasificadora de 1000 clases por la capa de
        # embedding (Linear 512 -> 128). Una capa lineal y = Wx + b es el bloque
        # basico visto en U1/1 Regresion_Lineal_1.ipynb (nn.Linear).
        self.backbone.fc = nn.Linear(in_features, embedding_dim)

        if freeze_backbone:
            # Congelar = requires_grad False: el autograd (U1/5  MLP4.ipynb) no
            # calcula gradientes para esos pesos y el optimizador no los toca.
            # Es la variante de la propuesta original (ResNet congelada,
            # DocumentacionRN_Yugi punto 1) y de U3/2_FineTuning.ipynb, donde el
            # backbone se ajusta poco (alli con lr chico, aca directamente fijo).
            for name, param in self.backbone.named_parameters():
                # Solo la capa de embedding nueva (fc) queda entrenable.
                if not name.startswith("fc."):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass: la imagen atraviesa las capas convolucionales
        # (U2/1 Capas_Convolucionales.ipynb), los bloques residuales
        # (U2/2 Redes_Convolucionales.ipynb) y la Linear final -> vector 128-D.
        x = self.backbone(x)
        # Normalizacion L2: divide cada vector por su norma para que ||v|| = 1.
        # Con vectores unitarios, "cerca" y "lejos" dependen solo de la
        # direccion, no de la escala, lo que estabiliza la Triplet Loss.
        return F.normalize(x, p=2, dim=1)
