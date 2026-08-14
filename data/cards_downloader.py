"""Descarga del dataset de cartas Yu-Gi-Oh! desde la API de YGOPRODeck.

Corresponde al script ``cards_downloader.py`` descripto en la propuesta
``DocumentacionRN_Yugi`` (seccion "Descargar Dataset"): la API
https://db.ygoprodeck.com/api/v7/cardinfo.php devuelve un JSON con todas las
cartas, y cada carta trae una o mas imagenes (``card_images``) con su URL.

Diferencias con el prototipo Colab:
- Se persiste ``cards.json`` ademas de las imagenes, para que los Dataset de
  ``dev/dataset.py`` puedan mapear id -> nombre sin volver a llamar a la API.
- Se puede limitar la cantidad de cartas (``--max-cards``) para entrenar con un
  subset manejable, como recomienda la guia ``GuiaEstudio/06_YuGiOh_Triplet.md``
  ("Completar descarga antes de entrenar en serio" aplica al dataset completo).

La idea de armar un dataset propio de imagenes descargadas de internet es la
misma que se practica en FCC 04 (custom datasets) y en el Lab
``U1/Copia de L1.ipynb`` Parte 1 (``PlatesDataSet``).
"""

import argparse
import json
import time
from pathlib import Path

import requests

# URL de la API publica de YGOPRODeck (misma que usa el prototipo y la propuesta).
API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

# Rutas relativas a esta carpeta data/, para que el script funcione desde cualquier cwd.
DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR / "yugioh_card_images"
CARDS_JSON = DATA_DIR / "cards.json"

# Pausa entre descargas para no saturar el servidor (rate limiting basico).
REQUEST_DELAY = 0.05
TIMEOUT = 15


def fetch_card_list() -> list[dict]:
    """Pide a la API el listado completo de cartas y devuelve la lista de dicts."""
    print(f"Consultando API: {API_URL}")
    response = requests.get(API_URL, timeout=TIMEOUT)
    # raise_for_status corta el programa si la API devolvio un error HTTP.
    response.raise_for_status()
    # El JSON tiene la forma {"data": [ {carta}, {carta}, ... ]}.
    return response.json()["data"]


def download_images(cards: list[dict]) -> None:
    """Descarga la imagen 'small' de cada carta a OUTPUT_DIR/{image_id}.jpg.

    Cada carta puede tener varias ilustraciones (artes alternativas) en
    ``card_images``; se descargan todas porque son "positivas" naturales para
    la Triplet Loss (misma carta, distinta vista), ver
    ``GuiaEstudio/06_YuGiOh_Triplet.md`` seccion "Triplet Loss".
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = skipped = errors = 0
    total = sum(len(card.get("card_images", [])) for card in cards)

    for card in cards:
        for image_info in card.get("card_images", []):
            # El nombre de archivo es el id de la IMAGEN (no el de la carta),
            # asi las artes alternativas no se pisan entre si.
            image_id = image_info["id"]
            dest = OUTPUT_DIR / f"{image_id}.jpg"

            # Descarga reanudable: si el archivo ya existe y no esta vacio, se salta.
            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue

            try:
                # 'image_url_small' (~268x391 px) alcanza para el modelo y pesa poco;
                # 'image_url' seria la version HD.
                url = image_info["image_url_small"]
                with requests.get(url, timeout=TIMEOUT, stream=True) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        # Escritura en bloques para no cargar toda la imagen en RAM.
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                downloaded += 1
            except requests.RequestException as exc:
                errors += 1
                print(f"  Error con imagen {image_id}: {exc}")
                # Borra el archivo parcial para que el reintento no lo saltee.
                dest.unlink(missing_ok=True)
                time.sleep(1.0)

            time.sleep(REQUEST_DELAY)

            done = downloaded + skipped + errors
            if done % 100 == 0:
                print(f"  Progreso: {done}/{total} imagenes")

    print(f"Listo: {downloaded} descargadas, {skipped} ya existian, {errors} errores.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el dataset de cartas Yu-Gi-Oh!")
    # Subset parametrizable: 800 cartas por defecto para una demo local rapida.
    # Con --max-cards 0 se descarga el dataset completo (~13.000 cartas) como
    # plantea la propuesta original.
    parser.add_argument("--max-cards", type=int, default=800,
                        help="Cantidad maxima de cartas a usar (0 = todas)")
    args = parser.parse_args()

    cards = fetch_card_list()
    print(f"La API devolvio {len(cards)} cartas.")

    if args.max_cards > 0:
        # Se toman las primeras N (la API las devuelve en orden estable por nombre),
        # suficiente variedad visual para aprender el espacio de embeddings.
        cards = cards[: args.max_cards]
        print(f"Usando subset de {len(cards)} cartas (--max-cards {args.max_cards}).")

    # Se guarda el JSON con los metadatos ANTES de descargar imagenes:
    # dev/dataset.py lo necesita para mapear image_id -> nombre de carta.
    # (El prototipo Colab no lo persistia y dependia de la variable en memoria.)
    with open(CARDS_JSON, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False)
    print(f"Metadatos guardados en {CARDS_JSON}")

    download_images(cards)


if __name__ == "__main__":
    main()
