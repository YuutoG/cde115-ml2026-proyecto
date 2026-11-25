"""Preparación reproducible de cohortes externas publicadas en Kaggle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import tempfile
from pathlib import Path

import kagglehub
import pandas as pd
from PIL import Image, ImageChops


SOURCES = {
    "nih": ("nih-chest-xrays/sample/versions/4", "sample_labels.csv", "sample/images"),
    "shenzhen": ("raddar/tuberculosis-chest-xrays-shenzhen/versions/1",
                 "shenzhen_metadata.csv", "images/images"),
    "montgomery": ("raddar/tuberculosis-chest-xrays-montgomery/versions/1",
                   "montgomery_metadata.csv", "images/images"),
}
LABELS = ("pneumonia", "pleural_effusion", "tuberculosis", "edema")
ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(source: str, csv_path: Path):
    data = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    records = []
    seen = set()
    excluded = 0
    for row in data.to_dict("records"):
        if source == "nih":
            name, age = row["Image Index"], row["Patient Age"]
            match = re.fullmatch(r"(\d{1,3})Y", age)
            group = row["Patient ID"]
            if not re.fullmatch(r"\d+", group):
                raise ValueError(f"Identificador NIH inválido para {name}")
            findings = set(row["Finding Labels"].split("|"))
            if not findings or "" in findings or ("No Finding" in findings and len(findings) > 1):
                raise ValueError(f"Hallazgos NIH inválidos para {name}")
            labels = {"pneumonia": int("Pneumonia" in findings),
                      "pleural_effusion": int("Effusion" in findings),
                      "edema": int("Edema" in findings), "tuberculosis": ""}
            sex = row["Patient Gender"]
            patient_group = f"nih-{group}"
        else:
            name, age = row["study_id"], row["age"]
            match = re.fullmatch(r"(\d{1,3})", age)
            prefix = "CHNCXR" if source == "shenzhen" else "MCUCXR"
            image_match = re.fullmatch(rf"{prefix}_(\d{{4}})_([01])\.png", name)
            if image_match is None:
                raise ValueError(f"Nombre {source} inválido: {name}")
            labels = {label: "" for label in LABELS}
            labels["tuberculosis"] = int(image_match.group(2))
            sex = row["sex" if source == "shenzhen" else "gender"]
            # No hay identificador de paciente verificado: no inferir uno del nombre.
            patient_group = source
        if not re.fullmatch(r"[A-Za-z0-9_]+\.png", name) or name in seen:
            raise ValueError(f"Nombre de imagen inválido o duplicado: {name}")
        seen.add(name)
        if match is None or not 15 <= int(match.group(1)) <= 120:
            excluded += 1
            continue
        records.append({"file_name": name, "age": int(match.group(1)),
                        "group_id": patient_group, "sex": sex, **labels})
    if not records:
        raise ValueError(f"No hay imágenes elegibles en {source}")
    return records, excluded


def _copy_image(source: Path, destination: Path, max_value: int | None = None) -> tuple[str, str, int]:
    original_hash = _sha256(source)
    with Image.open(source) as image:
        if image.format != "PNG" or image.width * image.height > 30_000_000:
            raise ValueError(f"PNG inválido o demasiado grande: {source}")
        if image.mode == "P":
            palette = image.getpalette()
            if (palette is None or len(palette) % 3 or "transparency" in image.info
                    or any(palette[i] != palette[i + 1] or palette[i] != palette[i + 2]
                           for i in range(0, len(palette), 3))):
                raise ValueError(f"Paleta no monocromática: {source}")
            image.convert("L").save(destination)
            ceiling = 255
        elif image.mode in ("RGB", "RGBA", "LA"):
            bands = image.split()
            if image.mode in ("RGBA", "LA") and bands[-1].getextrema() != (255, 255):
                raise ValueError(f"Transparencia en {source}")
            if image.mode != "LA" and (
                    ImageChops.difference(bands[0], bands[1]).getbbox()
                    or ImageChops.difference(bands[0], bands[2]).getbbox()):
                raise ValueError(f"Imagen no monocromática: {source}")
            bands[0].save(destination)
            ceiling = 255
        elif image.mode in ("L", "I", "I;16", "I;16B", "I;16L"):
            if image.mode != "L" and max_value is None:
                raise ValueError(f"Indique --max-value verificado para el PNG de 16 bits {source}")
            ceiling = 255 if image.mode == "L" else max_value
            low, high = image.getextrema()
            if low < 0 or high > ceiling:
                raise ValueError(f"Intensidad inválida en {source}")
            shutil.copyfile(source, destination)
        else:
            raise ValueError(f"Modo de imagen no monocromático {image.mode}: {source}")
    if max_value is not None and ceiling != max_value:
        raise ValueError(f"El rango declarado no coincide con la profundidad de {source}")
    return original_hash, _sha256(destination), ceiling


def prepare_source(source: str, output_root: Path, max_images: int | None, seed: int,
                   max_value: int | None = None):
    handle, metadata_name, image_subdir = SOURCES[source]
    target = output_root / source
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"El destino ya existe y no se sobrescribirá: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".external-{source}-", dir=output_root) as tmp:
        temporary = Path(tmp)
        downloaded = Path(kagglehub.dataset_download(handle, output_dir=str(temporary / "download")))
        if not downloaded.resolve().is_relative_to(temporary.resolve()):
            raise RuntimeError(
                "Kagglehub ignoró output_dir; no se usará ni borrará una caché compartida"
            )
        records, excluded = _records(source, downloaded / metadata_name)
        available = len(records)
        if max_images is not None and max_images < available:
            records = random.Random(seed).sample(records, max_images)
        records.sort(key=lambda row: row["file_name"])
        staged = temporary / "result"
        image_dir = staged / "images"
        image_dir.mkdir(parents=True)
        ceilings = set()
        hashes = set()
        for row in records:
            input_path = downloaded / image_subdir / row["file_name"]
            if not input_path.is_file():
                raise FileNotFoundError(f"Falta la imagen {input_path}")
            original_hash, output_hash, ceiling = _copy_image(
                input_path, image_dir / row["file_name"], max_value)
            if original_hash in hashes:
                raise ValueError(f"Imagen duplicada byte a byte: {input_path}")
            hashes.add(original_hash)
            ceilings.add(ceiling)
            row.update(original_sha256=original_hash, sha256=output_hash)
        if len(ceilings) != 1:
            raise ValueError(f"Profundidades mezcladas en {source}; separar antes de evaluar")
        with (staged / "labels.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        metadata = {
            "kaggle_handle_version": handle, "source": source, "seed": seed,
            "max_images": max_images, "available_eligible": available,
            "excluded_invalid_or_underage": excluded, "selected": len(records),
            "external_max_value": ceilings.pop(),
            "known_counts": {label: sum(row[label] != "" for row in records) for label in LABELS},
            "positive_counts": {label: sum(row[label] == 1 for row in records) for label in LABELS},
            "grouping": "patient_id" if source == "nih" else "entire_source_patient_ids_unverified",
            "warning": "Verificar autorización, procedencia, solapamiento y referencia clínica.",
        }
        (staged / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        # mkdir reserva el nombre sin sobrescribir incluso ante ejecuciones concurrentes.
        target.mkdir()
        try:
            for entry in staged.iterdir():
                shutil.move(str(entry), target / entry.name)
        except Exception as error:
            raise RuntimeError(f"Salida incompleta conservada en {target}; revísela: {error}") from error
    print(f"{source}: {len(records)} imágenes en {target}; excluidas por edad: {excluded}")
    return metadata


def main(family: str):
    parser = argparse.ArgumentParser(description=f"Prepara cohortes externas para {family}.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / f"external_{family}")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Límite aleatorio por fuente; por defecto, todas las imágenes elegibles.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-value", type=int,
                        help="Rango máximo verificado; obligatorio para PNG de 16 bits.")
    if family == "xrv":
        parser.add_argument("--confirm-authorized", action="store_true",
                            help="Confirma que dispone de autorización para usar las copias NLM.")
        parser.add_argument("--source", choices=("shenzhen", "montgomery", "both"), default="both")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images debe ser positivo")
    if args.max_value is not None and not 1 <= args.max_value <= 65535:
        parser.error("--max-value debe estar entre 1 y 65535")
    if family == "xrv" and not args.confirm_authorized:
        parser.error("Confirme la autorización de uso de las copias NLM con --confirm-authorized")
    sources = ["nih"] if family == "imagenet" else (
        ["shenzhen", "montgomery"] if args.source == "both" else [args.source])
    for source in sources:
        prepare_source(source, args.output_dir, args.max_images, args.seed, args.max_value)
