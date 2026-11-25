"""Preprocesamiento compartido de radiografías y geometría para explicaciones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .data import LABELS


@dataclass(frozen=True)
class Geometry:
    original_shape: tuple[int, int]
    crop: tuple[int, int, int, int]
    padding: tuple[int, int, int, int]


def read_image(source: str | Path | object) -> np.ndarray:
    """Lee un PNG monocromático sin descartar la profundidad de bits."""
    try:
        with Image.open(source) as image:
            if image.format != "PNG":
                raise ValueError("Solo se aceptan imágenes PNG")
            if image.width * image.height > 30_000_000:
                raise ValueError("La imagen supera el límite de 30 millones de píxeles")
            if image.mode not in ("L", "I", "I;16", "I;16B", "I;16L"):
                raise ValueError(f"Se requiere un PNG monocromático, no {image.mode}")
            pixels = np.asarray(image).copy()
    except Image.DecompressionBombError as error:
        raise ValueError("La imagen excede el límite de seguridad de píxeles") from error
    if pixels.ndim != 2 or not pixels.size or pixels.min() < 0 or pixels.max() > 65535:
        raise ValueError("La imagen contiene píxeles inválidos")
    return pixels


def preprocess(
    image: np.ndarray, size: int, *, crop_black: bool = False,
    max_value: int | None = None,
) -> tuple[torch.Tensor, Geometry]:
    """Normaliza a [-1024, 1024], conserva proporciones y registra geometría."""
    pixels = np.asarray(image)
    if pixels.ndim != 2 or not pixels.size or not np.isfinite(pixels).all():
        raise ValueError("Se requiere una imagen bidimensional finita")
    if size < 1:
        raise ValueError("El tamaño debe ser positivo")
    ceiling = max_value if max_value is not None else (255 if pixels.dtype == np.uint8 else 65535)
    if ceiling <= 0 or pixels.min() < 0 or pixels.max() > ceiling:
        raise ValueError("Intensidad fuera del rango declarado")
    height, width = pixels.shape
    top, left, bottom, right = 0, 0, height, width
    if crop_black:
        black = pixels <= ceiling * 0.005
        valid_rows = np.flatnonzero(np.mean(black, axis=1) < 0.995)
        valid_cols = np.flatnonzero(np.mean(black, axis=0) < 0.995)
        if valid_rows.size and valid_cols.size:
            margin_y, margin_x = max(1, height // 100), max(1, width // 100)
            t = max(0, int(valid_rows[0]) - margin_y)
            b = min(height, int(valid_rows[-1]) + margin_y + 1)
            l = max(0, int(valid_cols[0]) - margin_x)
            r = min(width, int(valid_cols[-1]) + margin_x + 1)
            if (b - t) >= height * 0.5 and (r - l) >= width * 0.5:
                top, left, bottom, right = t, l, b, r
    cropped = pixels[top:bottom, left:right].astype(np.float32)
    h, w = cropped.shape
    side = max(h, w)
    pad_top, pad_left = (side - h) // 2, (side - w) // 2
    pad_bottom, pad_right = side - h - pad_top, side - w - pad_left
    tensor = torch.from_numpy(cropped / ceiling * 2048.0 - 1024.0)[None, None]
    tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=-1024)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0]
    return tensor, Geometry((height, width), (top, left, bottom, right),
                            (pad_top, pad_left, pad_bottom, pad_right))


def restore_map(cam: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Devuelve el mapa a las coordenadas del PNG original."""
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or not np.isfinite(cam).all():
        raise ValueError("Mapa Grad-CAM inválido")
    top, left, bottom, right = geometry.crop
    pt, pl, pb, pr = geometry.padding
    h, w = bottom - top, right - left
    resized = F.interpolate(torch.from_numpy(cam)[None, None], size=(h + pt + pb, w + pl + pr),
                            mode="bilinear", align_corners=False)[0, 0].numpy()
    result = np.zeros(geometry.original_shape, dtype=np.float32)
    result[top:bottom, left:right] = resized[pt:pt + h, pl:pl + w]
    return result


_CACHE: dict = {}
_CACHE_BYTES = 0
CACHE_BUDGET_BYTES = 6 * 1024**3


def clear_cache() -> None:
    global _CACHE_BYTES
    _CACHE.clear()
    _CACHE_BYTES = 0


def _load_tensor(path, size, crop_black, max_value, use_cache):
    """Tensor preprocesado; el caché guarda float16 (exacto para [-1024, 1024] enteros ±0.5)."""
    global _CACHE_BYTES
    key = (str(path), size, crop_black, max_value)
    if use_cache and key in _CACHE:
        return _CACHE[key].float()
    tensor, _ = preprocess(read_image(path), size, crop_black=crop_black, max_value=max_value)
    if use_cache:
        stored = tensor.half()
        if _CACHE_BYTES + stored.numel() * 2 <= CACHE_BUDGET_BYTES:
            _CACHE[key] = stored
            _CACHE_BYTES += stored.numel() * 2
            return stored.float()
    return tensor


def warm_cache(records, size: int, *, crop_black: bool = False, max_value: int | None = None,
               workers: int = 8) -> int:
    """Precarga en memoria los tensores con hilos; devuelve cuántos quedaron en caché."""
    from concurrent.futures import ThreadPoolExecutor

    paths = list(records["image_path"])
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(lambda p: _load_tensor(p, size, crop_black, max_value, True), paths))
    return sum((str(p), size, crop_black, max_value) in _CACHE for p in paths)


class XRayDataset(Dataset):
    """Devuelve (imagen, etiquetas) o (imagen, etiquetas, máscara) si ``mask`` es True.

    La máscara vale 1 si la etiqueta está confirmada (columna ``label_mask``).
    Con ``cache`` se reutilizan tensores float16 en memoria (usar num_workers=0).
    """

    def __init__(self, records, size: int, *, crop_black: bool = False,
                 augment: bool | str = False, max_value: int | None = None,
                 mask: bool = False, cache: bool = False):
        self.records = records.reset_index(drop=True)
        self.size = size
        self.crop_black = crop_black
        self.augment = {False: "none", True: "rotate"}.get(augment, augment)
        if self.augment not in ("none", "rotate", "translate", "rotate_translate"):
            raise ValueError(f"Aumento desconocido: {augment}")
        self.max_value = max_value
        self.mask = mask
        self.cache = cache
        if mask:
            from .data import label_mask

            self.known = label_mask(self.records).to_numpy(dtype="float32")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        tensor = _load_tensor(row["image_path"], self.size, self.crop_black,
                              self.max_value, self.cache)
        if self.augment in ("rotate", "rotate_translate"):
            angle = float(torch.empty(1).uniform_(-2, 2).item())
            tensor = TF.rotate(tensor, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=-1024)
        if self.augment in ("translate", "rotate_translate"):
            limit = self.size * 0.02
            offsets = torch.empty(2).uniform_(-limit, limit).round().int().tolist()
            tensor = TF.affine(tensor, angle=0, translate=offsets, scale=1, shear=0,
                               interpolation=TF.InterpolationMode.BILINEAR, fill=-1024)
        labels = torch.tensor([float(row[label]) for label in LABELS], dtype=torch.float32)
        if self.mask:
            return tensor, labels, torch.from_numpy(self.known[index].copy())
        return tensor, labels
