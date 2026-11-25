"""Modelos XRV, artefactos y explicaciones compartidos por notebook y app."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from .data import LABELS, _sha256_file
from .imaging import preprocess, restore_map

MODEL_SPECS = {
    "resnet50": ("resnet50-res512-all", 512),
    "densenet_all": ("densenet121-res224-all", 224),
    "densenet_pc": ("densenet121-res224-pc", 224),
}
# Familia ImageNet: (arquitectura torchvision, tamaño de entrada).
IMAGENET_SPECS = {
    "resnet50_in": ("resnet50", 512),
    "densenet121_in": ("densenet121", 384),
    "efficientnet_b0_in": ("efficientnet_b0", 384),
}
FAMILY_MODELS = {"xrv": tuple(MODEL_SPECS), "imagenet": tuple(IMAGENET_SPECS)}
IMAGE_SIZES = {**{k: v[1] for k, v in MODEL_SPECS.items()},
               **{k: v[1] for k, v in IMAGENET_SPECS.items()}}


class XRayModel(nn.Module):
    def __init__(self, name: str, *, pretrained: bool = True, cache_dir: str | None = None):
        super().__init__()
        if name not in MODEL_SPECS:
            raise ValueError(f"Modelo desconocido: {name}")
        weights, self.size = MODEL_SPECS[name]
        self.name = name
        if name == "resnet50":
            if pretrained:
                self.core = xrv.models.ResNet(weights=weights, cache_dir=cache_dir).model
            else:
                self.core = torchvision.models.resnet50(weights=None)
                self.core.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.core.fc = nn.Linear(self.core.fc.in_features, len(LABELS))
        else:
            self.core = xrv.models.DenseNet(weights=weights if pretrained else None, cache_dir=cache_dir)
            self.core.classifier = nn.Linear(self.core.classifier.in_features, len(LABELS))
            self.core.op_threshs = None
            self.core.apply_sigmoid = False
            self.core.input_resolution = self.size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.core(image)

    def cam_layer(self):
        if self.name == "resnet50":
            return self.core.layer4[-1]
        return self.core.features.denseblock4

    def head_parameters(self):
        head = self.core.fc if self.name == "resnet50" else self.core.classifier
        return list(head.parameters())


class ImageNetModel(nn.Module):
    """Backbone torchvision con pesos ImageNet y cabeza de 4 logits.

    Recibe el mismo tensor monocanal en [-1024, 1024] que los modelos XRV y lo
    convierte a 3 canales con la normalización de ImageNet.
    """

    def __init__(self, name: str, *, pretrained: bool = True, cache_dir: str | None = None):
        super().__init__()
        if name not in IMAGENET_SPECS:
            raise ValueError(f"Modelo desconocido: {name}")
        architecture, self.size = IMAGENET_SPECS[name]
        self.name = name
        self.architecture = architecture
        weights = "DEFAULT" if pretrained else None
        self.core = getattr(torchvision.models, architecture)(weights=weights)
        if architecture == "resnet50":
            self.core.fc = nn.Linear(self.core.fc.in_features, len(LABELS))
        elif architecture == "densenet121":
            self.core.classifier = nn.Linear(self.core.classifier.in_features, len(LABELS))
        else:
            self.core.classifier[-1] = nn.Linear(self.core.classifier[-1].in_features, len(LABELS))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        unit = (image / 2048.0 + 0.5).clamp(0, 1).expand(-1, 3, -1, -1)
        return self.core((unit - self.mean) / self.std)

    def cam_layer(self):
        if self.architecture == "resnet50":
            return self.core.layer4[-1]
        if self.architecture == "densenet121":
            return self.core.features.denseblock4
        return self.core.features[-1]

    def head_parameters(self):
        head = self.core.fc if self.architecture == "resnet50" else self.core.classifier
        return list(head.parameters())


def build_model(name: str, *, pretrained: bool = True, cache_dir: str | None = None):
    if name in MODEL_SPECS:
        return XRayModel(name, pretrained=pretrained, cache_dir=cache_dir)
    if name in IMAGENET_SPECS:
        return ImageNetModel(name, pretrained=pretrained, cache_dir=cache_dir)
    raise ValueError(f"Modelo desconocido: {name}")


def save_checkpoint(path: str | Path, model) -> None:
    torch.save({"name": model.name, "labels": list(LABELS), "state_dict": model.state_dict()}, path)


def _validate_thresholds(values) -> None:
    if not isinstance(values, list) or len(values) != len(LABELS):
        raise ValueError("Cantidad de umbrales inválida")
    if any(value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 1)
           for value in values):
        raise ValueError("Umbral inválido")


def load_models(model_dir: str | Path, device: str | torch.device = "cpu"):
    directory = Path(model_dir)
    with (directory / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    family = manifest.get("family", "xrv")
    if family not in FAMILY_MODELS:
        raise ValueError("Manifiesto incompatible: familia desconocida")
    names = FAMILY_MODELS[family]
    if manifest.get("labels") != list(LABELS) or set(manifest.get("models", {})) != set(names):
        raise ValueError("Manifiesto incompatible: etiquetas o modelos")
    thresholds = manifest.get("thresholds", {})
    if set(thresholds) != {*names, "ensemble"}:
        raise ValueError("Manifiesto incompatible: faltan umbrales")
    for values in thresholds.values():
        _validate_thresholds(values)
    weights = np.asarray(manifest.get("ensemble_weights"), dtype=float)
    if (weights.shape != (len(names),) or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1)):
        raise ValueError("Pesos del ensemble inválidos")
    loaded = {}
    for name in names:
        entry = manifest["models"][name]
        if entry.get("file") != f"{name}.pt":
            raise ValueError("Nombre de checkpoint inválido")
        if _sha256_file(directory / entry["file"]) != entry.get("sha256"):
            raise ValueError(f"El checkpoint no coincide con el manifiesto: {name}")
        preprocessing = entry.get("preprocessing", {})
        if not isinstance(preprocessing.get("crop_black"), bool) or preprocessing.get("max_value") is not None:
            raise ValueError("Configuración de preprocesamiento incompatible")
        temperatures = entry.get("temperatures", [1.0] * len(LABELS))
        if (len(temperatures) != len(LABELS)
                or not all(isinstance(t, (int, float)) and np.isfinite(t) and t > 0 for t in temperatures)):
            raise ValueError("Temperaturas inválidas")
        checkpoint = torch.load(directory / entry["file"], map_location="cpu", weights_only=True)
        if checkpoint.get("name") != name or checkpoint.get("labels") != list(LABELS):
            raise ValueError(f"Checkpoint incompatible: {name}")
        model = build_model(name, pretrained=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        loaded[name] = model
    return loaded, manifest


def member_probabilities(model, batch: torch.Tensor, temperatures, tta: bool = False) -> np.ndarray:
    """Probabilidades calibradas sigmoid(logit/T) de un modelo, con TTA de espejo opcional."""
    with torch.no_grad():
        logits = model(batch)
        if tta:
            logits = 0.5 * (logits + model(torch.flip(batch, dims=[-1])))
    scaled = logits[0].double().cpu().numpy() / np.asarray(temperatures, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-scaled))


def predict_image(models, manifest, image: np.ndarray, label_index: int | None = None, device="cpu"):
    """Calcula probabilidades calibradas y Grad-CAM por miembro, más fusión heurística.

    Con ``label_index=None`` los mapas corresponden a la mayor puntuación del
    ensemble (el primer índice en caso de empate). Un mapa degenerado se devuelve
    como ``None`` y la fusión también es ``None``.
    """
    if label_index is not None and not 0 <= label_index < len(LABELS):
        raise ValueError("Etiqueta inválida")
    probabilities, maps, inputs = {}, {}, {}
    for name, model in models.items():
        entry = manifest["models"][name]
        config = entry["preprocessing"]
        tensor, geometry = preprocess(image, IMAGE_SIZES[name], crop_black=config["crop_black"],
                                      max_value=config.get("max_value"))
        batch = tensor.unsqueeze(0).to(device)
        probabilities[name] = member_probabilities(
            model, batch, entry.get("temperatures", [1.0] * len(LABELS)), bool(entry.get("tta")))
        if (probabilities[name].shape != (len(LABELS),)
                or not np.isfinite(probabilities[name]).all()):
            raise ValueError(f"Probabilidades inválidas para {name}")
        inputs[name] = (batch, geometry)
    weights = np.asarray(manifest["ensemble_weights"])
    combined_probability = sum(weights[i] * probabilities[name] for i, name in enumerate(models))
    if label_index is None:
        label_index = int(np.argmax(combined_probability))
    for name, model in models.items():
        batch, geometry = inputs[name]
        with torch.enable_grad(), GradCAM(model=model, target_layers=[model.cam_layer()]) as gradcam:
            cam = gradcam(input_tensor=batch, targets=[ClassifierOutputTarget(label_index)])[0]
        if not np.isfinite(cam).all() or float(cam.max()) <= 0:
            maps[name] = None
        else:
            maps[name] = restore_map(cam / cam.max(), geometry)
    combined_map = (None if any(m is None for m in maps.values())
                    else sum(weights[i] * maps[name] for i, name in enumerate(models)))
    return probabilities, combined_probability, maps, combined_map


def display_background(image: np.ndarray) -> np.ndarray:
    """Fondo RGB en [0, 1] con contraste por percentiles 1–99."""
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        raise ValueError("La radiografía no tiene contraste suficiente para visualizarla")
    gray = np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)
    return np.repeat(gray[..., None], 3, axis=2)


def overlay_map(background: np.ndarray, cam: np.ndarray, *, opacity: float = 0.65,
                visual_threshold: float = 0.005) -> np.ndarray:
    """Superpone un Grad-CAM suavizado (colormap turbo) sobre el fondo RGB."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    smooth = np.clip(gaussian_filter(cam, sigma=0.005 * min(cam.shape)), 0, 1)
    color = plt.get_cmap("turbo")(smooth)[..., :3]
    alpha = opacity * np.clip((smooth - visual_threshold) / (1 - visual_threshold), 0, 1)
    return np.clip(background * (1 - alpha[..., None]) + color * alpha[..., None], 0, 1)
