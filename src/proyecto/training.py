"""Herramientas independientes para clasificación multietiqueta de cuatro etiquetas.

Los loaders producen ``(entradas, etiquetas)`` o ``(entradas, etiquetas, máscara)``:
entradas es un tensor que se pasa directamente al modelo, etiquetas tiene forma
(N, 4) con valores 0/1 y la máscara opcional (N, 4) marca con 1 las etiquetas
confirmadas. Las posiciones con máscara 0 no entran en la pérdida ni en las
métricas. El modelo devuelve logits (N, 4). El llamador es responsable de
construir los datasets, preprocesamiento y aumentos sin fuga de información.

Flujo: ``search_grouped_cv`` únicamente con entrenamiento y folds agrupados;
``train_final`` con los parámetros y épocas elegidos; ``predict`` sobre un
conjunto de calibración independiente; ``calibrate_thresholds``; finalmente
``predict`` y ``evaluate_multilabel`` en test, sin ajustar nada allí.
"""

from __future__ import annotations

import copy
import inspect
import random
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from .calibration import expected_calibration_error

N_LABELS = 4


def seed_everything(seed: int) -> None:
    """Fija semillas de Python, NumPy y PyTorch; no garantiza determinismo GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _labels(y: Any) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != N_LABELS or len(y) == 0:
        raise ValueError("Las etiquetas deben tener forma (N, 4), con N > 0")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Las etiquetas deben ser binarias y no contener ausentes")
    return y.astype(np.int64)


def _mask(mask: Any, y: np.ndarray) -> np.ndarray:
    """Valida la máscara de etiquetas confirmadas; ``None`` confirma todo."""
    if mask is None:
        return np.ones_like(y)
    m = np.asarray(mask)
    if m.shape != y.shape or not np.all(np.isin(m, [0, 1])):
        raise ValueError("La máscara debe ser binaria y tener la forma de las etiquetas")
    return m.astype(np.int64)


def _probabilities(y: np.ndarray, probabilities: Any) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Las probabilidades deben ser finitas, en [0, 1] y de forma (N, 4)")
    return p


def pos_weight_from_labels(y_train: Any, mask: Any = None) -> torch.Tensor:
    """Devuelve negativos/positivos confirmados por etiqueta; exige ambas clases."""
    y = _labels(y_train)
    m = _mask(mask, y)
    positives = (y * m).sum(axis=0)
    negatives = ((1 - y) * m).sum(axis=0)
    missing = np.flatnonzero((positives == 0) | (negatives == 0))
    if len(missing):
        raise ValueError(f"Entrenamiento sin ambas clases en etiquetas {missing.tolist()}")
    return torch.tensor(negatives / positives, dtype=torch.float32)


class AsymmetricLoss(nn.Module):
    """Pérdida asimétrica por clase con refuerzo positivo; devuelve valores por elemento."""

    def __init__(self, gamma_neg, gamma_pos, clip, pos_weight_boost,
                 label_smoothing: float = 0.01, eps: float = 1e-8):
        super().__init__()
        for name, value in (("gamma_neg", gamma_neg), ("gamma_pos", gamma_pos),
                            ("clip", clip), ("pos_weight_boost", pos_weight_boost)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.label_smoothing = float(label_smoothing)
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smooth = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        xs_pos = torch.sigmoid(logits)
        xs_neg = (1.0 - xs_pos + self.clip).clamp(max=1.0)
        loss = (self.pos_weight_boost * smooth * torch.log(xs_pos.clamp(min=self.eps))
                + (1.0 - smooth) * torch.log(xs_neg.clamp(min=self.eps)))
        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        return -loss * torch.pow((1.0 - pt).clamp(min=self.eps), gamma)


class MaskedLoss(nn.Module):
    """Promedia una pérdida elemento a elemento solo sobre etiquetas confirmadas."""

    def __init__(self, elementwise: nn.Module):
        super().__init__()
        self.elementwise = elementwise

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.elementwise(logits, targets)
        if mask is None:
            return loss.mean()
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def build_criterion(kind: str, y_train: Any, mask: Any = None,
                    device: str | torch.device = "cpu") -> MaskedLoss:
    """Crea la pérdida ``bce`` (ponderada) o ``asl`` (asimétrica) según prevalencia.

    En ``asl`` los exponentes y el recorte dependen de la prevalencia confirmada
    de cada etiqueta (más énfasis en negativos fáciles para etiquetas raras) y el
    refuerzo positivo es ``clip((neg/pos) ** 0.25, 1, 1.65)``.
    """
    y = _labels(y_train)
    weights = pos_weight_from_labels(y, mask)
    if kind == "bce":
        base: nn.Module = nn.BCEWithLogitsLoss(pos_weight=weights, reduction="none")
    elif kind == "asl":
        m = _mask(mask, y)
        prevalence = (y * m).sum(axis=0) / np.maximum(m.sum(axis=0), 1)
        gamma_neg, gamma_pos, clip = [], [], []
        for value in prevalence:
            if value >= 0.35:
                gamma_neg.append(1.5), gamma_pos.append(0.5), clip.append(0.01)
            elif value >= 0.15:
                gamma_neg.append(2.0), gamma_pos.append(0.0), clip.append(0.02)
            else:
                gamma_neg.append(2.5), gamma_pos.append(0.0), clip.append(0.03)
        boost = np.clip(np.power(weights.numpy(), 0.25), 1.0, 1.65)
        base = AsymmetricLoss(gamma_neg, gamma_pos, clip, boost)
    else:
        raise ValueError(f"Pérdida desconocida: {kind}")
    return MaskedLoss(base).to(device)


def _batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
        raise ValueError("Cada batch debe ser un par (entradas, etiquetas) o una terna con máscara")
    x, y = batch[0], batch[1]
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Entradas y etiquetas deben ser tensores")
    y = y.to(device=device, dtype=torch.float32)
    if y.ndim != 2 or y.shape[1] != N_LABELS or not torch.all((y == 0) | (y == 1)):
        raise ValueError("Cada batch de etiquetas debe ser binario y tener forma (N, 4)")
    if len(batch) == 3:
        m = batch[2].to(device=device, dtype=torch.float32)
        if m.shape != y.shape or not torch.all((m == 0) | (m == 1)):
            raise ValueError("La máscara del batch debe ser binaria y de forma (N, 4)")
    else:
        m = torch.ones_like(y)
    return x.to(device), y, m


def _amp_setup(device: torch.device, amp: bool) -> tuple[torch.dtype | None, Any]:
    """Elige bfloat16 si la GPU lo soporta; si no, float16 con GradScaler."""
    if not amp or device.type != "cuda":
        return None, None
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


def _forward(model: nn.Module, x: torch.Tensor, device: torch.device,
             dtype: torch.dtype | None, *, tta: bool = False) -> torch.Tensor:
    """Logits en float32; con ``tta`` promedia el original y su espejo horizontal."""
    if dtype is not None:
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        logits = model(x)
        if tta:
            logits = 0.5 * (logits + model(torch.flip(x, dims=[-1])))
    return logits.float()


def _run_epoch(model: nn.Module, loader: Iterable, criterion: nn.Module,
               device: torch.device, optimizer: torch.optim.Optimizer | None = None,
               *, amp: bool = False, collect: bool = False):
    """Una época; con ``collect`` devuelve además (etiquetas, probabilidades, máscara)."""
    training = optimizer is not None
    model.train(training)
    dtype, scaler = _amp_setup(device, amp)
    total, count = 0.0, 0
    ys, ps, ms = [], [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            x, y, m = _batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, x, device, dtype)
            if logits.shape != y.shape or not torch.isfinite(logits).all():
                raise ValueError("El modelo debe devolver logits finitos de forma (N, 4)")
            loss = criterion(logits, y, m)
            if not torch.isfinite(loss):
                raise ValueError("La pérdida no es finita")
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
            if collect:
                ys.append(y.cpu().numpy())
                ps.append(torch.sigmoid(logits).detach().cpu().numpy())
                ms.append(m.cpu().numpy())
    if not count:
        raise ValueError("El loader está vacío")
    mean_loss = total / count
    if not collect:
        return mean_loss
    return (mean_loss, np.concatenate(ys).astype(np.int64), np.concatenate(ps),
            np.concatenate(ms).astype(np.int64))


def macro_metric(y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None,
                 metric: str = "auprc") -> tuple[float, list[float]]:
    """Media macro de AUPRC/AUROC por etiqueta sobre muestras confirmadas.

    Ignora etiquetas sin ambas clases. Devuelve NaN si ninguna es definida.
    """
    function = average_precision_score if metric == "auprc" else roc_auc_score
    m = _mask(mask, y)
    values = []
    for j in range(N_LABELS):
        known = m[:, j] == 1
        truth = y[known, j]
        if truth.min(initial=1) == truth.max(initial=0) or not truth.size:
            values.append(float("nan"))
        else:
            values.append(float(function(truth, p[known, j])))
    finite = [v for v in values if np.isfinite(v)]
    return (float(np.mean(finite)) if finite else float("nan")), values


def _cosine(optimizer: torch.optim.Optimizer, steps: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=1e-6)


def fit(
    model: nn.Module, train_loader: Iterable, y_train: Any,
    optimizer: torch.optim.Optimizer, *, epochs: int, device: str | torch.device = "cpu",
    val_loader: Iterable | None = None, patience: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    mask_train: Any = None, loss: str = "bce", warmup_epochs: int = 0,
    head_parameters: Sequence[nn.Parameter] | None = None, warmup_lr: float = 3e-4,
    monitor: str = "val_loss", cosine: bool = False, amp: bool = False,
) -> dict[str, Any]:
    """Entrena logits con pérdida enmascarada en una o dos etapas.

    ``y_train``/``mask_train`` deben corresponder a las muestras del loader de
    entrenamiento; solo calculan pesos de la pérdida. Con ``warmup_epochs`` el
    cuerpo del modelo permanece congelado y solo se entrena ``head_parameters``;
    después se usa ``optimizer`` (p. ej. con LR diferencial) y, con ``cosine``,
    un decaimiento coseno. La paciencia solo cuenta tras el calentamiento.

    Con validación se restaura el mejor estado según ``monitor``: ``val_loss``
    (mínimo) o ``pr_auc`` (máximo AUPRC macro en etiquetas confirmadas). Sin
    validación se ejecutan exactamente ``epochs`` y no hay selección.
    Devuelve historial y mejor época (base 1). ``progress_callback`` recibe
    pérdidas, métricas y tiempos al terminar cada época.
    """
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs debe ser un entero positivo")
    if patience is not None and (not isinstance(patience, int) or patience < 1 or val_loader is None):
        raise ValueError("patience requiere validación y debe ser un entero positivo")
    if monitor not in ("val_loss", "pr_auc"):
        raise ValueError("monitor debe ser 'val_loss' o 'pr_auc'")
    if warmup_epochs < 0 or (warmup_epochs and warmup_epochs >= epochs):
        raise ValueError("warmup_epochs debe ser menor que epochs")
    if warmup_epochs and not head_parameters:
        raise ValueError("warmup_epochs requiere head_parameters")
    device = torch.device(device)
    model.to(device)
    if amp and device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    criterion = build_criterion(loss, y_train, mask_train, device)
    extra = {"amp": amp} if amp else {}
    history: list[dict[str, float]] = []
    best_score, best_epoch, best_state, stale = None, 0, None, 0
    best_loss = float("inf")
    started = time.perf_counter()

    warmup_optimizer = None
    frozen: list[nn.Parameter] = []
    if warmup_epochs:
        head_ids = {id(p) for p in head_parameters}
        frozen = [p for p in model.parameters() if id(p) not in head_ids]
        for p in frozen:
            p.requires_grad = False
        accepted = inspect.signature(type(optimizer).__init__).parameters
        defaults = {k: v for k, v in optimizer.defaults.items() if k in accepted}
        warmup_optimizer = type(optimizer)(list(head_parameters), **defaults)
        for group in warmup_optimizer.param_groups:
            group["lr"] = warmup_lr
    scheduler = None
    active = warmup_optimizer or optimizer
    if not warmup_epochs and cosine:
        scheduler = _cosine(optimizer, epochs)

    for epoch in range(1, epochs + 1):
        if warmup_epochs and epoch == warmup_epochs + 1:
            for p in frozen:
                p.requires_grad = True
            active = optimizer
            scheduler = _cosine(optimizer, epochs - warmup_epochs) if cosine else None
        in_warmup = epoch <= warmup_epochs
        epoch_started = time.perf_counter()
        train_loss = _run_epoch(model, train_loader, criterion, device, active, **extra)
        if scheduler is not None:
            scheduler.step()
        row = {"train_loss": train_loss, "stage": "warmup" if in_warmup else "fine_tuning"}
        improved = False
        if val_loader is not None:
            if monitor == "pr_auc":
                val_loss, vy, vp, vm = _run_epoch(model, val_loader, criterion, device,
                                                  collect=True, **extra)
                pr_auc, _ = macro_metric(vy, vp, vm, "auprc")
                row["val_pr_auc"] = pr_auc
                score = pr_auc if np.isfinite(pr_auc) else -1.0
                improved = best_score is None or score > best_score + 1e-4
            else:
                val_loss = _run_epoch(model, val_loader, criterion, device, **extra)
                score = -val_loss
                improved = best_score is None or score > best_score
            row["val_loss"] = val_loss
            if improved:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            elif not in_warmup:
                stale += 1
            best_loss = min(best_loss, val_loss)
        history.append(row)
        early_stopped = patience is not None and stale >= patience
        if progress_callback is not None:
            progress_callback({
                "event": "epoch",
                "epoch": epoch,
                "total_epochs": epochs,
                "stage": row["stage"],
                "train_loss": train_loss,
                "val_loss": row.get("val_loss"),
                "val_pr_auc": row.get("val_pr_auc"),
                "best_epoch": best_epoch if val_loader is not None else epoch,
                "best_val_loss": best_loss if val_loader is not None else None,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "elapsed_seconds": time.perf_counter() - started,
                "early_stopped": early_stopped,
            })
        if early_stopped:
            break
    for p in frozen:
        p.requires_grad = True
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(history)
    return {"history": history, "best_epoch": best_epoch}


def train_final(
    model: nn.Module, train_loader: Iterable, y_train: Any,
    optimizer: torch.optim.Optimizer, *, epochs: int, seed: int = 0,
    device: str | torch.device = "cpu",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    **fit_options: Any,
) -> dict[str, Any]:
    """Entrena durante épocas fijas sin validación ni selección.

    ``fit_options`` (pérdida, máscara, calentamiento, AMP...) se pasan a ``fit``.
    Para reproducir la inicialización, llame también a ``seed_everything``
    *antes* de crear modelo, optimizador y loader.
    """
    seed_everything(seed)
    return fit(model, train_loader, y_train, optimizer, epochs=epochs, device=device,
               progress_callback=progress_callback, **fit_options)


def predict_logits(
    model: nn.Module, loader: Iterable, *, device: str | torch.device = "cpu",
    tta: bool = False, amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (etiquetas, logits, máscara) en el orden del loader.

    Con ``tta`` el logit es el promedio del original y su reflejo horizontal.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()
    dtype, _ = _amp_setup(device, amp)
    labels, logits_all, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y, m = _batch(batch, device)
            logits = _forward(model, x, device, dtype, tta=tta)
            if logits.shape != y.shape or not torch.isfinite(logits).all():
                raise ValueError("El modelo debe devolver logits finitos de forma (N, 4)")
            labels.append(y.cpu().numpy())
            logits_all.append(logits.cpu().numpy())
            masks.append(m.cpu().numpy())
    if not labels:
        raise ValueError("El loader está vacío")
    return (np.concatenate(labels).astype(np.int64), np.concatenate(logits_all),
            np.concatenate(masks).astype(np.int64))


def predict(model: nn.Module, loader: Iterable, *, device: str | torch.device = "cpu",
            tta: bool = False, amp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (etiquetas, probabilidades sigmoid) en el orden del loader."""
    labels, logits, _ = predict_logits(model, loader, device=device, tta=tta, amp=amp)
    return labels, 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def calibrate_thresholds(y_cal: Any, probabilities: Any, *, min_sensitivity: float = 0.9,
                         mask: Any = None) -> list[dict[str, Any]]:
    """Maximiza especificidad sujeta a sensibilidad mínima en calibración.

    Regla positiva: probabilidad >= umbral. Empates de especificidad se
    resuelven con mayor sensibilidad y después mayor umbral. Con ``mask`` solo
    cuentan las etiquetas confirmadas. Sin positivos o negativos devuelve
    umbral NaN y razón explícita, no un valor arbitrario.
    """
    y = _labels(y_cal)
    p = _probabilities(y, probabilities)
    m = _mask(mask, y)
    if not 0 < min_sensitivity <= 1:
        raise ValueError("min_sensitivity debe estar en (0, 1]")
    results = []
    for j in range(N_LABELS):
        known = m[:, j] == 1
        truth, score = y[known, j], p[known, j]
        positives, negatives = truth == 1, truth == 0
        if not positives.any() or not negatives.any():
            reason = "sin positivos" if not positives.any() else "sin negativos"
            results.append({"threshold": float("nan"), "sensitivity": float("nan"),
                            "specificity": float("nan"), "status": reason})
            continue
        pos_sorted, neg_sorted = np.sort(score[positives]), np.sort(score[negatives])
        candidates = np.unique(score)
        sensitivity = 1 - np.searchsorted(pos_sorted, candidates, "left") / len(pos_sorted)
        specificity = np.searchsorted(neg_sorted, candidates, "left") / len(neg_sorted)
        valid = np.flatnonzero(sensitivity >= min_sensitivity)
        if not len(valid):
            results.append({"threshold": float("nan"), "sensitivity": float("nan"),
                            "specificity": float("nan"), "status": "restricción imposible"})
            continue
        best = max(valid, key=lambda i: (specificity[i], sensitivity[i], candidates[i]))
        results.append({"threshold": float(candidates[best]),
                        "sensitivity": float(sensitivity[best]),
                        "specificity": float(specificity[best]), "status": "ok"})
    return results


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def evaluate_multilabel(y_true: Any, probabilities: Any, thresholds: Sequence[float],
                        mask: Any = None) -> list[dict[str, Any]]:
    """Métricas por clase sobre etiquetas confirmadas; NaN = indefinida o sin umbral.

    La matriz es [[TN, FP], [FN, TP]]. AUROC y AUPRC se dejan indefinidas
    cuando falta alguna clase; Brier y ECE permanecen definidos. No ajusta
    umbrales. ``n_unknown`` cuenta las muestras excluidas por la máscara.
    """
    y = _labels(y_true)
    p = _probabilities(y, probabilities)
    m = _mask(mask, y)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (N_LABELS,) or np.any(np.isinf(thresholds)) or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Se requieren cuatro umbrales en [0, 1] o NaN")
    result = []
    for j, threshold in enumerate(thresholds):
        known = m[:, j] == 1
        truth, prob = y[known, j], p[known, j]
        positives, negatives = int(truth.sum()), int(len(truth) - truth.sum())
        both = positives > 0 and negatives > 0
        row: dict[str, Any] = {
            "n_positive": positives, "n_negative": negatives,
            "n_unknown": int((~known).sum()),
            "class_status": "ok" if both else ("sin positivos" if not positives else "sin negativos"),
            "threshold_status": "ok" if np.isfinite(threshold) else "sin umbral",
            "auroc": float(roc_auc_score(truth, prob)) if both else float("nan"),
            "auprc": float(average_precision_score(truth, prob)) if both else float("nan"),
            "brier": float(np.mean((prob - truth) ** 2)) if len(truth) else float("nan"),
            "ece": expected_calibration_error(truth, prob) if len(truth) else float("nan"),
            "confusion_matrix": None,
            "sensitivity": float("nan"), "specificity": float("nan"),
            "ppv": float("nan"), "npv": float("nan"),
        }
        if np.isfinite(threshold):
            pred = prob >= threshold
            tp = int(np.sum(pred & (truth == 1)))
            fp = int(np.sum(pred & (truth == 0)))
            fn, tn = positives - tp, negatives - fp
            row.update(confusion_matrix=[[tn, fp], [fn, tp]],
                       sensitivity=_ratio(tp, positives), specificity=_ratio(tn, negatives),
                       ppv=_ratio(tp, tp + fp), npv=_ratio(tn, tn + fn))
        result.append(row)
    return result


def _clusters(ids: Any, n: int) -> list[list[int]]:
    ids = np.asarray(ids)
    if ids.ndim != 1 or len(ids) != n:
        raise ValueError("patient_ids debe tener un identificador por muestra")
    clusters: dict[Any, list[int]] = {}
    for index, key in enumerate(ids.tolist()):
        if key is None or (isinstance(key, (float, np.floating)) and not np.isfinite(key)):
            raise ValueError("patient_ids no puede contener valores ausentes")
        try:
            clusters.setdefault(key, []).append(index)
        except TypeError as exc:
            raise ValueError("Cada patient_id debe ser hashable") from exc
    return list(clusters.values())


def bootstrap_patient_metrics(
    y_true: Any, probabilities: Any, thresholds: Sequence[float], patient_ids: Sequence[Any],
    *, n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 0, mask: Any = None,
) -> list[dict[str, Any]]:
    """IC percentil por remuestreo de pacientes completos con reemplazo.

    Los umbrales se mantienen fijos. Cada intervalo incluye ``n_valid`` y
    ``n_total``; réplicas con métricas indefinidas se excluyen y se explicitan.
    Devuelve NaN en los extremos si no hay ninguna réplica válida.
    """
    y = _labels(y_true)
    p = _probabilities(y, probabilities)
    m = _mask(mask, y)
    point = evaluate_multilabel(y, p, thresholds, m)
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap debe ser positivo y confidence debe estar en (0, 1)")
    members = _clusters(patient_ids, len(y))
    rng = np.random.default_rng(seed)
    names = ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "brier", "ece")
    samples = [[[] for _ in names] for _ in range(N_LABELS)]
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[i] for i in selected])
        rows = evaluate_multilabel(y[indices], p[indices], thresholds, m[indices])
        for j, row in enumerate(rows):
            for k, name in enumerate(names):
                if np.isfinite(row[name]):
                    samples[j][k].append(row[name])
    alpha = (1 - confidence) / 2
    for j, row in enumerate(point):
        row["intervals"] = {}
        for k, name in enumerate(names):
            values = samples[j][k]
            bounds = np.quantile(values, [alpha, 1 - alpha]) if values else [float("nan")] * 2
            row["intervals"][name] = {
                "low": float(bounds[0]), "high": float(bounds[1]),
                "n_valid": len(values), "n_total": n_bootstrap,
            }
    return point


def search_grouped_cv(
    y_train: Any, groups: Sequence[Any], folds: Iterable[tuple[Sequence[int], Sequence[int]]],
    *, suggest: Callable[[Any], dict[str, Any]],
    make_loaders: Callable[[np.ndarray, np.ndarray, dict[str, Any], int], tuple[Iterable, Iterable]],
    make_model: Callable[[dict[str, Any]], nn.Module],
    make_optimizer: Callable[[nn.Module, dict[str, Any]], torch.optim.Optimizer],
    n_trials: int, max_epochs: int, patience: int, seed: int = 0,
    device: str | torch.device = "cpu", initial_trials: Sequence[dict[str, Any]] = (),
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    mask: Any = None, objective: str = "auroc", monitor: str = "val_loss",
    fit_options: Callable[[dict[str, Any], nn.Module], dict[str, Any]] | None = None,
    storage: str | None = None, study_name: str | None = None, prune: bool = False,
    amp: bool = False,
) -> dict[str, Any]:
    """Busca con Optuna solo en entrenamiento usando folds agrupados explícitos.

    Índices son posiciones en ``y_train``/``groups``. Cada callback se invoca
    por trial y fold, tras fijar una semilla por fold: ``make_loaders`` debe
    crear datasets/loaders nuevos, ajustar transformaciones solo en train y
    devolver validación sin shuffle ni drop_last; ``make_model`` y
    ``make_optimizer`` crean objetos nuevos. ``suggest(trial)`` devuelve el
    diccionario completo que reciben los callbacks y ``fit_options(params,
    model)`` las opciones extra de ``fit`` (pérdida, calentamiento...).

    ``objective`` es ``auroc`` o ``auprc`` macro (etiquetas confirmadas, sin
    clases ausentes). Cada fold se evalúa con y sin TTA de espejo; ``tta`` indica
    si el promedio del mejor trial mejora con TTA. Con ``storage`` (URL SQLite de
    Optuna) la búsqueda se reanuda sin repetir pruebas completadas. Las épocas
    finales son la mediana redondeada (hacia arriba en empates) de las mejores
    épocas del mejor trial. Nunca acepta calibración ni test.
    """
    import optuna

    if objective not in ("auroc", "auprc"):
        raise ValueError("objective debe ser 'auroc' o 'auprc'")
    y = _labels(y_train)
    mask_array = _mask(mask, y)
    group_array = np.asarray(groups)
    if group_array.ndim != 1 or len(group_array) != len(y):
        raise ValueError("groups debe tener un identificador por muestra")
    if any(g is None or (isinstance(g, (float, np.floating)) and not np.isfinite(g))
           for g in group_array.tolist()):
        raise ValueError("groups no puede contener valores ausentes")
    if not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials debe ser positivo")
    checked = []
    for train_idx, val_idx in folds:
        a, b = np.asarray(train_idx), np.asarray(val_idx)
        for indices in (a, b):
            if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu":
                raise ValueError("Cada conjunto de índices debe ser un vector entero no vacío")
            if np.any(indices < 0) or np.any(indices >= len(y)) or len(np.unique(indices)) != len(indices):
                raise ValueError("Índices duplicados o fuera de rango")
        if np.intersect1d(a, b).size or set(group_array[a].tolist()) & set(group_array[b].tolist()):
            raise ValueError("Hay solapamiento de muestras o pacientes entre train y validación")
        pos_weight_from_labels(y[a], mask_array[a])
        checked.append((a.copy(), b.copy()))
    if not checked:
        raise ValueError("Se requiere al menos un fold")

    def run_objective(trial: Any) -> float:
        params = suggest(trial)
        if not isinstance(params, dict):
            raise TypeError("suggest debe devolver un diccionario")
        reports = []
        for fold_index, (a, b) in enumerate(checked):
            fold_seed = seed + fold_index
            seed_everything(fold_seed)
            train_loader, val_loader = make_loaders(a.copy(), b.copy(), params, fold_seed)
            model = make_model(params)
            optimizer = make_optimizer(model, params)

            def report_epoch(
                event, *, trial_number=trial.number + 1, current_fold=fold_index + 1
            ):
                if progress_callback is not None:
                    progress_callback({
                        **event, "phase": "cv", "trial": trial_number,
                        "total_trials": n_trials, "fold": current_fold,
                        "total_folds": len(checked),
                    })

            options = fit_options(params, model) if fit_options is not None else {}
            fitted = fit(model, train_loader, y[a], optimizer, epochs=max_epochs,
                         val_loader=val_loader, patience=patience, device=device,
                         progress_callback=report_epoch if progress_callback is not None else None,
                         mask_train=mask_array[a], monitor=monitor, amp=amp, **options)
            actual, logits, val_mask = predict_logits(model, val_loader, device=device, amp=amp)
            if not np.array_equal(actual, y[b]):
                raise ValueError("El loader de validación no coincide en orden y etiquetas con el fold")
            if not np.array_equal(val_mask, mask_array[b]):
                raise ValueError("La máscara del loader de validación no coincide con el fold")
            probabilities = 1 / (1 + np.exp(-logits.astype(np.float64)))
            metrics = evaluate_multilabel(actual, probabilities, [0.5] * N_LABELS, val_mask)
            aucs = [row["auroc"] for row in metrics]
            valid = [v for v in aucs if np.isfinite(v)]
            if not valid:
                raise ValueError(f"Fold {fold_index}: ninguna AUROC definida")
            auprc, auprc_classes = macro_metric(actual, probabilities, val_mask, "auprc")
            _, tta_logits, _ = predict_logits(model, val_loader, device=device, tta=True, amp=amp)
            tta_probabilities = 1 / (1 + np.exp(-tta_logits.astype(np.float64)))
            tta_auprc, _ = macro_metric(actual, tta_probabilities, val_mask, "auprc")
            tta_auroc, _ = macro_metric(actual, tta_probabilities, val_mask, "auroc")
            reports.append({"fold": fold_index, "best_epoch": fitted["best_epoch"],
                            "auroc_per_class": [v if np.isfinite(v) else None for v in aucs],
                            "auprc_per_class": [v if np.isfinite(v) else None for v in auprc_classes],
                            "class_status": [row["class_status"] for row in metrics],
                            "macro_auroc": float(np.mean(valid)),
                            "macro_auprc": auprc if np.isfinite(auprc) else None,
                            "macro_auroc_tta": tta_auroc if np.isfinite(tta_auroc) else None,
                            "macro_auprc_tta": tta_auprc if np.isfinite(tta_auprc) else None,
                            "n_epochs_run": len(fitted["history"])})
            if progress_callback is not None:
                progress_callback({
                    "event": "fold_complete", "phase": "cv",
                    "trial": trial.number + 1, "total_trials": n_trials,
                    "fold": fold_index + 1, "total_folds": len(checked),
                    "best_epoch": fitted["best_epoch"],
                    "macro_auroc": reports[-1]["macro_auroc"],
                    "macro_auprc": reports[-1]["macro_auprc"],
                })
            del model, optimizer, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            key = "macro_auprc" if objective == "auprc" else "macro_auroc"
            running = [r[key] for r in reports if r[key] is not None]
            if prune and running and fold_index + 1 < len(checked):
                trial.report(float(np.mean(running)), fold_index)
                if trial.should_prune():
                    trial.set_user_attr("pruned_after_fold", fold_index + 1)
                    raise optuna.TrialPruned()
        trial.set_user_attr("folds", reports)
        trial.set_user_attr("resolved_params", params)
        key = "macro_auprc" if objective == "auprc" else "macro_auroc"
        scores = [r[key] for r in reports if r[key] is not None]
        if not scores:
            raise ValueError("Ningún fold produjo la métrica objetivo")
        score = float(np.mean(scores))
        if progress_callback is not None:
            progress_callback({
                "event": "trial_complete", "phase": "cv",
                "trial": trial.number + 1, "total_trials": n_trials,
                "score": score, "objective": objective, "params": params,
            })
        return score

    pruner = optuna.pruners.MedianPruner(n_startup_trials=2) if prune else optuna.pruners.NopPruner()
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed), pruner=pruner,
        storage=storage, study_name=study_name, load_if_exists=storage is not None,
    )
    finished = [t for t in study.trials if t.state.is_finished()]
    if not study.trials:
        for params in initial_trials[:n_trials]:
            study.enqueue_trial(params)
    remaining = max(0, n_trials - len(finished))
    if remaining:
        study.optimize(run_objective, n_trials=remaining)
    best = study.best_trial
    reports = best.user_attrs["folds"]
    epochs = int(np.floor(np.median([r["best_epoch"] for r in reports]) + 0.5))
    plain = [r["macro_auprc"] for r in reports if r["macro_auprc"] is not None]
    flipped = [r["macro_auprc_tta"] for r in reports if r["macro_auprc_tta"] is not None]
    use_tta = bool(plain and flipped and np.mean(flipped) > np.mean(plain))
    return {"params": best.user_attrs["resolved_params"], "epochs": max(1, epochs),
            "score": best.value, "folds": reports, "tta": use_tta, "study": study}
