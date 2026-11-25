"""Calibración de probabilidades: escalado por temperatura, ECE, Brier y curvas."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .data import LABELS

TEMPERATURE_RANGE = (0.25, 4.0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def logit(p, eps: float = 1e-7):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE con bins de igual ancho: media ponderada de |frecuencia - confianza|."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if not len(y):
        return float("nan")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        selected = bins == b
        if selected.any():
            total += selected.mean() * abs(y[selected].mean() - p[selected].mean())
    return float(total)


def fit_temperature(logits, y, mask=None, bounds=TEMPERATURE_RANGE) -> dict:
    """Ajusta T > 0 minimizando la NLL de sigmoid(logit / T) en etiquetas confirmadas.

    Devuelve ``at_bound=True`` si el óptimo queda pegado a un extremo del rango,
    señal de que la temperatura no es un óptimo interior fiable. Con una sola
    clase presente devuelve T=1 y ``status`` explicativo.
    """
    z, truth = np.asarray(logits, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if mask is not None:
        known = np.asarray(mask) == 1
        z, truth = z[known], truth[known]
    if not len(truth) or truth.min() == truth.max():
        return {"temperature": 1.0, "status": "sin ambas clases", "at_bound": False,
                "nll_before": float("nan"), "nll_after": float("nan")}

    def nll(temperature: float) -> float:
        p = np.clip(sigmoid(z / temperature), 1e-7, 1 - 1e-7)
        return float(-np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p)))

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    temperature = float(result.x)
    at_bound = bool(min(abs(temperature - bounds[0]), abs(temperature - bounds[1])) < 1e-2)
    return {"temperature": temperature, "status": "ok", "at_bound": at_bound,
            "nll_before": nll(1.0), "nll_after": nll(temperature)}


def fit_temperatures(logits, y, mask=None) -> list[dict]:
    """Una temperatura por etiqueta."""
    y = np.asarray(y)
    m = np.ones_like(y) if mask is None else np.asarray(mask)
    return [fit_temperature(logits[:, j], y[:, j], m[:, j]) for j in range(len(LABELS))]


def apply_temperatures(logits, temperatures) -> np.ndarray:
    """Probabilidades calibradas: sigmoid(logit_j / T_j)."""
    return sigmoid(np.asarray(logits, dtype=np.float64) / np.asarray(temperatures, dtype=np.float64))


def reliability_table(y, p, mask=None, n_bins: int = 10) -> pd.DataFrame:
    """Datos de la curva de fiabilidad por etiqueta (una fila por bin no vacío)."""
    y, p = np.asarray(y), np.asarray(p)
    m = np.ones_like(y) if mask is None else np.asarray(mask)
    rows = []
    for j, label in enumerate(LABELS):
        known = m[:, j] == 1
        truth, prob = y[known, j], p[known, j]
        bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
        for b in range(n_bins):
            selected = bins == b
            if selected.any():
                rows.append({
                    "label": label, "bin": b, "bin_low": b / n_bins, "bin_high": (b + 1) / n_bins,
                    "n": int(selected.sum()), "mean_predicted": float(prob[selected].mean()),
                    "observed_frequency": float(truth[selected].mean()),
                })
    return pd.DataFrame(rows, columns=["label", "bin", "bin_low", "bin_high", "n",
                                       "mean_predicted", "observed_frequency"])


def plot_reliability(tables: dict[str, pd.DataFrame], title: str = "Curvas de fiabilidad"):
    """Dibuja una figura por etiqueta con una curva por modelo/variante."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(LABELS), figsize=(5 * len(LABELS), 4.5), squeeze=False)
    for ax, label in zip(axes[0], LABELS, strict=True):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="calibración perfecta")
        for name, table in tables.items():
            part = table[table.label == label]
            ax.plot(part.mean_predicted, part.observed_frequency, marker="o", label=name)
        ax.set_title(label)
        ax.set_xlabel("probabilidad media predicha")
        ax.set_ylabel("frecuencia observada")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
