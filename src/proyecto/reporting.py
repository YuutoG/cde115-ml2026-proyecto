"""Puntos de operación, tablas de métricas y comparaciones listas para ``stats/``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .calibration import reliability_table
from .data import LABELS
from .training import (
    _clusters,
    bootstrap_patient_metrics,
    calibrate_thresholds,
    evaluate_multilabel,
    macro_metric,
)

SENSITIVITY_POINTS = {"sens95": 0.95, "sens90": 0.90, "sens85": 0.85, "sens80": 0.80}
POINTS = (*SENSITIVITY_POINTS, "youden")
PRINCIPAL = "sens85"


def _youden(y, p, mask) -> list[dict]:
    rows = []
    for j in range(len(LABELS)):
        known = mask[:, j] == 1
        truth, score = y[known, j], p[known, j]
        if truth.sum() == 0 or truth.sum() == len(truth):
            rows.append({"threshold": float("nan"), "sensitivity": float("nan"),
                         "specificity": float("nan"), "status": "falta una clase"})
            continue
        pos, neg = np.sort(score[truth == 1]), np.sort(score[truth == 0])
        candidates = np.unique(score)
        sens = 1 - np.searchsorted(pos, candidates, "left") / len(pos)
        spec = np.searchsorted(neg, candidates, "left") / len(neg)
        best = int(np.argmax(sens + spec - 1))
        rows.append({"threshold": float(candidates[best]), "sensitivity": float(sens[best]),
                     "specificity": float(spec[best]), "status": "ok"})
    return rows


def fit_operating_points(y, p, mask=None) -> dict[str, list[dict]]:
    """Umbrales por punto de operación, ajustados solo con datos de calibración."""
    y = np.asarray(y)
    m = np.ones_like(y) if mask is None else np.asarray(mask)
    points = {name: calibrate_thresholds(y, p, min_sensitivity=value, mask=m)
              for name, value in SENSITIVITY_POINTS.items()}
    points["youden"] = _youden(y, np.asarray(p), m)
    return points


def thresholds_of(points: dict[str, list[dict]], point: str = PRINCIPAL) -> list[float | None]:
    return [row["threshold"] if np.isfinite(row["threshold"]) else None for row in points[point]]


def all_thresholds(points: dict[str, list[dict]]) -> dict[str, list[float | None]]:
    return {point: thresholds_of(points, point) for point in points}


def operating_points_frame(family: str, model: str, points: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for point, results in points.items():
        for label, row in zip(LABELS, results, strict=True):
            rows.append({
                "family": family, "model": model, "label": label, "point": point,
                "principal": point == PRINCIPAL,
                "threshold": row["threshold"] if np.isfinite(row["threshold"]) else None,
                "cal_sensitivity": row["sensitivity"], "cal_specificity": row["specificity"],
                "status": row["status"],
            })
    return pd.DataFrame(rows)


def calibration_frames(family, model, y, variants: dict[str, np.ndarray], mask=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Brier/ECE por variante (raw, temperature) y puntos de las curvas de fiabilidad."""
    y = np.asarray(y)
    m = np.ones_like(y) if mask is None else np.asarray(mask)
    metrics, curves = [], []
    for variant, p in variants.items():
        rows = evaluate_multilabel(y, p, [float("nan")] * len(LABELS), m)
        for label, row in zip(LABELS, rows, strict=True):
            metrics.append({"family": family, "model": model, "label": label, "variant": variant,
                            "brier": row["brier"], "ece": row["ece"],
                            "n": row["n_positive"] + row["n_negative"]})
        table = reliability_table(y, p, m)
        table.insert(0, "variant", variant)
        table.insert(0, "model", model)
        table.insert(0, "family", family)
        curves.append(table)
    return pd.DataFrame(metrics), pd.concat(curves, ignore_index=True)


def split_metrics_frame(family, model, y, p, mask=None) -> pd.DataFrame:
    y = np.asarray(y)
    m = np.ones_like(y) if mask is None else np.asarray(mask)
    rows = evaluate_multilabel(y, p, [float("nan")] * len(LABELS), m)
    return pd.DataFrame([{
        "family": family, "model": model, "label": label, "auroc": r["auroc"], "auprc": r["auprc"],
        "prevalence": r["n_positive"] / max(1, r["n_positive"] + r["n_negative"]),
        "brier": r["brier"], "ece": r["ece"],
    } for label, r in zip(LABELS, rows, strict=True)])


def test_frames(family, model, y, p, groups, thresholds_by_point, *, mask=None,
                n_bootstrap=500, seed=42):
    """Métricas con IC, matrices de confusión y PR-AUC en test con umbrales congelados.

    ``thresholds_by_point`` mapea cada punto operativo a cuatro umbrales (None = sin umbral).
    """
    y = np.asarray(y)
    m = np.ones_like(y) if mask is None else np.asarray(mask)
    metric_rows, confusion_rows, pr_rows = [], [], []
    for point, values in thresholds_by_point.items():
        thresholds = [t if t is not None else float("nan") for t in values]
        results = bootstrap_patient_metrics(y, p, thresholds, groups, n_bootstrap=n_bootstrap,
                                            seed=seed, mask=m)
        for label, row in zip(LABELS, results, strict=True):
            for metric in ("sensitivity", "specificity", "ppv", "npv"):
                interval = row["intervals"][metric]
                metric_rows.append({
                    "family": family, "model": model, "label": label, "point": point,
                    "metric": metric, "value": row[metric], "low": interval["low"],
                    "high": interval["high"], "n_unknown": row["n_unknown"],
                })
            if row["confusion_matrix"] is not None:
                (tn, fp), (fn, tp) = row["confusion_matrix"]
                confusion_rows.append({"family": family, "model": model, "label": label,
                                       "point": point, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
            if point == PRINCIPAL:
                for metric in ("auroc", "auprc", "brier", "ece"):
                    interval = row["intervals"][metric]
                    metric_rows.append({
                        "family": family, "model": model, "label": label, "point": "none",
                        "metric": metric, "value": row[metric], "low": interval["low"],
                        "high": interval["high"], "n_unknown": row["n_unknown"],
                    })
                interval = row["intervals"]["auprc"]
                pr_rows.append({
                    "family": family, "model": model, "label": label, "auprc": row["auprc"],
                    "prevalence": row["n_positive"] / max(1, row["n_positive"] + row["n_negative"]),
                    "auprc_low": interval["low"], "auprc_high": interval["high"],
                })
    return pd.DataFrame(metric_rows), pd.DataFrame(confusion_rows), pd.DataFrame(pr_rows)


def paired_auprc_difference(y, p_a, p_b, groups, *, mask=None, n_bootstrap=500, seed=42) -> pd.DataFrame:
    """PR-AUC(A) − PR-AUC(B) por etiqueta y macro, remuestreando pacientes."""
    y = np.asarray(y)
    m = np.ones_like(y) if mask is None else np.asarray(mask)
    members = _clusters(groups, len(y))
    rng = np.random.default_rng(seed)

    def per_label(index):
        values = []
        for j in range(len(LABELS)):
            known = index[m[index, j] == 1]
            truth = y[known, j]
            if truth.sum() == 0 or truth.sum() == len(truth):
                values.append(np.nan)
            else:
                values.append(average_precision_score(truth, p_a[known, j])
                              - average_precision_score(truth, p_b[known, j]))
        return values

    full = per_label(np.arange(len(y)))
    samples = []
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(members), size=len(members))
        samples.append(per_label(np.concatenate([members[i] for i in chosen])))
    samples = np.asarray(samples)
    rows = []
    for j, label in enumerate([*LABELS, "macro"]):
        if label == "macro":
            complete = np.isfinite(samples).all(axis=1)
            column = samples[complete].mean(axis=1)
            value = float(np.mean(full)) if np.isfinite(full).all() else np.nan
        else:
            column = samples[:, j]
            value = full[j]
        column = column[np.isfinite(column)]
        low, high = (np.quantile(column, [0.025, 0.975]) if len(column) else (np.nan, np.nan))
        rows.append({
            "label": label, "difference": value, "low": float(low), "high": float(high),
            "n_valid": len(column), "n_total": n_bootstrap,
            "interpretation": (("IC excluye 0" if low > 0 or high < 0 else "IC incluye 0")
                               if len(column) >= max(20, int(.9 * n_bootstrap))
                               else "réplicas válidas insuficientes"),
        })
    return pd.DataFrame(rows)


def masking_frame(family, model, y, p, mask) -> pd.DataFrame:
    """Compara métricas excluyendo etiquetas ausentes (masked) y tratándolas como negativas."""
    y = np.asarray(y)
    rows = []
    for name, m in (("masked", np.asarray(mask)), ("unmasked", np.ones_like(y))):
        for j, label in enumerate(LABELS):
            part = evaluate_multilabel(y, p, [float("nan")] * len(LABELS), m)[j]
            rows.append({"family": family, "model": model, "label": label, "masking": name,
                         "auprc": part["auprc"], "auroc": part["auroc"],
                         "n": part["n_positive"] + part["n_negative"]})
    return pd.DataFrame(rows)


def macro_specificity(y, p, points, point, mask=None) -> float:
    """Especificidad macro en las etiquetas con umbral disponible."""
    y = np.asarray(y)
    m = np.ones_like(y) if mask is None else np.asarray(mask)
    thresholds = [t if t is not None else float("nan") for t in thresholds_of(points, point)]
    rows = evaluate_multilabel(y, p, thresholds, m)
    values = [r["specificity"] for r in rows if np.isfinite(r["specificity"])]
    return float(np.mean(values)) if values else float("nan")


def macro_brier(y, p, mask=None) -> float:
    y = np.asarray(y)
    m = np.ones_like(y) if mask is None else np.asarray(mask)
    rows = evaluate_multilabel(y, p, [float("nan")] * len(LABELS), m)
    return float(np.mean([r["brier"] for r in rows]))


def macro_auprc(y, p, mask=None) -> float:
    return macro_metric(np.asarray(y), np.asarray(p), mask, "auprc")[0]
