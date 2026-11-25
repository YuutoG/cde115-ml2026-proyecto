"""Evaluación opcional con etiquetas externas explícitamente conocidas."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import LABELS, _image_filename, _sha256_file


def load_external(csv_path, image_dir):
    """Carga file_name, age, group_id y etiquetas 0/1/vacías, sin imputar ausencias."""
    data = pd.read_csv(csv_path)
    required = {"file_name", "age", "group_id", *LABELS}
    if not required.issubset(data.columns):
        raise ValueError(f"Faltan columnas externas: {sorted(required - set(data.columns))}")
    if data["group_id"].isna().any():
        raise ValueError("Faltan identificadores de grupo externo")
    data["age"] = pd.to_numeric(data["age"], errors="raise")
    if data["age"].isna().any() or (~data["age"].between(0, 120)).any():
        raise ValueError("Edades externas ausentes o inválidas")
    data = data.loc[data.age >= 15].copy()
    for label in LABELS:
        data[label] = pd.to_numeric(data[label], errors="raise")
        if not data[label].dropna().isin([0, 1]).all():
            raise ValueError(f"Etiqueta externa inválida: {label}")
    directory = Path(image_dir)
    data["image_path"] = [str(directory / _image_filename(str(name))) for name in data.file_name]
    missing = sum(not Path(path).is_file() for path in data.image_path)
    if missing:
        raise ValueError(f"Faltan {missing} imágenes externas")
    if data.empty:
        raise ValueError("No hay pacientes externos elegibles")
    return data.reset_index(drop=True)


def ensure_no_exact_overlap(data, internal_hashes):
    """Rechaza duplicados byte a byte; no detecta variantes recodificadas."""
    known = set(internal_hashes)
    overlaps = sum(_sha256_file(Path(path)) in known for path in data.image_path)
    if overlaps:
        raise ValueError(f"{overlaps} imágenes externas son duplicados exactos del dataset interno")


def evaluate_external(data, probabilities, thresholds, *, n_bootstrap=200, seed=42):
    """Evalúa solo valores anotados, sin tratar valores desconocidos como negativos."""
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(data), len(LABELS)) or not np.isfinite(probabilities).all():
        raise ValueError("Probabilidades externas inválidas")
    if np.any((probabilities < 0) | (probabilities > 1)) or len(thresholds) != len(LABELS):
        raise ValueError("Probabilidades o umbrales externos inválidos")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap debe ser positivo")
    rng = np.random.default_rng(seed)
    rows = []
    for index, label in enumerate(LABELS):
        known = data[label].notna().to_numpy()
        y = data.loc[known, label].to_numpy(dtype=int)
        p = probabilities[known, index]
        row = {"label": label, "known": len(y), "unknown": int((~known).sum()),
               "patients": int(data.loc[known, "group_id"].nunique())}
        if len(y) == 0 or np.unique(y).size < 2:
            row["status"] = "sin ambas clases conocidas"
        else:
            row.update(status="ok", auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p))
            threshold = thresholds[index]
            if threshold is not None:
                if not 0 <= threshold <= 1:
                    raise ValueError(f"Umbral inválido para {label}")
                pred = p >= threshold
                row["sensitivity"] = float(pred[y == 1].mean())
                row["specificity"] = float((~pred[y == 0]).mean())
            groups = data.loc[known, "group_id"].to_numpy()
            clusters = [np.flatnonzero(groups == group) for group in pd.unique(groups)]
            samples = {"auroc": [], "auprc": [], "sensitivity": [], "specificity": []}
            for _ in range(n_bootstrap if len(clusters) > 1 else 0):
                chosen = rng.integers(len(clusters), size=len(clusters))
                indices = np.concatenate([clusters[i] for i in chosen])
                actual, predicted = y[indices], p[indices]
                if np.unique(actual).size != 2:
                    continue
                samples["auroc"].append(float(roc_auc_score(actual, predicted)))
                samples["auprc"].append(float(average_precision_score(actual, predicted)))
                if threshold is not None:
                    decisions = predicted >= threshold
                    samples["sensitivity"].append(float(decisions[actual == 1].mean()))
                    samples["specificity"].append(float((~decisions[actual == 0]).mean()))
            row["intervals"] = {
                name: {"low": float(np.quantile(values, .025)) if values else None,
                       "high": float(np.quantile(values, .975)) if values else None,
                       "n_valid": len(values), "n_total": n_bootstrap}
                for name, values in samples.items()
            }
        rows.append(row)
    return pd.DataFrame(rows)