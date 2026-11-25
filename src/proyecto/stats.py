"""Almacén de estadísticas en ``stats/`` con diccionario de datos y reanudación.

Cada archivo escrito se valida contra ``SCHEMAS`` (descripción y significado de
cada columna) y se documenta en ``stats/manifest.json`` y ``stats/README.md``,
de modo que quien consuma los CSV en el futuro pueda interpretarlos.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

_CI = {
    "low": "Límite inferior del IC 95% (bootstrap por paciente).",
    "high": "Límite superior del IC 95% (bootstrap por paciente).",
}

SCHEMAS: dict[str, dict] = {
    "nested_oof": {
        "description": "Predicciones OOF de ensembles sin calibrar en CV anidada. "
                       "No son predicciones del modelo final ni proceden de test.",
        "columns": {
            "seed": "Semilla de repetición.", "fold": "Fold exterior (base 1).",
            "family": "Familia.", "optimizer": "Optimizador del brazo.",
            "augmentation": "Perfil de aumento del brazo.", "source_row": "Fila de origen.",
            "group_id": "Grupo de paciente.", "label": "Etiqueta.",
            "truth": "Etiqueta registrada.", "known": "Máscara de etiqueta.",
            "probability": "Media de probabilidades sin calibrar de los miembros.",
        },
    },
    "nested_selection": {
        "description": "Selecciones de cada búsqueda interna, sin observar el fold exterior.",
        "columns": {
            "seed": "Semilla de repetición.", "fold": "Fold exterior (base 1).",
            "family": "Familia.", "model": "Arquitectura.", "optimizer": "Optimizador.",
            "augmentation": "Perfil de aumento.", "inner_ap": "AP macro de la búsqueda interna.",
            "epochs": "Épocas de entrenamiento exterior.", "tta": "TTA elegida dentro del fold interno.",
            "params": "Parámetros elegidos (JSON).",
        },
    },
    "nested_metrics": {
        "description": "AP y AUROC por etiqueta sobre predicciones OOF agrupadas por repetición y brazo.",
        "columns": {
            "seed": "Semilla de repetición.", "family": "Familia.", "optimizer": "Optimizador.",
            "augmentation": "Perfil de aumento.", "label": "Etiqueta.",
            "known": "Etiquetas observadas.", "positive": "Positivos observados.",
            "ap": "Average Precision.", "auroc": "AUROC.", "status": "Estado de clases.",
        },
    },
    "nested_comparisons": {
        "description": "Diferencia de AP entre brazos OOF que cambian un factor; IC exploratorios, "
                       "sin ajuste por comparaciones múltiples ni incertidumbre de entrenamiento.",
        "columns": {
            "seed": "Semilla de repetición.", "family": "Familia.", "factor": "Factor comparado.",
            "arm": "Brazo evaluado (optimizador/aumento).", "reference": "Brazo de referencia.",
            "label": "Etiqueta o macro.", "difference": "AP del brazo menos la referencia.",
            **_CI, "n_valid": "Réplicas válidas.", "n_total": "Réplicas solicitadas.",
            "interpretation": "Interpretación exploratoria del IC.",
        },
    },
    "splits": {
        "description": "Partición congelada de las imágenes. El test se usa una sola vez, "
                       "al final, con modelos y umbrales ya fijados. Se reutiliza al reanudar.",
        "columns": {
            "source_row": "Fila de origen en export.csv; identifica la imagen.",
            "group_id": "Identificador de paciente (las particiones nunca separan un grupo).",
            "split": "development (CV e hiperparámetros), calibration (temperatura y umbrales) o test.",
        },
    },
    "label_audit": {
        "description": "Auditoría de etiquetas registradas, negativos aparentes y ausentes por partición. "
                       "No sustituye la verificación clínica.",
        "columns": {
            "split": "Partición.", "label": "Patología.", "positives": "Positivos registrados.",
            "negatives": "Negativos aparentes (sin otra patología registrada).",
            "unknown": "Ceros no confirmados: la imagen tiene otra patología registrada y se "
                       "excluyen de la pérdida y de las métricas.",
        },
    },
    "cv_trials": {
        "description": "Un registro por fold de cada prueba de la búsqueda de hiperparámetros "
                       "(validación cruzada agrupada solo sobre development).",
        "columns": {
            "family": "xrv (TorchXRayVision) o imagenet (torchvision).",
            "model": "Arquitectura.", "optimizer": "Optimizador del estudio.",
            "trial": "Número de prueba Optuna (base 1; único dentro de cada estudio).",
            "fold": "Fold de validación (base 1).", "best_epoch": "Mejor época según PR-AUC macro (base 1).",
            "n_epochs_run": "Épocas realmente ejecutadas (parada temprana).",
            "macro_auprc": "PR-AUC macro del fold (etiquetas confirmadas, umbral-independiente).",
            "macro_auroc": "AUROC macro del fold.",
            "macro_auprc_tta": "PR-AUC macro con promedio de imagen y su reflejo horizontal.",
            "macro_auroc_tta": "AUROC macro con TTA de espejo.",
            "params": "Hiperparámetros de la prueba (JSON).",
        },
    },
    "cv_best": {
        "description": "Mejor configuración por modelo según la métrica de selección (PR-AUC macro en CV).",
        "columns": {
            "family": "Familia de pesos.", "model": "Arquitectura.",
            "cv_macro_auprc": "PR-AUC macro medio en CV del mejor trial.",
            "epochs": "Épocas del entrenamiento final (mediana de mejores épocas).",
            "tta": "Si el TTA de espejo mejoró la PR-AUC en CV y se usa en inferencia.",
            "params": "Hiperparámetros elegidos (JSON).",
        },
    },
    "calibration_temperatures": {
        "description": "Temperatura por etiqueta ajustada en el conjunto de calibración (p = sigmoid(logit/T)). "
                       "T>1 suaviza probabilidades sobreconfiadas.",
        "columns": {
            "family": "Familia.", "model": "Modelo o ensemble.", "label": "Patología.",
            "temperature": "T ajustada.", "at_bound": "True si T quedó en un extremo del rango [0.25, 4]; "
                                                       "no es un óptimo interior fiable.",
            "nll_before": "Log-loss con T=1.", "nll_after": "Log-loss con la T ajustada.",
            "status": "ok o motivo por el que no se ajustó.",
        },
    },
    "operating_points": {
        "description": "Curva de umbrales fijados en calibración y evaluados en el conjunto indicado. "
                       "Compensan sensibilidad y especificidad; 'principal' es el punto recomendado.",
        "columns": {
            "family": "Familia.", "model": "Modelo o ensemble.", "label": "Patología.",
            "point": "sens95/sens90/sens85/sens80 (sensibilidad mínima en calibración), "
                     "youden (máx. S+E−1, referencia).",
            "principal": "True para el punto principal (máxima especificidad con sensibilidad ≥ 85%).",
            "threshold": "Umbral sobre la probabilidad calibrada (vacío si no se pudo ajustar).",
            "cal_sensitivity": "Sensibilidad lograda en calibración.",
            "cal_specificity": "Especificidad lograda en calibración.",
            "status": "ok o razón por la que no hay umbral.",
        },
    },
    "calibration_metrics": {
        "description": "Calidad de calibración de probabilidades antes/después del escalado por temperatura.",
        "columns": {
            "family": "Familia.", "model": "Modelo o ensemble.", "label": "Patología.",
            "variant": "raw (sin escalar) o temperature.",
            "brier": "Error cuadrático medio de la probabilidad (menor es mejor).",
            "ece": "Error de calibración esperado, 10 bins (menor es mejor).",
            "n": "Etiquetas confirmadas evaluadas.",
        },
    },
    "reliability_curves": {
        "description": "Puntos de las curvas de fiabilidad por bin de probabilidad.",
        "columns": {
            "family": "Familia.", "model": "Modelo o ensemble.", "label": "Patología.",
            "variant": "raw o temperature.", "bin": "Índice de bin.",
            "bin_low": "Límite inferior del bin.", "bin_high": "Límite superior del bin.",
            "n": "Muestras en el bin.", "mean_predicted": "Probabilidad media predicha.",
            "observed_frequency": "Frecuencia observada de positivos.",
        },
    },
    "calibration_split_metrics": {
        "description": "Métricas en el conjunto de calibración con los umbrales de operating_points "
                       "(optimistas: los umbrales se eligieron aquí).",
        "columns": {
            "family": "Familia.", "model": "Modelo o ensemble.", "label": "Patología.",
            "auroc": "AUROC.", "auprc": "PR-AUC.", "prevalence": "Prevalencia de positivos (línea base de PR-AUC).",
            "brier": "Brier.", "ece": "ECE.",
        },
    },
    "family_comparison": {
        "description": "Selección exploratoria de ensembles por AP macro en calibración. "
                       "Desempates: especificidad con sensibilidad 85% y Brier en calibración.",
        "columns": {
            "family": "Familia.", "cv_macro_auprc": "Mejor PR-AUC macro medio en CV entre sus arquitecturas.",
            "ensemble_cal_macro_auprc": "PR-AUC macro del ensemble en calibración.",
            "ensemble_cal_macro_brier": "Brier macro del ensemble en calibración tras temperatura.",
            "macro_specificity_at_sens85": "Especificidad macro del ensemble al punto sens85 (calibración).",
            "selected": "True para la familia ganadora.",
            "reason": "Criterio que decidió.",
        },
    },
    "test_metrics": {
        "description": "Evaluación en test interno con umbrales congelados. Si la cohorte ya se "
                       "examinó, no constituye una confirmación independiente. IC por paciente.",
        "columns": {
            "family": "Familia.", "model": "Modelo o ensemble.", "label": "Patología.",
            "point": "Punto operativo.", "metric": "Métrica.", "value": "Estimación puntual.",
            **_CI, "n_unknown": "Etiquetas ausentes excluidas.",
        },
    },
    "test_confusion": {
        "description": "Matrices de confusión en test por punto operativo.",
        "columns": {
            "family": "Familia.", "model": "Modelo o ensemble.", "label": "Patología.",
            "point": "Punto operativo.", "TP": "Verdaderos positivos.", "FP": "Falsos positivos.",
            "FN": "Falsos negativos.", "TN": "Verdaderos negativos.",
        },
    },
    "test_pr_auc": {
        "description": "PR-AUC por etiqueta en test con prevalencia como línea base.",
        "columns": {
            "family": "Familia.", "model": "Modelo o ensemble.", "label": "Patología.",
            "auprc": "PR-AUC.", "prevalence": "Prevalencia (PR-AUC de un clasificador aleatorio).",
            "auprc_low": "IC 95% inferior.", "auprc_high": "IC 95% superior.",
        },
    },
    "family_test_difference": {
        "description": "Diferencia pareada de PR-AUC (xrv − imagenet) en test mediante bootstrap por paciente.",
        "columns": {
            "label": "Patología o 'macro'.", "difference": "PR-AUC xrv − imagenet.",
            **_CI, "n_valid": "Réplicas con métrica definida.", "n_total": "Réplicas solicitadas.",
            "interpretation": "Interpretación exploratoria del IC.",
        },
    },
    "masking_sensitivity": {
        "description": "Análisis de sensibilidad: métricas con y sin excluir etiquetas ausentes.",
        "columns": {
            "family": "Familia.", "model": "Modelo o ensemble.", "label": "Patología.",
            "masking": "masked (ceros con otra patología excluidos) o unmasked (se tratan como negativos).",
            "auprc": "PR-AUC.", "auroc": "AUROC.", "n": "Muestras usadas.",
        },
    },
    "run_info": {
        "description": "Configuración y entorno de la corrida (reproducibilidad).",
        "columns": {"key": "Parámetro.", "value": "Valor (JSON)."},
    },
}


class StatsStore:
    def __init__(self, directory: str | Path = "stats"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.directory / f"{name}.csv"

    def has(self, name: str) -> bool:
        return self.path(name).is_file()

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path(name))

    def write(self, name: str, frame: pd.DataFrame, *, config: dict | None = None) -> Path:
        schema = SCHEMAS.get(name)
        if schema is None:
            raise KeyError(f"Archivo sin diccionario de datos: {name}")
        undocumented = [c for c in frame.columns if c not in schema["columns"]]
        if undocumented:
            raise ValueError(f"Columnas sin documentar en {name}: {undocumented}")
        frame.to_csv(self.path(name), index=False)
        self._update_manifest(name, frame, config)
        return self.path(name)

    def append_rows(self, name: str, frame: pd.DataFrame) -> Path:
        """Añade filas (p. ej. progreso de CV) conservando lo ya guardado."""
        if self.has(name):
            frame = pd.concat([self.read(name), frame], ignore_index=True)
        return self.write(name, frame)

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.directory / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str),
                        encoding="utf-8")
        return path

    def read_json(self, name: str) -> dict | None:
        path = self.directory / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def storage_url(self, name: str) -> str:
        """URL SQLite de Optuna para reanudar búsquedas interrumpidas."""
        return f"sqlite:///{(self.directory / f'{name}.db').resolve()}"

    def _update_manifest(self, name: str, frame: pd.DataFrame, config: dict | None) -> None:
        path = self.directory / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        manifest[f"{name}.csv"] = {
            "description": SCHEMAS[name]["description"],
            "columns": {c: SCHEMAS[name]["columns"][c] for c in frame.columns},
            "rows": len(frame),
            "updated": datetime.now(UTC).isoformat(timespec="seconds"),
            **({"config": config} if config else {}),
        }
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        self._write_readme(manifest)

    def _write_readme(self, manifest: dict) -> None:
        lines = ["# stats/", "",
                 "Estadísticas generadas por el notebook. Documento autogenerado; ver también `manifest.json`.",
                 "Uso de investigación: no son evidencia de desempeño clínico.", ""]
        for file, entry in sorted(manifest.items()):
            lines += [f"## {file}", "", entry["description"], "", "| columna | significado |", "|---|---|"]
            lines += [f"| `{c}` | {d} |" for c, d in entry["columns"].items()]
            lines.append("")
        (self.directory / "README.md").write_text("\n".join(lines), encoding="utf-8")
