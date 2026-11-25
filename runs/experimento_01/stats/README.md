# stats/

Estadísticas generadas por el notebook. Documento autogenerado; ver también `manifest.json`.
Uso de investigación: no son evidencia de desempeño clínico.

## calibration_metrics.csv

Calidad de calibración de probabilidades antes/después del escalado por temperatura.

| columna | significado |
|---|---|
| `family` | Familia. |
| `model` | Modelo o ensemble. |
| `label` | Patología. |
| `variant` | raw (sin escalar) o temperature. |
| `brier` | Error cuadrático medio de la probabilidad (menor es mejor). |
| `ece` | Error de calibración esperado, 10 bins (menor es mejor). |
| `n` | Etiquetas confirmadas evaluadas. |

## calibration_split_metrics.csv

Métricas en el conjunto de calibración con los umbrales de operating_points (optimistas: los umbrales se eligieron aquí).

| columna | significado |
|---|---|
| `family` | Familia. |
| `model` | Modelo o ensemble. |
| `label` | Patología. |
| `auroc` | AUROC. |
| `auprc` | PR-AUC. |
| `prevalence` | Prevalencia de positivos (línea base de PR-AUC). |
| `brier` | Brier. |
| `ece` | ECE. |

## calibration_temperatures.csv

Temperatura por etiqueta ajustada en el conjunto de calibración (p = sigmoid(logit/T)). T>1 suaviza probabilidades sobreconfiadas.

| columna | significado |
|---|---|
| `family` | Familia. |
| `model` | Modelo o ensemble. |
| `label` | Patología. |
| `temperature` | T ajustada. |
| `at_bound` | True si T quedó en un extremo del rango [0.25, 4]; no es un óptimo interior fiable. |
| `nll_before` | Log-loss con T=1. |
| `nll_after` | Log-loss con la T ajustada. |
| `status` | ok o motivo por el que no se ajustó. |

## cv_best.csv

Mejor configuración por modelo según la métrica de selección (PR-AUC macro en CV).

| columna | significado |
|---|---|
| `family` | Familia de pesos. |
| `model` | Arquitectura. |
| `cv_macro_auprc` | PR-AUC macro medio en CV del mejor trial. |
| `epochs` | Épocas del entrenamiento final (mediana de mejores épocas). |
| `tta` | Si el TTA de espejo mejoró la PR-AUC en CV y se usa en inferencia. |
| `params` | Hiperparámetros elegidos (JSON). |

## cv_trials.csv

Un registro por fold de cada prueba de la búsqueda de hiperparámetros (validación cruzada agrupada solo sobre development).

| columna | significado |
|---|---|
| `family` | xrv (TorchXRayVision) o imagenet (torchvision). |
| `model` | Arquitectura. |
| `optimizer` | Optimizador del estudio. |
| `trial` | Número de prueba Optuna (base 1; único dentro de cada estudio). |
| `fold` | Fold de validación (base 1). |
| `best_epoch` | Mejor época según PR-AUC macro (base 1). |
| `n_epochs_run` | Épocas realmente ejecutadas (parada temprana). |
| `macro_auprc` | PR-AUC macro del fold (etiquetas confirmadas, umbral-independiente). |
| `macro_auroc` | AUROC macro del fold. |
| `macro_auprc_tta` | PR-AUC macro con promedio de imagen y su reflejo horizontal. |
| `macro_auroc_tta` | AUROC macro con TTA de espejo. |
| `params` | Hiperparámetros de la prueba (JSON). |

## family_comparison.csv

Selección exploratoria de ensembles por AP macro en calibración. Desempates: especificidad con sensibilidad 85% y Brier en calibración.

| columna | significado |
|---|---|
| `family` | Familia. |
| `cv_macro_auprc` | Mejor PR-AUC macro medio en CV entre sus arquitecturas. |
| `ensemble_cal_macro_auprc` | PR-AUC macro del ensemble en calibración. |
| `ensemble_cal_macro_brier` | Brier macro del ensemble en calibración tras temperatura. |
| `macro_specificity_at_sens85` | Especificidad macro del ensemble al punto sens85 (calibración). |
| `selected` | True para la familia ganadora. |
| `reason` | Criterio que decidió. |

## family_test_difference.csv

Diferencia pareada de PR-AUC (xrv − imagenet) en test mediante bootstrap por paciente.

| columna | significado |
|---|---|
| `label` | Patología o 'macro'. |
| `difference` | PR-AUC xrv − imagenet. |
| `low` | Límite inferior del IC 95% (bootstrap por paciente). |
| `high` | Límite superior del IC 95% (bootstrap por paciente). |
| `n_valid` | Réplicas con métrica definida. |
| `n_total` | Réplicas solicitadas. |
| `interpretation` | Interpretación exploratoria del IC. |

## label_audit.csv

Auditoría de etiquetas registradas, negativos aparentes y ausentes por partición. No sustituye la verificación clínica.

| columna | significado |
|---|---|
| `split` | Partición. |
| `label` | Patología. |
| `positives` | Positivos registrados. |
| `negatives` | Negativos aparentes (sin otra patología registrada). |
| `unknown` | Ceros no confirmados: la imagen tiene otra patología registrada y se excluyen de la pérdida y de las métricas. |

## masking_sensitivity.csv

Análisis de sensibilidad: métricas con y sin excluir etiquetas ausentes.

| columna | significado |
|---|---|
| `family` | Familia. |
| `model` | Modelo o ensemble. |
| `label` | Patología. |
| `masking` | masked (ceros con otra patología excluidos) o unmasked (se tratan como negativos). |
| `auprc` | PR-AUC. |
| `auroc` | AUROC. |
| `n` | Muestras usadas. |

## operating_points.csv

Curva de umbrales fijados en calibración y evaluados en el conjunto indicado. Compensan sensibilidad y especificidad; 'principal' es el punto recomendado.

| columna | significado |
|---|---|
| `family` | Familia. |
| `model` | Modelo o ensemble. |
| `label` | Patología. |
| `point` | sens95/sens90/sens85/sens80 (sensibilidad mínima en calibración), youden (máx. S+E−1, referencia). |
| `principal` | True para el punto principal (máxima especificidad con sensibilidad ≥ 85%). |
| `threshold` | Umbral sobre la probabilidad calibrada (vacío si no se pudo ajustar). |
| `cal_sensitivity` | Sensibilidad lograda en calibración. |
| `cal_specificity` | Especificidad lograda en calibración. |
| `status` | ok o razón por la que no hay umbral. |

## reliability_curves.csv

Puntos de las curvas de fiabilidad por bin de probabilidad.

| columna | significado |
|---|---|
| `family` | Familia. |
| `model` | Modelo o ensemble. |
| `variant` | raw o temperature. |
| `label` | Patología. |
| `bin` | Índice de bin. |
| `bin_low` | Límite inferior del bin. |
| `bin_high` | Límite superior del bin. |
| `n` | Muestras en el bin. |
| `mean_predicted` | Probabilidad media predicha. |
| `observed_frequency` | Frecuencia observada de positivos. |

## run_info.csv

Configuración y entorno de la corrida (reproducibilidad).

| columna | significado |
|---|---|
| `key` | Parámetro. |
| `value` | Valor (JSON). |

## splits.csv

Partición congelada de las imágenes. El test se usa una sola vez, al final, con modelos y umbrales ya fijados. Se reutiliza al reanudar.

| columna | significado |
|---|---|
| `source_row` | Fila de origen en export.csv; identifica la imagen. |
| `group_id` | Identificador de paciente (las particiones nunca separan un grupo). |
| `split` | development (CV e hiperparámetros), calibration (temperatura y umbrales) o test. |

## test_confusion.csv

Matrices de confusión en test por punto operativo.

| columna | significado |
|---|---|
| `family` | Familia. |
| `model` | Modelo o ensemble. |
| `label` | Patología. |
| `point` | Punto operativo. |
| `TP` | Verdaderos positivos. |
| `FP` | Falsos positivos. |
| `FN` | Falsos negativos. |
| `TN` | Verdaderos negativos. |

## test_metrics.csv

Evaluación en test interno con umbrales congelados. Si la cohorte ya se examinó, no constituye una confirmación independiente. IC por paciente.

| columna | significado |
|---|---|
| `family` | Familia. |
| `model` | Modelo o ensemble. |
| `label` | Patología. |
| `point` | Punto operativo. |
| `metric` | Métrica. |
| `value` | Estimación puntual. |
| `low` | Límite inferior del IC 95% (bootstrap por paciente). |
| `high` | Límite superior del IC 95% (bootstrap por paciente). |
| `n_unknown` | Etiquetas ausentes excluidas. |

## test_pr_auc.csv

PR-AUC por etiqueta en test con prevalencia como línea base.

| columna | significado |
|---|---|
| `family` | Familia. |
| `model` | Modelo o ensemble. |
| `label` | Patología. |
| `auprc` | PR-AUC. |
| `prevalence` | Prevalencia (PR-AUC de un clasificador aleatorio). |
| `auprc_low` | IC 95% inferior. |
| `auprc_high` | IC 95% superior. |
