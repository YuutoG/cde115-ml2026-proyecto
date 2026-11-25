"""Entrenamiento y evaluación reproducibles sin utilizar el test para selección."""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from . import reporting
from .calibration import apply_temperatures, fit_temperatures
from .data import (
    LABELS,
    Dataset,
    _sha256_file,
    grouped_multilabel_kfold,
    grouped_train_test_split,
    label_mask,
)
from .imaging import XRayDataset, read_image, warm_cache
from .models import FAMILY_MODELS, IMAGE_SIZES, build_model, save_checkpoint
from .stats import StatsStore
from .training import (
    bootstrap_patient_metrics,
    predict_logits,
    search_grouped_cv,
    seed_everything,
    train_final,
)

WARMUP_EPOCHS = 2


def _records_hash(records: pd.DataFrame) -> str:
    columns = ["source_row", "group_id", "image_sha256", *LABELS]
    columns += [c for c in ("other_pathology", "age", "sex") if c in records]
    payload = records.loc[:, columns].sort_values("source_row").to_json(
        orient="records", double_precision=15
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_fingerprint(development, calibration, config, family, use_mask):
    modules = ("data.py", "imaging.py", "models.py", "training.py", "pipeline.py",
               "calibration.py", "reporting.py")
    payload = {
        "development": _records_hash(development),
        "calibration": _records_hash(calibration),
        "config": config,
        "family": family,
        "use_mask": use_mask,
        "code": {name: _sha256_file(Path(__file__).parent / name) for name in modules},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "torchvision", "torchxrayvision", "optuna")},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def valid_images(records: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    valid = []
    for path in records["image_path"]:
        try:
            read_image(path)
            valid.append(True)
        except (OSError, ValueError):
            valid.append(False)
    return records.loc[valid].reset_index(drop=True), len(valid) - sum(valid)


def split_records(records: pd.DataFrame, seed: int = 42):
    wrapper = Dataset(records, pd.DataFrame(), {})
    development, test = grouped_train_test_split(
        wrapper, test_size=0.2, random_state=seed
    )
    development, calibration = grouped_train_test_split(
        Dataset(development, pd.DataFrame(), {}), test_size=0.2, random_state=seed + 1
    )
    parts = (development, calibration, test)
    for i, part in enumerate(parts):
        if part.empty:
            raise ValueError("Una partición está vacía")
        for label in LABELS:
            if part[label].nunique() != 2:
                raise ValueError(
                    f"Partición {i}: faltan positivos o negativos para {label}"
                )
        for other in parts[:i]:
            if set(part.group_id) & set(other.group_id) or set(part.image_sha256) & set(
                other.image_sha256
            ):
                raise ValueError("Fuga de pacientes o duplicados entre particiones")
    return parts


def default_workers() -> int:
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return max(1, min(8, count or 1))


def loader(
    records,
    size,
    batch_size,
    *,
    crop_black=False,
    augment=False,
    shuffle=False,
    seed=42,
    mask=False,
    cache=False,
    workers=None,
):
    """DataLoader con máscara opcional y caché en memoria si todo el conjunto cabe.

    Con caché completo se usa ``num_workers=0`` (los tensores ya están en RAM);
    si no cabe, se descarta el caché y se paralelizan lecturas con procesos.
    """
    generator = torch.Generator().manual_seed(seed)
    workers = default_workers() if workers is None else workers
    use_cache = False
    if cache:
        use_cache = warm_cache(records, size, crop_black=crop_black, workers=workers) == len(records)
    return DataLoader(
        XRayDataset(records, size, crop_black=crop_black, augment=augment, mask=mask,
                    cache=use_cache),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 if use_cache else workers,
        persistent_workers=not use_cache and workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def frozen_split(records: pd.DataFrame, seed: int, store: StatsStore | None = None):
    """Partición desarrollo/calibración/test persistida en ``stats/splits.csv``.

    Si el archivo existe se reutiliza (el test sigue siendo el mismo en cada
    reanudación) y se valida contra los registros; si no, se crea con ``seed``.
    """
    if store is not None and store.has("splits"):
        saved = pd.read_csv(store.path("splits"), dtype={"group_id": str}, keep_default_na=False)
        identity = store.read_json("splits_identity")
        if identity != {"records_hash": _records_hash(records), "seed": seed}:
            raise ValueError("La identidad de la cohorte o semilla no coincide con la partición guardada")
        if saved.source_row.duplicated().any() or records.source_row.duplicated().any():
            raise ValueError("Filas duplicadas en la partición o los registros")
        if set(saved.split) != {"development", "calibration", "test"}:
            raise ValueError("Nombres de partición inválidos o partición vacía")
        by_row = dict(zip(saved.source_row, saved.split, strict=True))
        if set(by_row) != set(records.source_row):
            raise ValueError("stats/splits.csv no coincide con los registros actuales; "
                             "bórrelo solo si desea una partición nueva")
        parts = tuple(records.loc[records.source_row.map(by_row) == name].reset_index(drop=True)
                      for name in ("development", "calibration", "test"))
        groups = dict(zip(saved.source_row, saved.group_id, strict=True))
        if any(groups[row.source_row] != str(row.group_id) for row in records.itertuples()):
            raise ValueError("Los grupos guardados no coinciden con la cohorte")
        for i, part in enumerate(parts):
            for other in parts[:i]:
                if set(part.group_id) & set(other.group_id) or set(part.image_sha256) & set(other.image_sha256):
                    raise ValueError("Fuga de pacientes o duplicados entre particiones")
        return parts
    if records.source_row.duplicated().any():
        raise ValueError("Filas de origen duplicadas")
    parts = split_records(records, seed)
    if store is not None:
        store.write_json("splits_identity", {"records_hash": _records_hash(records), "seed": seed})
        store.write("splits", pd.concat([
            pd.DataFrame({"source_row": part.source_row, "group_id": part.group_id, "split": name})
            for name, part in zip(("development", "calibration", "test"), parts, strict=True)
        ], ignore_index=True), config={"seed": seed})
    return parts


def label_audit_frame(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, part in parts.items():
        known = label_mask(part)
        for label in LABELS:
            positives = int(part[label].sum())
            confirmed = int(known[label].sum())
            rows.append({"split": name, "label": label, "positives": positives,
                         "negatives": confirmed - positives, "unknown": len(part) - confirmed})
    return pd.DataFrame(rows)


def _report_training_progress(model_name, event):
    prefix = f"{model_name} | "
    if event["event"] == "epoch":
        if event["phase"] == "cv":
            context = (
                f"prueba {event['trial']}/{event['total_trials']}, "
                f"fold {event['fold']}/{event['total_folds']}"
            )
        else:
            context = "entrenamiento final"
        message = (
            f"{prefix}{context} | época {event['epoch']}/{event['total_epochs']} "
            f"| pérdida train={event['train_loss']:.4f}"
        )
        if event.get("val_pr_auc") is not None and event["val_pr_auc"] == event["val_pr_auc"]:
            message += f", PR-AUC val={event['val_pr_auc']:.3f}"
        if event["val_loss"] is not None:
            message += (
                f", validación={event['val_loss']:.4f} "
                f"(mejor época {event['best_epoch']}, "
                f"pérdida={event['best_val_loss']:.4f})"
            )
        message += (
            f" | {event['epoch_seconds']:.1f}s esta época, "
            f"{event['elapsed_seconds']:.1f}s transcurridos"
        )
        if event["early_stopped"]:
            message += " | parada temprana"
        print(message, flush=True)
    elif event["event"] == "fold_complete":
        print(
            f"{prefix}prueba {event['trial']}/{event['total_trials']}, "
            f"fold {event['fold']}/{event['total_folds']} completado "
            f"| PR-AUC macro={event.get('macro_auprc') or float('nan'):.3f}, "
            f"AUROC macro={event['macro_auroc']:.3f} "
            f"| mejor época={event['best_epoch']}",
            flush=True,
        )
    elif event["event"] == "trial_complete":
        print(
            f"{prefix}prueba {event['trial']}/{event['total_trials']} completada "
            f"| {event.get('objective', 'auroc')} macro CV={event['score']:.3f} "
            f"| parámetros={event['params']}",
            flush=True,
        )


def _replace_rows(store, name, frame, family, model):
    if store.has(name):
        old = store.read(name)
        old = old.loc[~((old.family == family) & (old.model == model))]
        frame = pd.concat([old, frame], ignore_index=True)
    store.write(name, frame)


def _trial_rows(family, model, study, optimizer):
    rows = []
    for trial in study.trials:
        for report in trial.user_attrs.get("folds", []):
            rows.append({
                "family": family, "model": model, "optimizer": optimizer, "trial": trial.number + 1,
                "fold": report["fold"] + 1, "best_epoch": report["best_epoch"],
                "n_epochs_run": report["n_epochs_run"], "macro_auprc": report["macro_auprc"],
                "macro_auroc": report["macro_auroc"], "macro_auprc_tta": report["macro_auprc_tta"],
                "macro_auroc_tta": report["macro_auroc_tta"],
                "params": json.dumps(trial.user_attrs.get("resolved_params", trial.params)),
            })
    return pd.DataFrame(rows)


def _suggest(trial, optimizer="adamw", augmentation=None):
    lr_bounds = (1e-5, 3e-4) if optimizer == "adamw" else (1e-4, 1e-2)
    return {
        "optimizer": optimizer,
        "lr": trial.suggest_float("lr", *lr_bounds, log=True),
        "head_lr_mult": trial.suggest_float("head_lr_mult", 1.0, 20.0, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "crop_black": trial.suggest_categorical("crop_black", [False, True]),
        "augment": (augmentation if augmentation is not None else
                    trial.suggest_categorical("augment", ["none", "rotate", "translate", "rotate_translate"])),
        "loss": trial.suggest_categorical("loss", ["bce", "asl"]),
    }


def _optimizer(model, params):
    head = list(model.head_parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    groups = [{"params": body, "lr": params["lr"]},
              {"params": head, "lr": params["lr"] * params["head_lr_mult"]}]
    if params.get("optimizer", "adamw") == "adamw":
        return torch.optim.AdamW(groups, weight_decay=params["weight_decay"])
    if params["optimizer"] == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=params["weight_decay"])
    raise ValueError(f"Optimizador desconocido: {params['optimizer']}")


def run_nested_experiments(development, config, *, families=("xrv", "imagenet"),
                           optimizers=("adamw", "sgd"),
                           augmentations=("none", "rotate", "translate", "rotate_translate"),
                           seeds=(42, 43, 44), outer_folds=3, inner_folds=3,
                           store=None, use_mask=True):
    """Evalúa brazos predefinidos mediante CV anidada agrupada y predicciones OOF.

    Cada brazo ajusta los demás hiperparámetros exclusivamente dentro del
    entrenamiento exterior. Los ensembles son medias de probabilidades sin
    calibrar; no son los ensembles finales calibrados y no usan test.
    """
    records = development.reset_index(drop=True)
    if records.source_row.duplicated().any():
        raise ValueError("Filas de origen duplicadas")
    if not families or len(set(families)) != len(families) or any(f not in FAMILY_MODELS for f in families):
        raise ValueError("Familias inválidas o duplicadas")
    if not optimizers or len(set(optimizers)) != len(optimizers) or any(o not in ("adamw", "sgd") for o in optimizers):
        raise ValueError("Optimizadores inválidos o duplicados")
    allowed_augmentations = ("none", "rotate", "translate", "rotate_translate")
    if not augmentations or len(set(augmentations)) != len(augmentations) or any(a not in allowed_augmentations for a in augmentations):
        raise ValueError("Aumentos inválidos o duplicados")
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
        raise ValueError("Semillas inválidas o duplicadas")
    y = records.loc[:, LABELS].to_numpy(dtype=int)
    masks = label_mask(records).to_numpy(dtype=int) if use_mask else np.ones_like(y)
    positions = {row: i for i, row in enumerate(records.source_row)}
    device = config["device"]
    amp = bool(config.get("amp", False))
    cache = bool(config.get("cache", False))
    warmup = min(int(config.get("warmup_epochs", WARMUP_EPOCHS)), max(0, config["epochs"] - 1))
    prediction_rows, selection_rows, metric_rows = [], [], []
    experiment_config = {**config, "families": list(families), "optimizers": list(optimizers),
                         "augmentations": list(augmentations), "seeds": list(seeds),
                         "outer_folds": outer_folds, "inner_folds": inner_folds}
    fingerprint = _run_fingerprint(records, records.iloc[:0], experiment_config, "nested", use_mask)
    cache_dir = None
    if store is not None:
        identity = {"fingerprint": fingerprint, "config": experiment_config, "use_mask": use_mask}
        previous = store.read_json("nested_identity")
        if previous is not None and previous != identity:
            raise ValueError("Identidad de CV anidada incompatible; use otro directorio de corrida")
        if previous is None and any(store.has(name) for name in ("nested_oof", "nested_selection", "nested_metrics")):
            raise ValueError("Resultados anidados sin identidad; use otro directorio de corrida")
        store.write_json("nested_identity", identity)
        cache_dir = store.directory / "nested_cache"
        cache_dir.mkdir(exist_ok=True)

    for seed in seeds:
        outer = grouped_multilabel_kfold(Dataset(records, pd.DataFrame(), {}),
                                         n_splits=outer_folds, random_state=seed)
        for outer_index, (outer_train, outer_valid) in enumerate(outer, 1):
            train_pos = np.array([positions[row] for row in outer_train.source_row])
            valid_pos = np.array([positions[row] for row in outer_valid.source_row])
            local_positions = {row: i for i, row in enumerate(outer_train.source_row)}
            inner = grouped_multilabel_kfold(Dataset(outer_train, pd.DataFrame(), {}),
                                             n_splits=inner_folds, random_state=seed + outer_index)
            fold_indices = [([local_positions[x] for x in a.source_row],
                             [local_positions[x] for x in b.source_row]) for a, b in inner]
            for family in families:
                for optimizer_name in optimizers:
                    for augmentation in augmentations:
                        cache_path = (cache_dir / f"{seed}_{outer_index}_{family}_{optimizer_name}_{augmentation}.json"
                                      if cache_dir is not None else None)
                        if cache_path is not None and cache_path.exists():
                            saved = json.loads(cache_path.read_text(encoding="utf-8"))
                            if saved.get("fingerprint") != fingerprint or \
                                    saved.get("source_rows") != outer_valid.source_row.tolist() or \
                                    len(saved.get("selection", [])) != len(FAMILY_MODELS[family]):
                                raise ValueError(f"Resultados anidados incompatibles: {cache_path}")
                            ensemble = np.asarray(saved["probabilities"], dtype=float)
                            if ensemble.shape != (len(valid_pos), len(LABELS)) or \
                                    not np.isfinite(ensemble).all() or np.any((ensemble < 0) | (ensemble > 1)):
                                raise ValueError(f"Predicciones anidadas inválidas: {cache_path}")
                            selection_rows.extend(saved["selection"])
                        else:
                            ensemble = None
                        member_probabilities = []
                        arm_selections = []
                        for name in (() if ensemble is not None else FAMILY_MODELS[family]):
                            size = IMAGE_SIZES[name]

                            def make_loaders(a, b, params, fold_seed):
                                return (
                                    loader(outer_train.iloc[a], size, config["batch_size"],
                                           crop_black=params["crop_black"], augment=params["augment"],
                                           shuffle=True, seed=fold_seed, mask=use_mask, cache=cache,
                                           workers=config.get("workers")),
                                    loader(outer_train.iloc[b], size, config["batch_size"],
                                           crop_black=params["crop_black"], mask=use_mask, cache=cache,
                                           workers=config.get("workers")),
                                )

                            def fit_options(params, model):
                                options = {"loss": params["loss"], "cosine": True}
                                if warmup:
                                    options.update(warmup_epochs=warmup, head_parameters=model.head_parameters())
                                return options

                            result = search_grouped_cv(
                                y[train_pos], records.group_id.to_numpy()[train_pos], fold_indices,
                                suggest=lambda trial: _suggest(trial, optimizer_name, augmentation),
                                make_loaders=make_loaders,
                                make_model=lambda params: build_model(name, pretrained=True),
                                make_optimizer=_optimizer, n_trials=config["trials"],
                                max_epochs=config["epochs"], patience=config["patience"],
                                seed=seed, device=device, mask=masks[train_pos], objective="auprc",
                                monitor="pr_auc", fit_options=fit_options, prune=False, amp=amp,
                            )
                            params = result["params"]
                            seed_everything(seed + outer_index)
                            model = build_model(name, pretrained=True)
                            train_loader = loader(outer_train, size, config["batch_size"],
                                                  crop_black=params["crop_black"], augment=augmentation,
                                                  shuffle=True, seed=seed + outer_index, mask=use_mask,
                                                  cache=cache, workers=config.get("workers"))
                            train_final(model, train_loader, y[train_pos], _optimizer(model, params),
                                        epochs=max(result["epochs"], warmup + 1), seed=seed + outer_index,
                                        device=device, mask_train=masks[train_pos], loss=params["loss"],
                                        cosine=True, amp=amp,
                                        **({"warmup_epochs": warmup, "head_parameters": model.head_parameters()}
                                           if warmup else {}))
                            actual, logits, val_mask = predict_logits(
                                model, loader(outer_valid, size, config["batch_size"],
                                              crop_black=params["crop_black"], mask=use_mask, cache=cache,
                                              workers=config.get("workers")),
                                device=device, tta=result["tta"], amp=amp)
                            if not np.array_equal(actual, y[valid_pos]) or not np.array_equal(val_mask, masks[valid_pos]):
                                raise ValueError("Orden o máscara de validación exterior incorrectos")
                            member_probabilities.append(1 / (1 + np.exp(-logits.astype(np.float64))))
                            arm_selections.append({
                                "seed": seed, "fold": outer_index, "family": family, "model": name,
                                "optimizer": optimizer_name, "augmentation": augmentation,
                                "inner_ap": result["score"], "epochs": max(result["epochs"], warmup + 1),
                                "tta": result["tta"], "params": json.dumps(params),
                            })
                            del model, train_loader
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        if ensemble is None:
                            ensemble = np.mean(member_probabilities, axis=0)
                            selection_rows.extend(arm_selections)
                            if cache_path is not None:
                                payload = {"fingerprint": fingerprint,
                                           "source_rows": outer_valid.source_row.tolist(),
                                           "probabilities": ensemble.tolist(), "selection": arm_selections}
                                temporary = cache_path.with_suffix(".tmp")
                                temporary.write_text(json.dumps(payload, allow_nan=False, default=str), encoding="utf-8")
                                temporary.replace(cache_path)
                        for i, position in enumerate(valid_pos):
                            for j, label in enumerate(LABELS):
                                prediction_rows.append({
                                    "seed": seed, "fold": outer_index, "family": family,
                                    "optimizer": optimizer_name, "augmentation": augmentation,
                                    "source_row": records.source_row.iloc[position],
                                    "group_id": records.group_id.iloc[position], "label": label,
                                    "truth": y[position, j], "known": masks[position, j],
                                    "probability": ensemble[i, j],
                                })

    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    for keys, subset in predictions.groupby(["seed", "family", "optimizer", "augmentation"], sort=False):
        if len(subset) != len(records) * len(LABELS) or subset.duplicated(["source_row", "label"]).any():
            raise ValueError("Predicciones OOF incompletas o duplicadas")
        for label in LABELS:
            part = subset.loc[(subset.label == label) & (subset.known == 1)]
            actual = part.truth.to_numpy(dtype=int)
            valid = np.unique(actual).size == 2
            metric_rows.append(dict(zip(("seed", "family", "optimizer", "augmentation"), keys, strict=True)) | {
                "label": label, "known": len(part), "positive": int(actual.sum()),
                "ap": float(average_precision_score(actual, part.probability)) if valid else np.nan,
                "auroc": float(roc_auc_score(actual, part.probability)) if valid else np.nan,
                "status": "ok" if valid else "sin ambas clases conocidas",
            })
    metrics = pd.DataFrame(metric_rows)
    if store is not None:
        store.write("nested_oof", predictions)
        store.write("nested_selection", selections)
        store.write("nested_metrics", metrics)
    return {"oof": predictions, "selection": selections, "metrics": metrics}


def compare_nested_arms(oof, *, n_bootstrap=1000, seed=42, store=None):
    """Compara brazos OOF pareados, cambiando un factor por contraste."""
    outputs = []
    for (run_seed, family), group in oof.groupby(["seed", "family"], sort=False):
        arms = {(opt, aug): frame for (opt, aug), frame in
                group.groupby(["optimizer", "augmentation"], sort=False)}
        comparisons = []
        for optimizer, augmentation in arms:
            if optimizer != "adamw" and ("adamw", augmentation) in arms:
                comparisons.append(((optimizer, augmentation), ("adamw", augmentation), "optimizer"))
            if augmentation != "none" and (optimizer, "none") in arms:
                comparisons.append(((optimizer, augmentation), (optimizer, "none"), "augmentation"))
        for arm, reference, factor in comparisons:
            keys = ["source_row", "group_id", "fold", "label", "truth", "known"]
            left, right = arms[arm], arms[reference]
            merged = left.merge(right, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
            if len(merged) != len(left) or len(merged) != len(right):
                raise ValueError("Los brazos OOF no están emparejados por paciente, fold y etiqueta")
            pivot = lambda col: merged.pivot(index="source_row", columns="label", values=col).loc[:, LABELS]
            truth = pivot("truth").to_numpy(dtype=int)
            mask = pivot("known").to_numpy(dtype=int)
            p_a, p_b = pivot("probability_a").to_numpy(), pivot("probability_b").to_numpy()
            groups = merged.drop_duplicates("source_row").set_index("source_row").loc[
                pivot("truth").index, "group_id"].to_numpy()
            differences = reporting.paired_auprc_difference(
                truth, p_a, p_b, groups, mask=mask, n_bootstrap=n_bootstrap, seed=seed)
            differences.insert(0, "reference", "/".join(reference))
            differences.insert(0, "arm", "/".join(arm))
            differences.insert(0, "factor", factor)
            differences.insert(0, "family", family)
            differences.insert(0, "seed", run_seed)
            outputs.append(differences)
    if not outputs:
        raise ValueError("No hay brazos emparejados para comparar")
    result = pd.concat(outputs, ignore_index=True)
    if store is not None:
        store.write("nested_comparisons", result)
    return result


def run_training(development, calibration, model_dir, config, *, family="xrv", store=None,
                 use_mask=True):
    """Optimiza dentro de desarrollo; calibración solo ajusta temperatura y umbrales.

    ``family`` selecciona los pesos iniciales (``xrv`` o ``imagenet``). Con ``store``
    se guardan estadísticas y la búsqueda Optuna es reanudable; los modelos ya
    entrenados en ``model_dir`` se reutilizan si su checkpoint coincide con su hash.
    """
    names = FAMILY_MODELS[family]
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    device = config["device"]
    amp = bool(config.get("amp", torch.cuda.is_available()))
    cache = bool(config.get("cache", False))
    warmup = min(int(config.get("warmup_epochs", WARMUP_EPOCHS)), max(0, config["epochs"] - 1))
    y = development.loc[:, LABELS].to_numpy(dtype=int)
    mask = label_mask(development).to_numpy(dtype=int) if use_mask else np.ones_like(y)
    y_cal = calibration.loc[:, LABELS].to_numpy(dtype=int)
    mask_cal = label_mask(calibration).to_numpy(dtype=int) if use_mask else np.ones_like(y_cal)
    fingerprint = _run_fingerprint(development, calibration, config, family, use_mask)
    folds = grouped_multilabel_kfold(
        Dataset(development, pd.DataFrame(), {}), n_splits=config["folds"], random_state=seed,
    )
    positions = {row: i for i, row in enumerate(development.source_row)}
    fold_indices = [
        ([positions[x] for x in train.source_row],
         [positions[x] for x in val.source_row])
        for train, val in folds
    ]
    manifest = {
        "family": family,
        "labels": list(LABELS),
        "models": {},
        "ensemble_weights": [1 / len(names)] * len(names),
        "thresholds": {},
        "operating_points": {},
        "principal_point": reporting.PRINCIPAL,
        "seed": seed,
        "training_config": config,
        "label_masking": use_mask,
        "run_fingerprint": fingerprint,
        "development_hash": hashlib.sha256(
            "\n".join(sorted(development.image_sha256)).encode("ascii")
        ).hexdigest(),
        "versions": {
            "torch": str(torch.__version__),
            "torchxrayvision": importlib.metadata.version("torchxrayvision"),
        },
        "purpose": "research_only",
    }
    cal_raw, cal_temp = {}, {}
    cv_best_rows = []
    for name in names:
        size = IMAGE_SIZES[name]
        result_path = directory / f"{name}.result.json"
        logits_path = directory / f"{name}.cal_logits.npy"
        checkpoint = directory / f"{name}.pt"
        if result_path.is_file() and logits_path.is_file() and checkpoint.is_file():
            entry = json.loads(result_path.read_text(encoding="utf-8"))
            if entry.get("sha256") == _sha256_file(checkpoint) and \
                    entry.get("run_fingerprint") == fingerprint and \
                    entry.get("cal_logits_sha256") == _sha256_file(logits_path):
                print(f"\n{name}: se reutiliza el entrenamiento guardado en {directory}", flush=True)
                logits = np.load(logits_path)
                manifest["models"][name] = {k: v for k, v in entry.items()
                                            if k not in ("development_hash", "run_fingerprint",
                                                         "cal_logits_sha256")}
                cal_raw[name] = logits
                continue
            raise ValueError(f"Artefactos incompatibles para {name}; use un directorio de corrida nuevo")
        if any(path.exists() for path in (result_path, logits_path, checkpoint)):
            raise ValueError(f"Artefactos incompletos para {name}; use un directorio de corrida nuevo")
        print(
            f"\n[{family}] {name}: búsqueda CV con {len(development)} imágenes, "
            f"{len(fold_indices)} folds y {config['trials']} pruebas "
            f"(hasta {config['epochs']} épocas, paciencia {config['patience']}; dispositivo={device})",
            flush=True,
        )

        def make_loaders(a, b, params, fold_seed, image_size=size):
            return (
                loader(development.iloc[a], image_size, config["batch_size"],
                       crop_black=params["crop_black"], augment=params["augment"],
                       shuffle=True, seed=fold_seed, mask=use_mask, cache=cache, workers=config.get("workers")),
                loader(development.iloc[b], image_size, config["batch_size"],
                       crop_black=params["crop_black"], mask=use_mask, cache=cache, workers=config.get("workers")),
            )

        def fit_options(params, model):
            options = {"loss": params["loss"], "cosine": True}
            if warmup:
                options.update(warmup_epochs=warmup, head_parameters=model.head_parameters())
            return options

        optimizer_names = config.get("optimizers", ["adamw"])
        if not optimizer_names or len(set(optimizer_names)) != len(optimizer_names) or \
                any(name not in ("adamw", "sgd") for name in optimizer_names):
            raise ValueError("optimizers debe contener adamw y/o sgd sin duplicados")
        results = []
        for optimizer_name in optimizer_names:
            result = search_grouped_cv(
                y, development.group_id.to_numpy(), fold_indices,
                suggest=lambda trial, opt=optimizer_name: _suggest(trial, opt), make_loaders=make_loaders,
                make_model=lambda params, model_name=name: build_model(model_name, pretrained=True),
                make_optimizer=_optimizer,
                n_trials=config["trials"], max_epochs=config["epochs"], patience=config["patience"],
                seed=seed, device=device,
                initial_trials=[
                    {"crop_black": False, "augment": "none", "loss": "bce"},
                    {"crop_black": True, "augment": "none", "loss": "bce"},
                    {"crop_black": False, "augment": "rotate", "loss": "asl"},
                    {"crop_black": True, "augment": "rotate", "loss": "asl"},
                ],
                progress_callback=lambda event, model_name=name: _report_training_progress(
                    model_name, event),
                mask=mask, objective="auprc", monitor="pr_auc", fit_options=fit_options,
                storage=store.storage_url(f"optuna_{family}_{name}_{optimizer_name}_{fingerprint}") if store else None,
                study_name=f"{family}-{name}-{optimizer_name}-{fingerprint}", prune=True, amp=amp,
            )
            results.append((optimizer_name, result))
        _, result = max(results, key=lambda item: item[1]["score"])
        params = result["params"]
        print(
            f"{name}: mejor PR-AUC macro CV={result['score']:.3f}; épocas={result['epochs']}; "
            f"TTA={'sí' if result['tta'] else 'no'}; parámetros={params}", flush=True)
        if store is not None:
            _replace_rows(store, "cv_trials", pd.concat(
                [_trial_rows(family, name, item["study"], opt) for opt, item in results], ignore_index=True
            ), family, name)
        final_epochs = max(result["epochs"], warmup + 1)
        seed_everything(seed)
        model = build_model(name, pretrained=True)
        train_loader = loader(development, size, config["batch_size"],
                              crop_black=params["crop_black"], augment=params["augment"],
                              shuffle=True, seed=seed, mask=use_mask, cache=cache, workers=config.get("workers"))
        train_final(
            model, train_loader, y, _optimizer(model, params), epochs=final_epochs, seed=seed,
            device=device, mask_train=mask, loss=params["loss"], cosine=True, amp=amp,
            **({"warmup_epochs": warmup, "head_parameters": model.head_parameters()} if warmup else {}),
            progress_callback=lambda event, model_name=name: _report_training_progress(
                model_name, {**event, "phase": "final"}),
        )
        save_checkpoint(checkpoint, model)
        actual, logits, _ = predict_logits(
            model,
            loader(calibration, size, config["batch_size"], crop_black=params["crop_black"],
                   mask=use_mask, cache=cache, workers=config.get("workers")),
            device=device, tta=result["tta"], amp=amp)
        if not np.array_equal(actual, y_cal):
            raise ValueError("Orden de calibración incorrecto")
        np.save(logits_path, logits)
        entry = {
            "file": checkpoint.name,
            "sha256": _sha256_file(checkpoint),
            "preprocessing": {"crop_black": params["crop_black"], "max_value": None},
            "params": params,
            "epochs": final_epochs,
            "cv_macro_auprc": result["score"],
            "tta": result["tta"],
        }
        result_path.write_text(
            json.dumps({**entry, "development_hash": manifest["development_hash"],
                        "run_fingerprint": fingerprint, "cal_logits_sha256": _sha256_file(logits_path)}, indent=2),
            encoding="utf-8")
        manifest["models"][name] = entry
        cal_raw[name] = logits
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    temperature_rows, calibration_metrics, curves, split_rows, point_rows = [], [], [], [], []
    for name in names:
        fits = fit_temperatures(cal_raw[name], y_cal, mask_cal)
        manifest["models"][name]["temperatures"] = [f["temperature"] for f in fits]
        cal_temp[name] = apply_temperatures(cal_raw[name], manifest["models"][name]["temperatures"])
        for label, f in zip(LABELS, fits, strict=True):
            temperature_rows.append({
                "family": family, "model": name, "label": label,
                "temperature": f["temperature"], "at_bound": f["at_bound"],
                "nll_before": f["nll_before"], "nll_after": f["nll_after"], "status": f["status"]})
        cv_best_rows.append({
            "family": family, "model": name,
            "cv_macro_auprc": manifest["models"][name]["cv_macro_auprc"],
            "epochs": manifest["models"][name]["epochs"], "tta": manifest["models"][name]["tta"],
            "params": json.dumps(manifest["models"][name]["params"])})
    raw_probabilities = {n: 1 / (1 + np.exp(-cal_raw[n].astype(np.float64))) for n in names}
    weights = manifest["ensemble_weights"]
    raw_probabilities["ensemble"] = sum(w * raw_probabilities[n] for w, n in zip(weights, names))
    cal_temp["ensemble"] = sum(w * cal_temp[n] for w, n in zip(weights, names))
    for name in (*names, "ensemble"):
        points = reporting.fit_operating_points(y_cal, cal_temp[name], mask_cal)
        manifest["thresholds"][name] = reporting.thresholds_of(points)
        manifest["operating_points"][name] = reporting.all_thresholds(points)
        point_rows.append(reporting.operating_points_frame(family, name, points))
        metrics, curve = reporting.calibration_frames(
            family, name, y_cal, {"raw": raw_probabilities[name], "temperature": cal_temp[name]}, mask_cal)
        calibration_metrics.append(metrics)
        curves.append(curve)
        split_rows.append(reporting.split_metrics_frame(family, name, y_cal, cal_temp[name], mask_cal))
        if name == "ensemble":
            manifest["calibration_summary"] = {
                "macro_auprc": reporting.macro_auprc(y_cal, cal_temp[name], mask_cal),
                "macro_brier": reporting.macro_brier(y_cal, cal_temp[name], mask_cal),
                "macro_specificity_sens85": reporting.macro_specificity(
                    y_cal, cal_temp[name], points, "sens85", mask_cal),
            }
    if store is not None:
        for table, frame in (("calibration_temperatures", pd.DataFrame(temperature_rows)),
                             ("cv_best", pd.DataFrame(cv_best_rows)),
                             ("operating_points", pd.concat(point_rows, ignore_index=True)),
                             ("calibration_metrics", pd.concat(calibration_metrics, ignore_index=True)),
                             ("reliability_curves", pd.concat(curves, ignore_index=True)),
                             ("calibration_split_metrics", pd.concat(split_rows, ignore_index=True))):
            for model in frame.model.unique():
                _replace_rows(store, table, frame.loc[frame.model == model], family, model)
        store.write("run_info", pd.DataFrame(
            [{"key": f"{family}.{k}", "value": json.dumps(v, default=str)}
             for k, v in {**config, "label_masking": use_mask, "versions": manifest["versions"]}.items()]
        ) if not store.has("run_info") else pd.concat([
            store.read("run_info").loc[lambda d: ~d.key.str.startswith(f"{family}.")],
            pd.DataFrame([{"key": f"{family}.{k}", "value": json.dumps(v, default=str)}
                          for k, v in {**config, "label_masking": use_mask,
                                       "versions": manifest["versions"]}.items()]),
        ], ignore_index=True))
    with (directory / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return manifest


def compare_families(manifests: dict[str, dict], store: StatsStore | None = None,
                     tolerance: float = 0.005) -> pd.DataFrame:
    """Selecciona ensembles por AP en calibración, antes de consultar test.

    Es una selección exploratoria en la misma cohorte usada para calibrar, no
    una estimación independiente del desempeño ni del efecto del preentrenamiento.
    """
    rows = []
    for family, manifest in manifests.items():
        summary = manifest["calibration_summary"]
        rows.append({
            "family": family,
            "cv_macro_auprc": max(m["cv_macro_auprc"] for m in manifest["models"].values()),
            "ensemble_cal_macro_auprc": summary["macro_auprc"],
            "ensemble_cal_macro_brier": summary["macro_brier"],
            "macro_specificity_at_sens85": summary["macro_specificity_sens85"],
        })
    table = pd.DataFrame(rows)
    if table.empty or not np.isfinite(table["ensemble_cal_macro_auprc"]).all():
        raise ValueError("No se puede seleccionar familia sin AP macro de calibración finita")
    ordered = table.sort_values("ensemble_cal_macro_auprc", ascending=False).reset_index(drop=True)
    ordered_ap = ordered["ensemble_cal_macro_auprc"].to_numpy(dtype=float)
    winner, reason = ordered.loc[0, "family"], "mayor AP macro del ensemble en calibración"
    if len(ordered) > 1 and ordered_ap[0] - ordered_ap[1] < tolerance:
        if not np.isfinite(ordered["macro_specificity_at_sens85"]).all():
            raise ValueError("No se puede desempatar: especificidad de calibración no disponible")
        specificity = ordered.sort_values("macro_specificity_at_sens85", ascending=False)
        first, second = specificity.iloc[0], specificity.iloc[1]
        if first.macro_specificity_at_sens85 - second.macro_specificity_at_sens85 > 1e-9:
            winner, reason = first.family, "empate en AP de calibración: mayor especificidad con sensibilidad 85%"
        else:
            if not np.isfinite(ordered["ensemble_cal_macro_brier"]).all():
                raise ValueError("No se puede desempatar: Brier de calibración no disponible")
            winner = ordered.sort_values("ensemble_cal_macro_brier").iloc[0].family
            reason = "empate en AP de calibración y especificidad: menor Brier"
    table["selected"] = table.family == winner
    table["reason"] = np.where(table.selected, reason, "")
    if store is not None:
        store.write("family_comparison", table)
    return table


def evaluate_test(models, manifest, records, config, *, store=None, use_mask=True):
    """El llamador solo invoca esta función tras congelar modelos y umbrales.

    Devuelve (tablas, predicciones): tablas con claves ``metrics``, ``confusion`` y
    ``pr_auc``; predicciones son probabilidades calibradas por modelo y ensemble.
    """
    family = manifest.get("family", "xrv")
    truth = records.loc[:, LABELS].to_numpy(dtype=int)
    mask = label_mask(records).to_numpy(dtype=int) if use_mask else np.ones_like(truth)
    amp = bool(config.get("amp", torch.cuda.is_available()))
    predictions = {}
    for name, model in models.items():
        entry = manifest["models"][name]
        size = IMAGE_SIZES[name]
        actual, logits, _ = predict_logits(
            model,
            loader(records, size, config["batch_size"], crop_black=entry["preprocessing"]["crop_black"],
                   mask=True, cache=bool(config.get("cache", False)),
                   workers=config.get("workers")),
            device=config["device"], tta=bool(entry.get("tta")), amp=amp)
        if not np.array_equal(actual, truth):
            raise ValueError("Orden de test incorrecto")
        predictions[name] = apply_temperatures(logits, entry.get("temperatures", [1.0] * len(LABELS)))
    predictions["ensemble"] = sum(
        weight * predictions[name] for weight, name in zip(manifest["ensemble_weights"], models, strict=True)
    )
    metrics, confusion, pr = [], [], []
    for name, probability in predictions.items():
        m, c, p = reporting.test_frames(
            family, name, truth, probability, records.group_id.to_numpy(),
            manifest["operating_points"][name], mask=mask,
            n_bootstrap=config["bootstrap"], seed=config["seed"])
        metrics.append(m)
        confusion.append(c)
        pr.append(p)
    frames = {"metrics": pd.concat(metrics, ignore_index=True),
              "confusion": pd.concat(confusion, ignore_index=True),
              "pr_auc": pd.concat(pr, ignore_index=True)}
    if store is not None:
        for key, table in (("metrics", "test_metrics"), ("confusion", "test_confusion"),
                           ("pr_auc", "test_pr_auc")):
            old = store.read(table) if store.has(table) else pd.DataFrame()
            if len(old):
                old = old.loc[old.family != family]
            store.write(table, pd.concat([old, frames[key]], ignore_index=True))
    return frames, predictions


def compare_on_test(records, predictions_a, predictions_b, config, store=None, use_mask=True):
    """Diferencia pareada de PR-AUC (a − b) entre ensembles calibrados en test."""
    truth = records.loc[:, LABELS].to_numpy(dtype=int)
    mask = label_mask(records).to_numpy(dtype=int) if use_mask else np.ones_like(truth)
    table = reporting.paired_auprc_difference(
        truth, predictions_a["ensemble"], predictions_b["ensemble"], records.group_id.to_numpy(),
        mask=mask, n_bootstrap=config["bootstrap"], seed=config["seed"])
    if store is not None:
        store.write("family_test_difference", table)
    return table


def fairness_table(
    records, probabilities, thresholds, *, minimum_patients=20, minimum_class_patients=10,
    n_bootstrap=200, seed=42, use_mask=True
):
    """Informa grupos pequeños sin presentar estimaciones inestables como fiables."""
    data = records.copy()
    data["age_group"] = pd.cut(
        data.age,
        [14, 17.999, 39.999, 59.999, 120],
        labels=["15–17", "18–39", "40–59", "60+"],
        include_lowest=True,
    )
    rows = []
    thresholds = [x if x is not None else float("nan") for x in thresholds]
    all_mask = label_mask(data).to_numpy(dtype=int) if use_mask else np.ones((len(data), len(LABELS)))
    for column in ("sex", "age_group"):
        for group, indices in data.groupby(
            column, observed=True, dropna=False
        ).indices.items():
            subset = data.iloc[indices]
            count = subset.group_id.nunique()
            if count < minimum_patients:
                rows.append(
                    {
                        "attribute": column,
                        "group": str(group),
                        "patients": count,
                        "status": "muestra insuficiente",
                    }
                )
                continue
            metrics = bootstrap_patient_metrics(
                subset.loc[:, LABELS].to_numpy(),
                probabilities[indices],
                thresholds,
                subset.group_id.to_numpy(),
                n_bootstrap=n_bootstrap,
                seed=seed,
                mask=all_mask[indices],
            )
            for j, (label, metric) in enumerate(zip(LABELS, metrics, strict=True)):
                known = all_mask[indices, j] == 1
                positives = subset.loc[known & (subset[label].to_numpy() == 1), "group_id"].nunique()
                negatives = subset.loc[known & (subset[label].to_numpy() == 0), "group_id"].nunique()
                rows.append(
                    {
                        "attribute": column,
                        "group": str(group),
                        "patients": count,
                        "label": label,
                        "positive_patients": positives,
                        "negative_patients": negatives,
                        "status": ("ok" if min(positives, negatives) >= minimum_class_patients
                                   else "clases insuficientes para interpretación"),
                        **metric,
                    }
                )
    return pd.DataFrame(rows)


def fairness_differences(records, probabilities, thresholds, *, minimum_class_patients=10,
                         n_bootstrap=1000, seed=42, use_mask=True):
    """Diferencias exploratorias entre grupos, con bootstrap conjunto por paciente."""
    data = records.copy()
    data["age_group"] = pd.cut(data.age, [14, 17.999, 39.999, 59.999, 120],
                               labels=["15–17", "18–39", "40–59", "60+"],
                               include_lowest=True)
    y = data.loc[:, LABELS].to_numpy(dtype=int)
    mask = label_mask(data).to_numpy(dtype=bool) if use_mask else np.ones_like(y, dtype=bool)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != y.shape or not np.isfinite(probabilities).all() or \
            np.any((probabilities < 0) | (probabilities > 1)) or len(thresholds) != len(LABELS):
        raise ValueError("Probabilidades o umbrales de equidad inválidos")
    if data.empty or n_bootstrap < 1 or minimum_class_patients < 1 or data.group_id.isna().any():
        raise ValueError("Bootstrap, mínimo de pacientes o identificadores inválidos")
    clusters, _ = pd.factorize(data.group_id, sort=False)
    rng = np.random.default_rng(seed)
    n_patients = len(np.unique(clusters))
    weights = rng.multinomial(n_patients, np.full(n_patients, 1 / n_patients),
                              size=n_bootstrap)[:, clusters]
    rows = []
    for attribute in ("sex", "age_group"):
        groups = list(data.groupby(attribute, observed=True, dropna=False).indices.items())
        for (group_a, indices_a), (group_b, indices_b) in itertools.combinations(groups, 2):
            for j, label in enumerate(LABELS):
                threshold = thresholds[j]
                if threshold is not None and not 0 <= threshold <= 1:
                    raise ValueError(f"Umbral inválido para {label}")
                for metric, actual in (("sensitivity", 1), ("specificity", 0)):
                    indices = [idx[mask[idx, j] & (y[idx, j] == actual)]
                               for idx in (indices_a, indices_b)]
                    counts = [data.iloc[idx].group_id.nunique() for idx in indices]
                    row = {"attribute": attribute, "group_a": str(group_a), "group_b": str(group_b),
                           "label": label, "metric": metric, "patients_a": counts[0],
                           "patients_b": counts[1], "difference": np.nan, "low": np.nan,
                           "high": np.nan, "n_valid": 0, "n_total": n_bootstrap,
                           "status": "evidencia insuficiente"}
                    if threshold is not None and min(counts) >= minimum_class_patients:
                        rates, samples = [], []
                        correct = (probabilities[:, j] >= threshold) == bool(actual)
                        for idx in indices:
                            rates.append(float(correct[idx].mean()))
                            denominators = weights[:, idx].sum(axis=1)
                            numerators = weights[:, idx] @ correct[idx].astype(float)
                            samples.append(np.divide(numerators, denominators,
                                                     out=np.full(n_bootstrap, np.nan),
                                                     where=denominators > 0))
                        differences = samples[0] - samples[1]
                        valid = differences[np.isfinite(differences)]
                        row["difference"] = rates[0] - rates[1]
                        row["n_valid"] = len(valid)
                        if len(valid) >= max(20, int(.9 * n_bootstrap)):
                            row["low"], row["high"] = np.quantile(valid, [.025, .975])
                            row["status"] = ("diferencia observada; exploratoria" if
                                             row["low"] > 0 or row["high"] < 0 else
                                             "sin diferencia concluyente")
                    rows.append(row)
    return pd.DataFrame(rows)
