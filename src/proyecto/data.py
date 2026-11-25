"""CSV/image auditing, multilabel preparation, and leakage-safe data splits.

The public entry points are :func:`load_dataset`, :func:`audit_dataset`,
:func:`grouped_train_test_split`, and :func:`grouped_multilabel_kfold`.
Patient identifiers are used internally to form split groups, but are never
included in returned records or audit summaries.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABELS = ("pneumonia", "pleural_effusion", "tuberculosis", "edema")
_REQUIRED_COLUMNS = ("file_name", "edad", "diagnosticos", "anony_id")
_DIAGNOSIS_LABELS = {
    "neumonia": "pneumonia",
    "pneumonia": "pneumonia",
    "derrame pleural": "pleural_effusion",
    "pleural effusion": "pleural_effusion",
    "effusion": "pleural_effusion",
    "tuberculosis": "tuberculosis",
    "edema": "edema",
    "edema pulmonar": "edema",
    "pulmonary edema": "edema",
}
_HEALTHY_LABELS = {"sano", "saludable", "healthy", "normal"}
_OTHER_PATHOLOGIES = {"cardiomegalia", "atelectasia", "fibrosis pulmonar", "enfisema"}
_HASH_BUFFER_SIZE = 1024 * 1024


class DatasetError(ValueError):
    """Raised when the CSV or image directory cannot be used as a dataset."""


class SplitError(ValueError):
    """Raised when a leakage-safe split cannot be constructed."""


@dataclass(frozen=True)
class Dataset:
    """Loaded records, rejected records, and aggregate audit information.

    ``records`` contains one row per valid CSV record and four integer
    multilabel columns named by :data:`LABELS`. ``invalid_records`` contains
    the source row, image filename (when usable), and explicit reason codes.
    Raw patient identifiers are deliberately omitted from both tables.
    """

    records: pd.DataFrame
    invalid_records: pd.DataFrame
    audit: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_HASH_BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_diagnosis(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    unaccented = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(unaccented.casefold().strip().split())


def _parse_diagnoses(value: str) -> tuple[bool, tuple[int, ...]]:
    entries = [part.strip() for part in value.replace(";", "|").split("|")]
    if not value.strip() or any(not part for part in entries):
        raise ValueError("missing_diagnosis")

    healthy = False
    other_pathology = False
    labels = set()
    for entry in entries:
        normalized = _normalize_diagnosis(entry)
        if normalized in _HEALTHY_LABELS:
            healthy = True
        elif normalized in _DIAGNOSIS_LABELS:
            labels.add(_DIAGNOSIS_LABELS[normalized])
        elif normalized in _OTHER_PATHOLOGIES:
            other_pathology = True
        else:
            raise ValueError("unsupported_diagnosis")
    if healthy and other_pathology:
        raise ValueError("healthy_with_pathology")
    return healthy, tuple(int(label in labels) for label in LABELS)


def _has_other_pathology(value: str) -> bool:
    """True si el diagnóstico menciona hallazgos ajenos a las cuatro etiquetas."""
    entries = (_normalize_diagnosis(part) for part in value.replace(";", "|").split("|"))
    return any(entry in _OTHER_PATHOLOGIES for entry in entries)


def label_mask(records: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una máscara (1 = etiqueta confirmada) con las columnas de LABELS.

    Un cero solo cuenta como negativo confirmado si el registro no menciona otras
    patologías: el diagnóstico de esos registros puede estar incompleto, así que
    sus etiquetas ausentes se tratan como desconocidas. Los positivos siempre son
    confirmados.
    """
    if "other_pathology" in records:
        other = records["other_pathology"].to_numpy(dtype=bool)[:, None]
    else:
        other = np.zeros((len(records), 1), dtype=bool)
    labels = records.loc[:, list(LABELS)].to_numpy(dtype=int)
    return pd.DataFrame(((labels == 1) | ~other).astype(int), columns=list(LABELS), index=records.index)


def _image_filename(value: str) -> str:
    filename = value.strip()
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
    ):
        raise ValueError("invalid_file_name")
    if Path(filename).suffix.casefold() != ".png":
        filename += ".png"
    return filename


def _age_value(value: str) -> float:
    try:
        # Decimal accepts the normal CSV decimal form and avoids accepting NaN
        # or infinities as plausible ages.
        decimal_age = Decimal(value.strip().replace(",", "."))
    except (InvalidOperation, ValueError):
        raise ValueError("invalid_age") from None
    if not decimal_age.is_finite() or decimal_age < 0 or decimal_age > 120:
        raise ValueError("invalid_age")
    return float(decimal_age)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def load_dataset(
    csv_path: str | Path, image_dir: str | Path, *, encoding: str = "utf-8-sig"
) -> Dataset:
    """Audit and load a CSV and its PNG images.

    Expected CSV columns are ``file_name``, ``edad``, ``diagnosticos`` and
    ``anony_id``. A missing ``.png`` suffix is added to ``file_name``. Valid
    rows receive numeric ages, four binary label columns, exact image SHA-256
    hashes, and patient-component ``group_id`` values. Invalid rows are
    omitted from ``records`` and retained with explicit reason codes.

    All regular ``.png`` files immediately inside ``image_dir`` are hashed so
    the audit also reports exact duplicate files and unreferenced images.
    Patient identifiers and image filenames are not emitted in audit details.

    Raises:
        DatasetError: If the CSV is unreadable, malformed, or lacks required
            columns, or if ``image_dir`` is not a directory.
    """
    csv_file = Path(csv_path)
    images_path = Path(image_dir)
    if not images_path.is_dir():
        raise DatasetError("image_dir must be an existing directory")
    try:
        csv_bytes = csv_file.read_bytes()
        text = csv_bytes.decode(encoding)
    except (OSError, UnicodeError, LookupError):
        raise DatasetError("CSV file could not be read with the requested encoding") from None

    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fieldnames = reader.fieldnames or []
        if len(set(fieldnames)) != len(fieldnames):
            raise DatasetError("CSV contains duplicate column names")
        missing_columns = [name for name in _REQUIRED_COLUMNS if name not in fieldnames]
        if missing_columns:
            raise DatasetError(
                "CSV is missing required columns: " + ", ".join(missing_columns)
            )
        source_rows = list(reader)
    except csv.Error:
        raise DatasetError("CSV could not be parsed") from None

    image_hashes: dict[str, str] = {}
    unreadable_images: set[str] = set()
    png_paths = sorted(
        (
            path
            for path in images_path.iterdir()
            if path.is_file() and path.suffix.casefold() == ".png"
        ),
        key=lambda path: path.name,
    )
    for image_path in png_paths:
        try:
            image_hashes[image_path.name] = _sha256_file(image_path)
        except OSError:
            unreadable_images.add(image_path.name)

    image_rows: list[dict[str, Any]] = []
    duplicate_signatures: dict[str, set[tuple[bool, tuple[int, ...]]]] = defaultdict(set)
    duplicate_rows: dict[str, list[int]] = defaultdict(list)
    patient_ids_by_hash: dict[str, list[str]] = defaultdict(list)
    referenced_filenames: set[str] = set()
    missing_image_records = 0

    for source_row, raw_row in enumerate(source_rows, start=2):
        reasons: list[str] = []
        if None in raw_row:
            reasons.append("malformed_csv_row")

        patient_id = (raw_row.get("anony_id") or "").strip()
        if not patient_id:
            reasons.append("missing_patient_id")

        raw_filename = raw_row.get("file_name") or ""
        filename: str | None = None
        image_hash: str | None = None
        try:
            filename = _image_filename(raw_filename)
            referenced_filenames.add(filename)
            if filename in unreadable_images:
                reasons.append("unreadable_image")
            elif filename not in image_hashes:
                reasons.append("missing_image")
                missing_image_records += 1
        except ValueError as error:
            reasons.append(str(error))

        age: float | None = None
        try:
            age = _age_value(raw_row.get("edad") or "")
        except ValueError as error:
            reasons.append(str(error))

        diagnosis = raw_row.get("diagnosticos") or ""
        label_signature: tuple[bool, tuple[int, ...]] | None = None
        labels: tuple[int, ...] | None = None
        try:
            label_signature = _parse_diagnoses(diagnosis)
            healthy, labels = label_signature
            if healthy and any(labels):
                reasons.append("healthy_with_pathology")
        except ValueError as error:
            reasons.append(str(error))

        if filename in image_hashes:
            image_hash = image_hashes[filename]
            duplicate_rows[image_hash].append(len(image_rows))
            if label_signature is not None:
                duplicate_signatures[image_hash].add(label_signature)
            if patient_id:
                patient_ids_by_hash[image_hash].append(patient_id)

        image_rows.append(
            {
                "source_row": source_row,
                "file_name": filename or raw_filename.strip(),
                "image_path": str(images_path / filename) if filename else None,
                "image_sha256": image_hash,
                "age": age,
                "sex": (raw_row.get("sexo") or "").strip() or None,
                "labels": labels,
                "other_pathology": _has_other_pathology(diagnosis),
                "patient_id": patient_id or None,
                "reasons": reasons,
            }
        )

    conflicting_hashes = {
        image_hash
        for image_hash, signatures in duplicate_signatures.items()
        if len(signatures) > 1
    }
    for image_hash in conflicting_hashes:
        for row_index in duplicate_rows[image_hash]:
            reasons = image_rows[row_index]["reasons"]
            if "duplicate_image_label_conflict" not in reasons:
                reasons.append("duplicate_image_label_conflict")

    patient_groups = _UnionFind()
    for row in image_rows:
        patient_id = row["patient_id"]
        if patient_id:
            patient_groups.add(patient_id)
    for patient_ids in patient_ids_by_hash.values():
        first_patient = patient_ids[0]
        for patient_id in patient_ids[1:]:
            patient_groups.union(first_patient, patient_id)

    component_first_row: dict[str, int] = {}
    for row_index, row in enumerate(image_rows):
        patient_id = row["patient_id"]
        if patient_id:
            root = patient_groups.find(patient_id)
            component_first_row[root] = min(
                component_first_row.get(root, row_index), row_index
            )
    ordered_roots = sorted(component_first_row, key=component_first_row.get)
    group_ids = {root: f"group_{index:06d}" for index, root in enumerate(ordered_roots, 1)}

    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for row in image_rows:
        if row["reasons"]:
            invalid_rows.append(
                {
                    "source_row": row["source_row"],
                    "file_name": row["file_name"],
                    "reason_codes": tuple(dict.fromkeys(row["reasons"])),
                    "reasons": ";".join(dict.fromkeys(row["reasons"])),
                }
            )
            continue
        patient_id = row["patient_id"]
        root = patient_groups.find(patient_id)
        record = {
            "source_row": row["source_row"],
            "file_name": row["file_name"],
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "age": row["age"],
            "sex": row["sex"],
            "group_id": group_ids[root],
            "other_pathology": row["other_pathology"],
        }
        record.update(dict(zip(LABELS, row["labels"], strict=True)))
        valid_rows.append(record)

    record_columns = [
        "source_row",
        "file_name",
        "image_path",
        "image_sha256",
        "age",
        "sex",
        "group_id",
        "other_pathology",
        *LABELS,
    ]
    invalid_columns = ["source_row", "file_name", "reason_codes", "reasons"]
    records = pd.DataFrame(valid_rows, columns=record_columns)
    invalid_records = pd.DataFrame(invalid_rows, columns=invalid_columns)

    image_hash_counts = Counter(image_hashes.values())
    duplicate_image_groups = [
        {"sha256": image_hash, "file_count": count}
        for image_hash, count in sorted(image_hash_counts.items())
        if count > 1
    ]
    reason_counts = Counter(
        reason
        for row in invalid_rows
        for reason in row["reason_codes"]
    )
    duplicate_record_count = sum(
        max(0, len(row_indices) - 1) for row_indices in duplicate_rows.values()
    )
    audit = {
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "csv_rows": len(source_rows),
        "valid_records": len(valid_rows),
        "invalid_records": len(invalid_rows),
        "invalid_reason_counts": dict(sorted(reason_counts.items())),
        "image_files": len(png_paths),
        "hashed_image_files": len(image_hashes),
        "unreadable_image_files": len(unreadable_images),
        "missing_image_records": missing_image_records,
        "unreferenced_image_files": sum(
            path.name not in referenced_filenames for path in png_paths
        ),
        "duplicate_image_groups": duplicate_image_groups,
        "duplicate_image_file_count": sum(
            count - 1 for count in image_hash_counts.values() if count > 1
        ),
        "duplicate_image_record_count": duplicate_record_count,
        "conflicting_duplicate_image_groups": len(conflicting_hashes),
        "conflicting_duplicate_image_records": sum(
            len(duplicate_rows[image_hash]) for image_hash in conflicting_hashes
        ),
    }
    return Dataset(records=records, invalid_records=invalid_records, audit=audit)


def audit_dataset(csv_path: str | Path, image_dir: str | Path) -> dict[str, Any]:
    """Return aggregate CSV/image audit counts and exact SHA-256 summaries.

    The returned dictionary contains no patient identifiers or image
    filenames. Use :func:`load_dataset` when the validated records are needed.
    """
    return load_dataset(csv_path, image_dir).audit


def _group_vectors(records: pd.DataFrame) -> tuple[list[str], list[int], list[list[int]]]:
    required = {"group_id", *LABELS}
    missing = required.difference(records.columns)
    if missing:
        raise SplitError("records are missing split columns: " + ", ".join(sorted(missing)))
    if records.empty:
        raise SplitError("cannot split an empty dataset")

    group_order = list(dict.fromkeys(records["group_id"].tolist()))
    if any(pd.isna(group) for group in group_order):
        raise SplitError("group_id values must not be missing")
    group_sizes: list[int] = []
    group_labels: list[list[int]] = []
    for group_id in group_order:
        rows = records.loc[records["group_id"] == group_id]
        group_sizes.append(len(rows))
        group_labels.append(
            [int(rows[label].fillna(0).astype(int).max()) for label in LABELS]
        )
    return group_order, group_sizes, group_labels


def _assign_groups(
    group_ids: list[str],
    group_sizes: list[int],
    group_labels: list[list[int]],
    fractions: list[float],
    random_state: int | None,
) -> list[list[str]]:
    group_count = len(group_ids)
    split_count = len(fractions)
    if group_count < split_count:
        raise SplitError(
            f"at least {split_count} independent patient groups are required"
        )

    rng = random.Random(random_state)
    tie_breakers = [rng.random() for _ in group_ids]
    label_totals = [
        sum(labels[label_index] for labels in group_labels)
        for label_index in range(len(LABELS))
    ]
    rarity = [
        sum(
            value / label_totals[label_index]
            for label_index, value in enumerate(labels)
            if label_totals[label_index]
        )
        for labels in group_labels
    ]
    order = sorted(
        range(group_count),
        key=lambda index: (
            -rarity[index],
            -group_sizes[index],
            tie_breakers[index],
        ),
    )

    target_sizes = [fraction * sum(group_sizes) for fraction in fractions]
    target_labels = [
        [fraction * total for total in label_totals] for fraction in fractions
    ]
    assigned: list[list[str]] = [[] for _ in fractions]
    current_sizes = [0 for _ in fractions]
    current_labels = [[0 for _ in LABELS] for _ in fractions]

    def cost() -> float:
        total_cost = 0.0
        for split_index in range(split_count):
            size_target = max(target_sizes[split_index], 1.0)
            total_cost += (
                (current_sizes[split_index] - target_sizes[split_index]) ** 2 / size_target
            )
            for label_index in range(len(LABELS)):
                label_target = max(target_labels[split_index][label_index], 1.0)
                total_cost += (
                    (current_labels[split_index][label_index]
                     - target_labels[split_index][label_index]) ** 2
                    / label_target / len(LABELS)
                )
        return total_cost

    for position, group_index in enumerate(order):
        empty_splits = [
            split_index
            for split_index, group_list in enumerate(assigned)
            if not group_list
        ]
        remaining_groups = group_count - position
        candidate_splits = (
            empty_splits if remaining_groups == len(empty_splits) else range(split_count)
        )
        options: list[tuple[float, float, int]] = []
        for split_index in candidate_splits:
            current_sizes[split_index] += group_sizes[group_index]
            for label_index, value in enumerate(group_labels[group_index]):
                current_labels[split_index][label_index] += value
            score = cost()
            current_sizes[split_index] -= group_sizes[group_index]
            for label_index, value in enumerate(group_labels[group_index]):
                current_labels[split_index][label_index] -= value
            options.append((score, rng.random(), split_index))
        selected = min(options)[2]
        assigned[selected].append(group_ids[group_index])
        current_sizes[selected] += group_sizes[group_index]
        for label_index, value in enumerate(group_labels[group_index]):
            current_labels[selected][label_index] += value
    return assigned


def grouped_train_test_split(
    dataset: Dataset, *, test_size: float = 0.2, random_state: int | None = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/test records without separating linked patient groups.

    ``test_size`` is a fraction of records (not groups); indivisible groups
    can make the realized fraction differ slightly. The greedy assignment
    balances group-level multilabel prevalence and record counts.
    """
    if not math.isfinite(test_size) or not 0 < test_size < 1:
        raise SplitError("test_size must be greater than 0 and less than 1")
    records = dataset.records.reset_index(drop=True)
    group_ids, group_sizes, group_labels = _group_vectors(records)
    if len(group_ids) < 2:
        raise SplitError("at least two independent patient groups are required")
    train_groups, test_groups = _assign_groups(
        group_ids, group_sizes, group_labels, [1 - test_size, test_size], random_state
    )
    train_set, test_set = set(train_groups), set(test_groups)
    return (
        records.loc[records["group_id"].isin(train_set)].reset_index(drop=True),
        records.loc[records["group_id"].isin(test_set)].reset_index(drop=True),
    )


def grouped_multilabel_kfold(
    dataset: Dataset, *, n_splits: int = 5, random_state: int | None = 42
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create group-disjoint multilabel cross-validation folds.

    Group-level label frequencies and fold sizes are balanced greedily. The
    function fails explicitly if there are too few patient groups, or a
    present label occurs in fewer independent groups than ``n_splits``; in
    that case per-fold multilabel stratification is not feasible.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise SplitError("n_splits must be an integer of at least 2")
    records = dataset.records.reset_index(drop=True)
    group_ids, group_sizes, group_labels = _group_vectors(records)
    if len(group_ids) < n_splits:
        raise SplitError(
            f"n_splits={n_splits} exceeds the {len(group_ids)} independent patient groups"
        )
    for label_index, label in enumerate(LABELS):
        positive_groups = sum(labels[label_index] for labels in group_labels)
        if 0 < positive_groups < n_splits:
            raise SplitError(
                f"label {label!r} occurs in only {positive_groups} patient groups; "
                f"at least n_splits={n_splits} are required"
            )

    fold_groups = _assign_groups(
        group_ids,
        group_sizes,
        group_labels,
        [1 / n_splits] * n_splits,
        random_state,
    )
    all_groups = set(group_ids)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for test_group_list in fold_groups:
        test_groups = set(test_group_list)
        train_groups = all_groups - test_groups
        train = records.loc[records["group_id"].isin(train_groups)].reset_index(drop=True)
        test = records.loc[records["group_id"].isin(test_groups)].reset_index(drop=True)
        folds.append((train, test))
    return folds
