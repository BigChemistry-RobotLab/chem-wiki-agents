"""Load and query a typed CSV dataset for the simulated lab."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

type JsonScalar = str | int | float | bool | None
type ColumnType = Literal["string", "integer", "number", "boolean"]

_INTEGER_PATTERN = re.compile(r"[+-]?\d+")
_COLUMN_TYPES = frozenset({"string", "integer", "number", "boolean"})


@dataclass(frozen=True)
class ColumnDefinition:
    """Describe one agent-facing CSV column."""

    name: str
    type: ColumnType

    def as_json(self) -> dict[str, str]:
        """Return the public schema form."""
        return {"name": self.name, "type": self.type}


@dataclass(frozen=True)
class SimulatedLab:
    """Store typed experiment rows and material records."""

    inputs: tuple[ColumnDefinition, ...]
    outputs: tuple[ColumnDefinition, ...]
    materials: tuple[ColumnDefinition, ...]
    material_records: tuple[dict[str, JsonScalar], ...]
    experiments: dict[tuple[JsonScalar, ...], dict[str, JsonScalar]]
    dataset_rows: int
    max_batch_size: int

    def schema(self) -> dict[str, object]:
        """Return the runtime API schema."""
        return {
            "inputs": [item.as_json() for item in self.inputs],
            "outputs": [item.as_json() for item in self.outputs],
            "materials": [item.as_json() for item in self.materials],
            "required_inputs": [item.name for item in self.inputs],
            "max_batch_size": self.max_batch_size,
        }


def load_lab(  # ruff: ignore[complex-structure, too-many-branches, too-many-locals, too-many-statements, too-many-arguments]
    dataset_path: str | Path,
    *,
    inputs: list[str],
    outputs: list[str],
    aliases: dict[str, str] | None = None,
    column_types: dict[str, str] | None = None,
    materials_path: str | Path | None = None,
    material_fields: list[str] | None = None,
    max_batch_size: int = 100,
) -> SimulatedLab:
    """Load and validate one lab configuration."""
    aliases = aliases or {}
    column_types = column_types or {}
    material_fields = material_fields or []
    if not inputs:
        raise ValueError("At least one input column is required")
    if not outputs:
        raise ValueError("At least one output column is required")
    if max_batch_size < 1:
        raise ValueError("Maximum batch size must be at least 1")
    _reject_duplicates(inputs, "input")
    _reject_duplicates(outputs, "output")
    _reject_duplicates(material_fields, "material")

    dataset_headers, raw_rows = _read_csv(dataset_path)
    _require_columns(dataset_headers, [*inputs, *outputs], "dataset")

    material_headers: list[str] = []
    raw_material_rows: list[dict[str, str]] = []
    if materials_path is not None:
        if not material_fields:
            raise ValueError(
                "At least one material field is required with a materials CSV",
            )
        material_headers, raw_material_rows = _read_csv(materials_path)
        _require_columns(material_headers, material_fields, "materials CSV")
    elif material_fields:
        raise ValueError("Material fields require a materials CSV")
    else:
        material_fields = list(inputs)

    selected_columns = set(inputs) | set(outputs) | set(material_fields)
    _validate_aliases(aliases, selected_columns)
    _validate_type_overrides(column_types, selected_columns)
    exposed_names = {source: source for source in selected_columns}
    exposed_names.update({source: agent_name for agent_name, source in aliases.items()})
    _reject_exposed_name_collisions(exposed_names)

    dataset_types = _resolve_types(raw_rows, set(inputs) | set(outputs), column_types)
    typed_rows = [
        _convert_row(
            row, set(inputs) | set(outputs), dataset_types, allow_empty=set(outputs),
        )
        for row in raw_rows
    ]

    experiments: dict[tuple[JsonScalar, ...], dict[str, JsonScalar]] = {}
    for row_number, row in enumerate(typed_rows, start=2):
        missing = [name for name in inputs if row[name] is None]
        if missing:
            columns = ", ".join(missing)
            raise ValueError(
                f"Dataset row {row_number} has missing input values: {columns}",
            )
        key = tuple(row[name] for name in inputs)
        if key in experiments:
            raise ValueError(
                f"Dataset contains a duplicate input key at row {row_number}: {key!r}",
            )
        experiments[key] = {exposed_names[name]: row[name] for name in outputs}

    if materials_path is None:
        material_types = {name: dataset_types[name] for name in material_fields}
        material_source_rows = typed_rows
    else:
        material_types = _resolve_types(
            raw_material_rows, set(material_fields), column_types,
        )
        material_source_rows = [
            _convert_row(row, set(material_fields), material_types, allow_empty=set())
            for row in raw_material_rows
        ]
        for row_number, row in enumerate(material_source_rows, start=2):
            missing = [name for name in material_fields if row[name] is None]
            if missing:
                columns = ", ".join(missing)
                raise ValueError(
                    f"Materials CSV row {row_number} has missing values: {columns}",
                )

    records: list[dict[str, JsonScalar]] = []
    seen_records: set[tuple[JsonScalar, ...]] = set()
    for row in material_source_rows:
        key = tuple(row[name] for name in material_fields)
        if key in seen_records:
            continue
        seen_records.add(key)
        records.append({exposed_names[name]: row[name] for name in material_fields})

    return SimulatedLab(
        inputs=tuple(
            ColumnDefinition(exposed_names[name], dataset_types[name])
            for name in inputs
        ),
        outputs=tuple(
            ColumnDefinition(exposed_names[name], dataset_types[name])
            for name in outputs
        ),
        materials=tuple(
            ColumnDefinition(exposed_names[name], material_types[name])
            for name in material_fields
        ),
        material_records=tuple(records),
        experiments=experiments,
        dataset_rows=len(raw_rows),
        max_batch_size=max_batch_size,
    )


def _read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV and reject invalid or duplicate headers."""
    csv_path = Path(path)
    try:
        handle = csv_path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"Could not read CSV {csv_path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if any(not header for header in headers):
            raise ValueError(f"CSV has an empty column name: {csv_path}")
        _reject_duplicates(list(headers), "CSV column")
        return list(headers), [dict(row) for row in reader]


def _reject_duplicates(values: list[str], label: str) -> None:
    """Reject duplicate configured values."""
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"Duplicate {label} column: {value}")
        seen.add(value)


def _require_columns(headers: list[str], columns: list[str], label: str) -> None:
    """Require all selected columns in a CSV header."""
    missing = [column for column in columns if column not in headers]
    if missing:
        raise ValueError(f"Unknown {label} columns: {', '.join(missing)}")


def _validate_aliases(aliases: dict[str, str], selected: set[str]) -> None:
    """Validate agent-to-CSV alias mappings."""
    sources: set[str] = set()
    for agent_name, source in aliases.items():
        if not agent_name:
            raise ValueError("Alias names cannot be empty")
        if source not in selected:
            raise ValueError(f"Alias targets an unselected CSV column: {source}")
        if source in sources:
            raise ValueError(f"CSV column has more than one alias: {source}")
        sources.add(source)


def _reject_exposed_name_collisions(names: dict[str, str]) -> None:
    """Reject aliases that make two columns share one public name."""
    reverse: dict[str, str] = {}
    for source, exposed in names.items():
        previous = reverse.get(exposed)
        if previous is not None and previous != source:
            raise ValueError(
                f"Agent-facing column name {exposed!r} maps to both {previous!r} and {source!r}",
            )
        reverse[exposed] = source


def _validate_type_overrides(overrides: dict[str, str], headers: set[str]) -> None:
    """Validate explicit column type mappings."""
    for column, type_name in overrides.items():
        if column not in headers:
            raise ValueError(f"Type override targets an unknown CSV column: {column}")
        if type_name not in _COLUMN_TYPES:
            choices = ", ".join(sorted(_COLUMN_TYPES))
            raise ValueError(
                f"Invalid type {type_name!r} for {column}; use one of: {choices}",
            )


def _resolve_types(
    rows: list[dict[str, str]],
    columns: set[str],
    overrides: dict[str, str],
) -> dict[str, ColumnType]:
    """Infer selected column types and apply explicit overrides."""
    resolved: dict[str, ColumnType] = {}
    for column in columns:
        override = overrides.get(column)
        if override is not None:
            resolved[column] = cast("ColumnType", override)
            continue
        values = [row[column].strip() for row in rows if row[column].strip()]
        if values and all(value.casefold() in {"true", "false"} for value in values):
            resolved[column] = "boolean"
        elif values and all(_INTEGER_PATTERN.fullmatch(value) for value in values):
            resolved[column] = "integer"
        elif values and all(_is_finite_number(value) for value in values):
            resolved[column] = "number"
        else:
            resolved[column] = "string"
    return resolved


def _is_finite_number(value: str) -> bool:
    """Return whether text is a finite floating-point number."""
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def _convert_row(  # ruff: ignore[complex-structure]
    row: dict[str, str],
    columns: set[str],
    types: dict[str, ColumnType],
    *,
    allow_empty: set[str],
) -> dict[str, JsonScalar]:
    """Convert selected values to their configured JSON scalar types."""
    converted: dict[str, JsonScalar] = {}
    for column in columns:
        raw = row[column]
        if not raw.strip():
            converted[column] = None
            if column not in allow_empty:
                continue
        elif types[column] == "string":
            converted[column] = raw
        elif types[column] == "integer":
            try:
                converted[column] = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer value {raw!r} in column {column}",
                ) from exc
        elif types[column] == "number":
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid number value {raw!r} in column {column}",
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Invalid non-finite number {raw!r} in column {column}",
                )
            converted[column] = value
        else:
            folded = raw.strip().casefold()
            if folded not in {"true", "false"}:
                raise ValueError(f"Invalid boolean value {raw!r} in column {column}")
            converted[column] = folded == "true"
    return converted
