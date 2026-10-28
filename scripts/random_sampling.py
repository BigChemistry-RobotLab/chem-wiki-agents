"""Create a random-sampling campaign ledger from a CSV dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

SEQUENCE_COLUMN = "sequence"
PH_COLUMN = "pH"
SURFACE_TENSION_COLUMN = "surface_tension(mN/m)"
REQUIRED_COLUMNS = {SEQUENCE_COLUMN, PH_COLUMN, SURFACE_TENSION_COLUMN}


def load_dataset(path: Path) -> list[tuple[str, float, float]]:
    """Load only the campaign inputs and objective from a CSV file.

    Args:
        path: The source CSV path.

    Returns:
        Tuples that contain sequence, pH, and surface tension.

    Raises:
        ValueError: If columns or row values are invalid.

    """
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Dataset is missing required columns: {names}")

        rows: list[tuple[str, float, float]] = []
        for line_number, row in enumerate(reader, start=2):
            sequence = row[SEQUENCE_COLUMN].strip()
            if not sequence:
                raise ValueError(f"Sequence is empty on CSV line {line_number}")
            try:
                ph = float(row[PH_COLUMN])
                surface_tension = float(row[SURFACE_TENSION_COLUMN])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Dataset has a nonnumeric value on CSV line {line_number}",
                ) from exc
            rows.append((sequence, ph, surface_tension))

    if not rows:
        raise ValueError("Dataset has no data rows")
    return rows


def select_indices(row_count: int, budget: int, seed: int) -> list[int]:
    """Select row indices uniformly without replacement.

    Args:
        row_count: The number of available rows.
        budget: The number of rows to select.
        seed: The random seed.

    Returns:
        Selected row indices in sampling order.

    Raises:
        ValueError: If the budget is not valid.

    """
    if budget < 1:
        raise ValueError("Budget must be at least 1")
    if budget > row_count:
        raise ValueError(
            f"Budget {budget} exceeds the dataset size of {row_count}",
        )
    return random.Random(seed).sample(  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        range(row_count),
        budget,
    )


def make_records(
    rows: list[tuple[str, float, float]],
    indices: list[int],
    *,
    seed: int,
) -> list[dict[str, object]]:
    """Build campaign-compatible observation records.

    Args:
        rows: The loaded dataset rows.
        indices: The selected row indices.
        seed: The random seed used for selection.

    Returns:
        JSON-compatible observation records.

    """
    records: list[dict[str, object]] = []
    for step, index in enumerate(indices, start=1):
        sequence, ph, surface_tension = rows[index]
        records.append(
            {
                "request": {"sequence": sequence, "ph": ph},
                "reasoning": (
                    f"Uniform random selection without replacement; step {step} "
                    f"of {len(indices)}, seed {seed}."
                ),
                "result": {
                    "found": True,
                    "values": {"surface_tension": surface_tension},
                },
            },
        )
    return records


def write_jsonl_atomic(path: Path, records: list[dict[str, object]]) -> None:
    """Replace a JSONL file atomically.

    Args:
        path: The output path.
        records: The records to write.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            encoding="utf-8",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for record in records:
                output.write(json.dumps(record, sort_keys=True))
                output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        Path(temporary_path).replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def run(dataset: Path, output: Path, *, budget: int, seed: int) -> list[dict[str, object]]:
    """Run one random-sampling campaign.

    Args:
        dataset: The source CSV path.
        output: The destination JSONL path.
        budget: The number of requests.
        seed: The random seed.

    Returns:
        The records written to the ledger.

    """
    rows = load_dataset(dataset)
    indices = select_indices(len(rows), budget, seed)
    records = make_records(rows, indices, seed=seed)
    write_jsonl_atomic(output, records)
    return records


def _print_summary(records: list[dict[str, object]]) -> None:
    """Print the campaign size and best result."""
    best = min(records, key=_record_value)
    request = best["request"]
    if not isinstance(request, dict):
        raise TypeError("Internal request must be an object")
    request_data = cast("dict[str, object]", request)
    sys.stdout.write(
        f"Sampled {len(records)} points.\n"
        "Best result: "
        f"({request_data['sequence']}, {request_data['ph']}, {_record_value(best)})\n",
    )


def _record_value(record: dict[str, object]) -> float:
    """Get a surface-tension value from an internal record."""
    result = record["result"]
    if not isinstance(result, dict):
        raise TypeError("Internal result must be an object")
    result_data = cast("dict[str, object]", result)
    values = result_data["values"]
    if not isinstance(values, dict):
        raise TypeError("Internal values must be an object")
    values_data = cast("dict[str, object]", values)
    value = values_data["surface_tension"]
    if not isinstance(value, int | float):
        raise TypeError("Internal surface tension must be numeric")
    return float(value)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the random-sampling command-line interface."""
    parser = argparse.ArgumentParser(
        description="Sample a surface-tension dataset uniformly without replacement.",
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        records = run(args.dataset, args.output, budget=args.budget, seed=args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    _print_summary(records)


if __name__ == "__main__":
    main()
