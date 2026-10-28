"""Create an active-learning campaign ledger from a CSV dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

SEQUENCE_COLUMN = "sequence"
PH_COLUMN = "pH"
SURFACE_TENSION_COLUMN = "surface_tension(mN/m)"
REQUIRED_COLUMNS = {SEQUENCE_COLUMN, PH_COLUMN, SURFACE_TENSION_COLUMN}


@dataclass(frozen=True)
class ActiveLearningOptions:
    """Configure one active-learning trajectory."""

    budget: int = 100
    initial_samples: int = 10
    seed: int = 0
    xi: float = 0.01


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


def encode_features(rows: list[tuple[str, float, float]]) -> NDArray[np.float64]:
    """Encode sequence positions and standardized pH values.

    Args:
        rows: The loaded dataset rows.

    Returns:
        A feature matrix with one-hot sequence columns and one pH column.

    """
    maximum_length = max(len(sequence) for sequence, _, _ in rows)
    categories = [
        sorted({sequence[position] if position < len(sequence) else "" for sequence, _, _ in rows})
        for position in range(maximum_length)
    ]
    category_offsets = np.cumsum([0, *(len(values) for values in categories)])
    feature_count = int(category_offsets[-1]) + 1
    features = np.zeros((len(rows), feature_count), dtype=np.float64)

    for row_index, (sequence, _, _) in enumerate(rows):
        for position, values in enumerate(categories):
            symbol = sequence[position] if position < len(sequence) else ""
            column = int(category_offsets[position]) + values.index(symbol)
            features[row_index, column] = 1.0

    ph_values = np.asarray([ph for _, ph, _ in rows], dtype=np.float64)
    ph_scale = float(np.std(ph_values))
    if np.isclose(ph_scale, 0.0):
        ph_scale = 1.0
    features[:, -1] = (ph_values - float(np.mean(ph_values))) / ph_scale
    return features


def expected_improvement(
    means: NDArray[np.float64],
    standard_deviations: NDArray[np.float64],
    best_observed: float,
    xi: float,
) -> NDArray[np.float64]:
    """Calculate expected improvement for a minimization objective.

    Args:
        means: Predicted objective means.
        standard_deviations: Predicted objective standard deviations.
        best_observed: The smallest observed objective value.
        xi: The exploration offset.

    Returns:
        Expected-improvement values.

    """
    improvement = best_observed - means - xi
    scores = np.zeros_like(means)
    positive_variance = standard_deviations > 0.0
    z_scores = improvement[positive_variance] / standard_deviations[positive_variance]
    normal_density = np.exp(-(z_scores**2) / 2.0) / np.sqrt(2.0 * np.pi)
    scores[positive_variance] = (
        improvement[positive_variance] * ndtr(z_scores)
        + standard_deviations[positive_variance] * normal_density
    )
    return scores


def select_indices(
    rows: list[tuple[str, float, float]],
    *,
    options: ActiveLearningOptions,
) -> tuple[list[int], list[float | None]]:
    """Select initial random points and sequential active-learning points.

    Args:
        rows: The loaded dataset rows.
        options: The trajectory configuration.

    Returns:
        Selected indices and acquisition scores for each selection.

    Raises:
        ValueError: If a command option is not valid.

    """
    if options.budget < 1:
        raise ValueError("Budget must be at least 1")
    if options.budget > len(rows):
        raise ValueError(
            f"Budget {options.budget} exceeds the dataset size of {len(rows)}",
        )
    if options.initial_samples < 1:
        raise ValueError("Initial samples must be at least 1")
    if options.initial_samples > options.budget:
        raise ValueError("Initial samples cannot exceed the budget")
    if options.xi < 0.0:
        raise ValueError("xi cannot be negative")

    generator = np.random.default_rng(options.seed)
    initial = generator.choice(
        len(rows),
        size=options.initial_samples,
        replace=False,
    )
    selected = [int(index) for index in initial]
    acquisition_scores: list[float | None] = [None] * options.initial_samples
    features = encode_features(rows)
    objectives = np.asarray([value for _, _, value in rows], dtype=np.float64)

    while len(selected) < options.budget:
        observed = np.asarray(selected, dtype=np.int64)
        remaining = np.asarray(
            [index for index in range(len(rows)) if index not in selected],
            dtype=np.int64,
        )
        model = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            random_state=options.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(features[observed], objectives[observed])
        means, deviations = model.predict(features[remaining], return_std=True)
        scores = expected_improvement(
            means,
            deviations,
            float(np.min(objectives[observed])),
            options.xi,
        )
        choice_position = int(np.argmax(scores))
        selected.append(int(remaining[choice_position]))
        acquisition_scores.append(float(scores[choice_position]))

    return selected, acquisition_scores


def make_records(
    rows: list[tuple[str, float, float]],
    indices: list[int],
    acquisition_scores: list[float | None],
    *,
    options: ActiveLearningOptions,
) -> list[dict[str, object]]:
    """Build campaign-compatible observation records."""
    records: list[dict[str, object]] = []
    for step, (index, score) in enumerate(
        zip(indices, acquisition_scores, strict=True),
        start=1,
    ):
        sequence, ph, surface_tension = rows[index]
        if step <= options.initial_samples:
            reasoning = (
                "Uniform random initialization; "
                f"step {step} of {options.initial_samples}, seed {options.seed}."
            )
        else:
            reasoning = (
                "Sequential Bayesian optimization with a Matern-5/2 Gaussian "
                f"process; campaign step {step}, minimization expected improvement "
                f"{score:.12g}, xi {options.xi}."
            )
        records.append(
            {
                "request": {"sequence": sequence, "ph": ph},
                "reasoning": reasoning,
                "result": {
                    "found": True,
                    "values": {"surface_tension": surface_tension},
                },
            },
        )
    return records


def write_jsonl_atomic(path: Path, records: list[dict[str, object]]) -> None:
    """Replace a JSONL file atomically."""
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


def run(
    dataset: Path,
    output: Path,
    *,
    options: ActiveLearningOptions,
) -> list[dict[str, object]]:
    """Run one active-learning campaign."""
    rows = load_dataset(dataset)
    indices, scores = select_indices(
        rows,
        options=options,
    )
    records = make_records(
        rows,
        indices,
        scores,
        options=options,
    )
    write_jsonl_atomic(output, records)
    return records


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


def main(argv: Sequence[str] | None = None) -> None:
    """Run the active-learning command-line interface."""
    parser = argparse.ArgumentParser(
        description="Optimize surface tension with sequential Bayesian optimization.",
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--initial-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--xi", type=float, default=0.01)
    args = parser.parse_args(argv)
    try:
        options = ActiveLearningOptions(
            budget=args.budget,
            initial_samples=args.initial_samples,
            seed=args.seed,
            xi=args.xi,
        )
        records = run(args.dataset, args.output, options=options)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    _print_summary(records)


if __name__ == "__main__":
    main()
