"""Tests for the random and active-learning baseline scripts."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest


def _load_script(name: str) -> Any:
    """Load a repository script as a test module."""
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return cast("Any", module)


active_learning = _load_script("active_learning")
calculate_hits = _load_script("calculate_hits")
random_sampling = _load_script("random_sampling")


def _write_dataset(path: Path, row_count: int = 12) -> None:
    """Write a small surface-tension dataset."""
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "sequence",
                "pH",
                "concentration(mM)",
                "surface_tension(mN/m)",
            ],
        )
        writer.writeheader()
        for index in range(row_count):
            writer.writerow(
                {
                    "sequence": f"A{alphabet[index]}D",
                    "pH": (3.0, 7.0, 11.0)[index % 3],
                    "concentration(mM)": 1000 + index,
                    "surface_tension(mN/m)": 70.0 - index,
                },
            )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read JSON objects from a test ledger."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_ledger_shape(path: Path, expected_count: int) -> None:
    """Check the common campaign-ledger contract."""
    raw_ledger = path.read_text(encoding="utf-8")
    assert "concentration" not in raw_ledger
    records = _read_jsonl(path)
    assert len(records) == expected_count
    requests: list[tuple[object, object]] = []
    for record in records:
        assert set(record) == {"reasoning", "request", "result"}
        request = record["request"]
        result = record["result"]
        assert isinstance(request, dict)
        assert isinstance(result, dict)
        request_data = cast("dict[str, object]", request)
        result_data = cast("dict[str, object]", result)
        assert set(request_data) == {"sequence", "ph"}
        assert result_data["found"] is True
        assert set(result_data) == {"found", "values"}
        values = result_data["values"]
        assert isinstance(values, dict)
        assert set(values) == {"surface_tension"}
        requests.append((request_data["sequence"], request_data["ph"]))
    assert len(requests) == len(set(requests))


def test_random_sampling_is_deterministic_and_has_no_replacement(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.csv"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_dataset(dataset)

    random_sampling.run(dataset, first, budget=8, seed=31)
    random_sampling.run(dataset, second, budget=8, seed=31)

    assert first.read_bytes() == second.read_bytes()
    _assert_ledger_shape(first, 8)


@pytest.mark.parametrize("budget", [0, 13])
def test_random_sampling_rejects_invalid_budget(tmp_path: Path, budget: int) -> None:
    dataset = tmp_path / "dataset.csv"
    _write_dataset(dataset)

    with pytest.raises(ValueError, match="Budget"):
        random_sampling.run(
            tmp_path / "dataset.csv", tmp_path / "out.jsonl", budget=budget, seed=0,
        )


@pytest.mark.parametrize("module", [random_sampling, active_learning])
def test_baselines_require_campaign_columns(tmp_path: Path, module: Any) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("sequence,pH\nAAA,7\n", encoding="utf-8")

    with pytest.raises(ValueError, match="surface_tension"):
        module.load_dataset(dataset)


def test_active_learning_is_deterministic_on_synthetic_data(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_dataset(dataset)
    options = active_learning.ActiveLearningOptions(
        budget=7,
        initial_samples=3,
        seed=17,
        xi=0.02,
    )

    active_learning.run(dataset, first, options=options)
    active_learning.run(dataset, second, options=options)

    assert first.read_bytes() == second.read_bytes()
    _assert_ledger_shape(first, 7)
    records = _read_jsonl(first)
    first_reasoning = records[0]["reasoning"]
    fourth_reasoning = records[3]["reasoning"]
    assert isinstance(first_reasoning, str)
    assert isinstance(fourth_reasoning, str)
    assert "Uniform random initialization" in first_reasoning
    assert "Sequential Bayesian optimization" in fourth_reasoning


@pytest.mark.parametrize(
    ("budget", "initial_samples", "xi"),
    [(0, 1, 0.01), (13, 1, 0.01), (4, 0, 0.01), (4, 5, 0.01), (4, 1, -0.01)],
)
def test_active_learning_rejects_invalid_options(
    tmp_path: Path,
    budget: int,
    initial_samples: int,
    xi: float,
) -> None:
    dataset = tmp_path / "dataset.csv"
    _write_dataset(dataset)
    options = active_learning.ActiveLearningOptions(
        budget=budget,
        initial_samples=initial_samples,
        xi=xi,
    )

    with pytest.raises(ValueError, match=r"Budget|Initial samples|xi"):
        active_learning.run(dataset, tmp_path / "out.jsonl", options=options)


def test_short_real_dataset_campaigns_work_with_hit_calculator(tmp_path: Path) -> None:
    dataset = Path(__file__).parents[1] / "data" / "dataset.csv"
    random_ledger = tmp_path / "random.jsonl"
    active_ledger = tmp_path / "active.jsonl"

    random_sampling.run(dataset, random_ledger, budget=6, seed=4)
    active_learning.run(
        dataset,
        active_ledger,
        options=active_learning.ActiveLearningOptions(
            budget=5,
            initial_samples=3,
            seed=4,
        ),
    )

    _assert_ledger_shape(random_ledger, 6)
    _assert_ledger_shape(active_ledger, 5)
    random_analysis = calculate_hits.analyze_surface_tension(random_ledger, 50.0)
    active_analysis = calculate_hits.analyze_surface_tension(active_ledger, 50.0)
    assert len(random_analysis.measurements) == 6
    assert len(active_analysis.measurements) == 5


def test_hit_report_includes_statistics_ranking_and_ledger_quality(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "observations.jsonl"
    ledger.write_text(
        '{"request":{"sequence":"AAA","ph":7},"result":{"values":{"surface_tension":49}}}\n'
        '{"request":{"sequence":"AAA","ph":7},"result":{"values":{"surface_tension":50}}}\n'
        '{"request":{"sequence":"BBB","ph":3},"result":{"values":{"surface_tension":70}}}\n'
        '{"request":{"sequence":"CCC","ph":11},"result":{"found":false}}\n'
        '{"result":{"values":{"surface_tension":NaN}}}\n'
        "not JSON\n"
        "\n",
        encoding="utf-8",
    )

    analysis = calculate_hits.analyze_surface_tension(ledger, 50.0)
    report = calculate_hits.format_report(analysis, top=2)

    assert analysis.physical_lines == 7
    assert analysis.blank_lines == 1
    assert analysis.invalid_json_lines == 1
    assert analysis.missing_measurement_lines == 2
    assert len(analysis.measurements) == 3
    assert len(analysis.hits) == 1
    assert analysis.unique_condition_count == 2
    assert analysis.duplicate_count == 1
    assert "Hits:                 1 (33.3%)" in report
    assert "Median:               50.000000" in report
    assert "Top 2 conditions" in report
    assert "1. sequence=AAA, pH=7, surface_tension=49.000000" in report
    assert "Invalid JSON:          1" in report


def test_hit_report_handles_a_ledger_without_measurements(tmp_path: Path) -> None:
    ledger = tmp_path / "empty.jsonl"
    ledger.write_text('{"result":{"found":false}}\n', encoding="utf-8")

    analysis = calculate_hits.analyze_surface_tension(ledger, 50.0)
    report = calculate_hits.format_report(analysis)

    assert not analysis.measurements
    assert "Hits:                 0 (0.0%)" in report
    assert "No usable surface-tension measurements were found." in report
