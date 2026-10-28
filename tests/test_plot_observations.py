"""Tests for the observation plotting script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from matplotlib import pyplot as plt


def _load_script() -> Any:
    """Load the repository script as a test module."""
    path = Path(__file__).parents[1] / "scripts" / "plot_observations.py"
    spec = importlib.util.spec_from_file_location("plot_observations", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast("Any", module)


plot_observations = _load_script()


def _write_ledger(path: Path, results: list[float | None]) -> None:
    """Write a campaign ledger where ``None`` is an unsuccessful lookup."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for value in results:
        result: dict[str, object]
        if value is None:
            result = {"found": False}
        else:
            result = {"found": True, "values": {"surface_tension": value}}
        records.append({"request": {"sequence": "AAA", "ph": 7}, "result": result})
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def test_read_campaign_ignores_failures_and_uses_accepted_sample_index(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "campaign.jsonl"
    _write_ledger(ledger, [49.0, None, 50.0, 40.0, 1.0])

    trajectory = plot_observations.read_campaign(
        ledger,
        budget=3,
        threshold=50.0,
    )

    assert trajectory == pytest.approx((100.0, 50.0, 200 / 3))


def test_read_campaign_data_retains_attempts_and_conditions(tmp_path: Path) -> None:
    ledger = tmp_path / "campaign.jsonl"
    _write_ledger(ledger, [49.0, None, 50.0, 40.0, 1.0])

    campaign = plot_observations.read_campaign_data(ledger, budget=3)

    assert campaign.attempt_values == (49.0, None, 50.0, 40.0)
    assert [measurement.surface_tension for measurement in campaign.measurements] == [
        49.0,
        50.0,
        40.0,
    ]
    assert [measurement.ph for measurement in campaign.measurements] == [7.0, 7.0, 7.0]


def test_cumulative_hit_rate_hits_and_mean_trajectory() -> None:
    first = plot_observations.cumulative_hit_rate([40.0, 60.0, 45.0], 50.0)
    second = plot_observations.cumulative_hit_rate([60.0, 40.0, 70.0], 50.0)

    assert first == pytest.approx((100.0, 50.0, 200 / 3))
    assert second == pytest.approx((0.0, 50.0, 100 / 3))
    assert plot_observations.mean_trajectory([first, second]) == pytest.approx(
        (50.0, 50.0, 50.0),
    )
    assert plot_observations.cumulative_hits([40.0, 60.0, 45.0], 50.0) == (
        1.0,
        1.0,
        2.0,
    )


def test_bootstrap_band_is_deterministic_and_needs_multiple_campaigns() -> None:
    campaigns = ((0.0, 50.0), (100.0, 50.0), (50.0, 100.0))

    first = plot_observations.bootstrap_confidence_band(
        campaigns,
        resamples=200,
        seed=7,
    )
    second = plot_observations.bootstrap_confidence_band(
        campaigns,
        resamples=200,
        seed=7,
    )

    assert first == second
    assert first is not None
    lower, upper = first
    assert all(low <= high for low, high in zip(lower, upper, strict=True))
    assert plot_observations.bootstrap_confidence_band((campaigns[0],)) is None


def test_campaign_scores_and_primary_inference_are_deterministic() -> None:
    with_wiki = ((0.0, 1.0), (1.0, 2.0))
    without_wiki = ((2.0, 3.0), (3.0, 4.0))

    with_wiki_scores = plot_observations.campaign_scores(with_wiki)
    without_wiki_scores = plot_observations.campaign_scores(without_wiki)
    first_interval = plot_observations.bootstrap_mean_difference(
        with_wiki_scores,
        without_wiki_scores,
        resamples=200,
        seed=7,
    )
    second_interval = plot_observations.bootstrap_mean_difference(
        with_wiki_scores,
        without_wiki_scores,
        resamples=200,
        seed=7,
    )

    assert with_wiki_scores == (0.5, 1.5)
    assert without_wiki_scores == (2.5, 3.5)
    assert first_interval == second_interval


def test_exact_permutation_test_checks_all_assignments() -> None:
    p_value, extreme_assignments, total_assignments = (
        plot_observations.exact_permutation_test((1.0, 2.0), (3.0, 4.0))
    )

    assert p_value == pytest.approx(1 / 3)
    assert extreme_assignments == 2
    assert total_assignments == 6


def test_primary_comparison_uses_wiki_minus_no_wiki_direction() -> None:
    approaches = (
        plot_observations.Approach(
            key="no_wiki_agent",
            campaigns=((2.0, 3.0), (3.0, 4.0)),
        ),
        plot_observations.Approach(
            key="wiki_agent",
            campaigns=((0.0, 1.0), (1.0, 2.0)),
        ),
    )

    comparison = plot_observations.primary_comparison(
        approaches,
        resamples=200,
        seed=7,
    )

    assert comparison is not None
    assert comparison.with_wiki_mean == pytest.approx(1.0)
    assert comparison.without_wiki_mean == pytest.approx(3.0)
    assert comparison.mean_difference == pytest.approx(-2.0)
    assert comparison.permutation_p_value == pytest.approx(1 / 3)


def test_draw_approach_uses_summaries_and_marks_single_runs() -> None:
    figure, axes = plt.subplots()
    repeated = plot_observations.Approach(
        key="random",
        campaigns=((0.0, 50.0), (100.0, 50.0)),
    )
    single = plot_observations.Approach(
        key="wiki_agent",
        campaigns=((0.0, 50.0),),
    )

    plot_observations._draw_approach(axes, repeated, color="blue", seed=7)
    plot_observations._draw_approach(axes, single, color="pink", seed=8)

    assert len(axes.lines) == 2
    assert axes.lines[0].get_label() == "Random sampling (n = 2)"
    assert axes.lines[0].get_linestyle() == "-"
    assert axes.lines[1].get_label() == "With Wiki (single run)"
    assert axes.lines[1].get_linestyle() == "--"
    assert [text.get_text() for text in axes.texts] == ["50.0"]
    plt.close(figure)


def test_discovery_skips_empty_directories_and_uses_stable_order(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "observations"
    (input_dir / "no_wiki_agent").mkdir(parents=True)
    _write_ledger(input_dir / "unknown_method" / "1.jsonl", [40.0, 60.0])
    _write_ledger(input_dir / "active_learning" / "1.jsonl", [60.0, 40.0])
    _write_ledger(input_dir / "random" / "1.jsonl", [40.0, 40.0])

    approaches = plot_observations.discover_approaches(
        input_dir,
        budget=2,
        threshold=50.0,
    )

    assert [approach.key for approach in approaches] == [
        "random",
        "active_learning",
        "unknown_method",
    ]
    assert approaches[0].campaigns == ((1.0, 2.0),)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json\n", "invalid JSON"),
        ('{"result":{"found":true}}\n', "result.values"),
        ('{"result":{"found":"yes"}}\n', "result.found"),
        ("\n", "blank JSONL record"),
    ],
)
def test_malformed_campaigns_report_file_and_line(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(plot_observations.CampaignDataError, match=message) as error:
        plot_observations.read_campaign(path, budget=1, threshold=50.0)

    assert f"{path}:1" in str(error.value)


def test_incomplete_campaign_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "short.jsonl"
    _write_ledger(path, [40.0, None])

    with pytest.raises(plot_observations.CampaignDataError, match=r"found 1.*need 2"):
        plot_observations.read_campaign(path, budget=2, threshold=50.0)


def test_plot_is_created_with_headless_backend(tmp_path: Path) -> None:
    input_dir = tmp_path / "observations"
    _write_ledger(input_dir / "random" / "1.jsonl", [40.0, 60.0, 40.0])
    _write_ledger(input_dir / "random" / "2.jsonl", [60.0, 40.0, 60.0])
    _write_ledger(input_dir / "active_learning" / "1.jsonl", [40.0, 60.0, 40.0])
    _write_ledger(input_dir / "no_wiki_agent" / "1.jsonl", [60.0, 40.0, 60.0])
    _write_ledger(input_dir / "wiki_agent" / "1.jsonl", [40.0, 40.0, 60.0])
    output = tmp_path / "figures" / "performance.png"

    approaches = plot_observations.plot_observations(
        input_dir,
        output,
        budget=3,
        threshold=50.0,
    )

    assert [len(approach.campaigns) for approach in approaches] == [1, 1, 2, 1]
    assert output.is_file()
    assert output.stat().st_size > 0
    for name in plot_observations.ADDITIONAL_FIGURE_NAMES:
        companion = output.with_name(f"{name}{output.suffix}")
        assert companion.is_file()
        assert companion.stat().st_size > 0
