"""Plot cumulative surface-tension hits across sampling approaches."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

DEFAULT_INPUT_DIR = Path("data/observations")
DEFAULT_OUTPUT = Path("figures/approach_performance.pdf")
DEFAULT_BUDGET = 100
DEFAULT_THRESHOLD = 40.0
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_824

KNOWN_APPROACH_ORDER = (
    "no_wiki_agent",
    "wiki_agent",
    "random",
    "active_learning",
)
APPROACH_NAMES = {
    "random": "Random sampling",
    "active_learning": "Active learning",
    "no_wiki_agent": "Without Wiki",
    "wiki_agent": "With Wiki",
}
APPROACH_COLORS = {
    "no_wiki_agent": "#009E73",
    "wiki_agent": "#CC79A7",
    "random": "#7F8C8D",
    "active_learning": "#4C78A8",
}
APPROACH_LINE_STYLES = {
    "no_wiki_agent": "-",
    "wiki_agent": "--",
    "random": ":",
    "active_learning": "-.",
}
FALLBACK_COLORS = ("#0072B2", "#E69F00", "#D55E00", "#56B4E9")
FALLBACK_LINE_STYLES = ("-", "--", "-.", ":")
FULL_WIDTH_FIGURE = (7.0, 4.2)
ADDITIONAL_FIGURE_NAMES = (
    "final_hit_rate_distribution",
    "threshold_sensitivity",
    "best_surface_tension",
    "selection_mechanism",
)


class CampaignDataError(ValueError):
    """Indicate that campaign input cannot produce an unbiased plot."""


@dataclass(frozen=True, slots=True)
class Approach:
    """Store the cumulative trajectories for one sampling approach."""

    key: str
    campaigns: tuple[tuple[float, ...], ...]
    raw_campaigns: tuple[Campaign, ...] = ()

    @property
    def name(self) -> str:
        """The readable approach name."""
        return APPROACH_NAMES.get(self.key, self.key.replace("_", " ").title())


@dataclass(frozen=True, slots=True)
class Measurement:
    """Store one successful measurement and its request condition."""

    sequence: str
    surface_tension: float
    ph: float | None


@dataclass(frozen=True, slots=True)
class Campaign:
    """Store successful measurements and attempt outcomes for one campaign."""

    measurements: tuple[Measurement, ...]
    attempt_values: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class PrimaryComparison:
    """Store the primary Wiki-access comparison statistics."""

    with_wiki_scores: tuple[float, ...]
    without_wiki_scores: tuple[float, ...]
    with_wiki_mean: float
    without_wiki_mean: float
    mean_difference: float
    bootstrap_interval: tuple[float, float]
    permutation_p_value: float
    extreme_assignments: int
    total_assignments: int


def _as_object(value: object, *, path: Path, line_number: int, field: str) -> dict[str, object]:
    """Narrow a JSON value to an object or report its location."""
    if not isinstance(value, dict):
        raise CampaignDataError(f"{path}:{line_number}: {field} must be an object")
    return cast("dict[str, object]", value)


def _surface_tension(record: object, *, path: Path, line_number: int) -> float | None:
    """Extract a successful measurement and ignore a failed lookup."""
    record_data = _as_object(record, path=path, line_number=line_number, field="record")
    result = _as_object(
        record_data.get("result"),
        path=path,
        line_number=line_number,
        field="result",
    )
    found = result.get("found")
    if not isinstance(found, bool):
        raise CampaignDataError(f"{path}:{line_number}: result.found must be a boolean")
    if not found:
        return None

    values = _as_object(
        result.get("values"),
        path=path,
        line_number=line_number,
        field="result.values",
    )
    value = values.get("surface_tension")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CampaignDataError(
            f"{path}:{line_number}: result.values.surface_tension must be a number",
        )
    measurement = float(value)
    if not math.isfinite(measurement):
        raise CampaignDataError(
            f"{path}:{line_number}: result.values.surface_tension must be finite",
        )
    return measurement


def _request_ph(record: object) -> float | None:
    """Extract a finite pH value when the request contains one."""
    if not isinstance(record, dict):
        return None
    request = record.get("request")
    if not isinstance(request, dict):
        return None
    value = request.get("ph")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    ph = float(value)
    return ph if math.isfinite(ph) else None


def _request_sequence(record: object) -> str:
    """Extract the requested sequence when it is available."""
    if not isinstance(record, dict):
        return ""
    request = record.get("request")
    if not isinstance(request, dict):
        return ""
    sequence = request.get("sequence")
    return sequence if isinstance(sequence, str) else ""


def cumulative_hit_rate(
    measurements: Sequence[float],
    threshold: float,
) -> tuple[float, ...]:
    """Calculate the cumulative percentage below an exclusive threshold.

    Args:
        measurements: Successful measurements in accepted-sample order.
        threshold: The exclusive hit threshold in mN/m.

    Returns:
        The cumulative hit-rate percentage after each measurement.

    Raises:
        ValueError: If the threshold or a measurement is not finite.

    """
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")

    rates: list[float] = []
    hits = 0
    for sample_number, measurement in enumerate(measurements, start=1):
        if not math.isfinite(measurement):
            raise ValueError("Measurements must be finite")
        hits += measurement < threshold
        rates.append(hits / sample_number * 100.0)
    return tuple(rates)


def cumulative_hits(
    measurements: Sequence[float],
    threshold: float,
) -> tuple[float, ...]:
    """Calculate cumulative hits below an exclusive threshold."""
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")

    trajectory: list[float] = []
    hits = 0
    for measurement in measurements:
        if not math.isfinite(measurement):
            raise ValueError("Measurements must be finite")
        hits += measurement < threshold
        trajectory.append(float(hits))
    return tuple(trajectory)


def read_campaign_data(path: Path, *, budget: int) -> Campaign:
    """Read one ledger and retain data through the successful-measurement budget.

    Args:
        path: The JSONL campaign ledger.
        budget: The required count of successful measurements.

    Returns:
        The successful measurements and all attempts needed to obtain them.

    Raises:
        CampaignDataError: If a record is malformed or the campaign is incomplete.
        ValueError: If the budget is invalid.

    """
    if budget < 1:
        raise ValueError("Budget must be at least 1")

    try:
        ledger = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise CampaignDataError(f"Cannot read {path}: {exc}") from exc

    measurements: list[Measurement] = []
    attempt_values: list[float | None] = []
    with ledger:
        for line_number, line in enumerate(ledger, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                raise CampaignDataError(f"{path}:{line_number}: blank JSONL record")
            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise CampaignDataError(
                    f"{path}:{line_number}: invalid JSON ({exc.msg})",
                ) from exc
            measurement = _surface_tension(
                record,
                path=path,
                line_number=line_number,
            )
            if len(measurements) >= budget:
                continue
            attempt_values.append(measurement)
            if measurement is not None:
                measurements.append(
                    Measurement(
                        sequence=_request_sequence(record),
                        surface_tension=measurement,
                        ph=_request_ph(record),
                    ),
                )

    if len(measurements) < budget:
        raise CampaignDataError(
            f"{path}: incomplete campaign: found {len(measurements)} successful "
            f"measurements; need {budget}",
        )
    return Campaign(
        measurements=tuple(measurements),
        attempt_values=tuple(attempt_values),
    )


def read_campaign(path: Path, *, budget: int, threshold: float) -> tuple[float, ...]:
    """Read one ledger and return its budget-limited cumulative trajectory."""
    campaign = read_campaign_data(path, budget=budget)
    return cumulative_hit_rate(
        [measurement.surface_tension for measurement in campaign.measurements],
        threshold,
    )


def _approach_sort_key(key: str) -> tuple[int, int | str]:
    """Place known approaches first and unknown approaches alphabetically."""
    try:
        return 0, KNOWN_APPROACH_ORDER.index(key)
    except ValueError:
        return 1, key


def _approach_color(approach: Approach, index: int) -> str:
    """Return the manuscript color for an approach."""
    return APPROACH_COLORS.get(
        approach.key,
        FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
    )


def _approach_linestyle(approach: Approach, index: int) -> str:
    """Return the manuscript line style for an approach."""
    return APPROACH_LINE_STYLES.get(
        approach.key,
        FALLBACK_LINE_STYLES[index % len(FALLBACK_LINE_STYLES)],
    )


def _is_agent_approach(approach: Approach) -> bool:
    """Return true for an agent approach in the main comparison."""
    return approach.key in {"no_wiki_agent", "wiki_agent"}


def discover_approaches(
    input_dir: Path,
    *,
    budget: int,
    threshold: float,
) -> tuple[Approach, ...]:
    """Load campaigns from immediate approach subdirectories.

    Args:
        input_dir: The directory that contains approach subdirectories.
        budget: The successful-measurement budget per campaign.
        threshold: The exclusive hit threshold in mN/m.

    Returns:
        Nonempty approaches in stable display order.

    Raises:
        CampaignDataError: If the input is absent, invalid, or contains bad campaigns.

    """
    if not input_dir.is_dir():
        raise CampaignDataError(f"Input directory does not exist: {input_dir}")

    approaches: list[Approach] = []
    errors: list[str] = []
    for approach_dir in sorted(
        (entry for entry in input_dir.iterdir() if entry.is_dir()),
        key=lambda entry: _approach_sort_key(entry.name),
    ):
        campaign_paths = sorted(
            approach_dir.glob("*.jsonl"),
            key=lambda path: int(path.stem),
        )
        if not campaign_paths:
            continue
        campaigns: list[tuple[float, ...]] = []
        raw_campaigns: list[Campaign] = []
        for campaign_path in campaign_paths:
            try:
                raw_campaign = read_campaign_data(campaign_path, budget=budget)
                campaigns.append(
                    cumulative_hits(
                        [
                            measurement.surface_tension
                            for measurement in raw_campaign.measurements
                        ],
                        threshold,
                    ),
                )
                raw_campaigns.append(raw_campaign)
            except (CampaignDataError, ValueError) as exc:
                errors.append(str(exc))
        if campaigns:
            approaches.append(
                Approach(
                    key=approach_dir.name,
                    campaigns=tuple(campaigns),
                    raw_campaigns=tuple(raw_campaigns),
                ),
            )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise CampaignDataError(f"Invalid campaign data:\n{details}")
    if not approaches:
        raise CampaignDataError(f"No JSONL campaigns found in {input_dir}")
    return tuple(approaches)


def mean_trajectory(campaigns: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Calculate the arithmetic mean trajectory across campaigns."""
    if not campaigns:
        raise ValueError("At least one campaign is required")
    width = len(campaigns[0])
    if width == 0 or any(len(campaign) != width for campaign in campaigns):
        raise ValueError("Campaign trajectories must have the same nonzero length")
    return tuple(
        statistics.fmean(campaign[sample] for campaign in campaigns)
        for sample in range(width)
    )


def campaign_scores(campaigns: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Calculate the mean cumulative-hits score for each campaign."""
    if not campaigns:
        raise ValueError("At least one campaign is required")
    width = len(campaigns[0])
    if width == 0 or any(len(campaign) != width for campaign in campaigns):
        raise ValueError("Campaign trajectories must have the same nonzero length")
    return tuple(statistics.fmean(campaign) for campaign in campaigns)


def bootstrap_mean_difference(
    with_wiki_scores: Sequence[float],
    without_wiki_scores: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap the difference in group means within each condition."""
    if not with_wiki_scores or not without_wiki_scores:
        raise ValueError("Both score groups must be nonempty")
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be at least 1")

    rng = random.Random(  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        seed,
    )
    distribution = []
    for _ in range(resamples):
        with_wiki_mean = statistics.fmean(
            with_wiki_scores[rng.randrange(len(with_wiki_scores))]
            for _ in with_wiki_scores
        )
        without_wiki_mean = statistics.fmean(
            without_wiki_scores[rng.randrange(len(without_wiki_scores))]
            for _ in without_wiki_scores
        )
        distribution.append(with_wiki_mean - without_wiki_mean)

    distribution.sort()
    return _percentile(distribution, 0.025), _percentile(distribution, 0.975)


def exact_permutation_test(
    with_wiki_scores: Sequence[float],
    without_wiki_scores: Sequence[float],
) -> tuple[float, int, int]:
    """Run an exact two-sided unpaired permutation test on group means."""
    if not with_wiki_scores or not without_wiki_scores:
        raise ValueError("Both score groups must be nonempty")

    pooled = tuple(with_wiki_scores) + tuple(without_wiki_scores)
    with_wiki_count = len(with_wiki_scores)
    without_wiki_count = len(without_wiki_scores)
    pooled_total = sum(pooled)
    observed = abs(
        statistics.fmean(with_wiki_scores)
        - statistics.fmean(without_wiki_scores),
    )
    extreme_assignments = 0
    for indices in itertools.combinations(range(len(pooled)), with_wiki_count):
        selected_total = sum(pooled[index] for index in indices)
        difference = abs(
            selected_total / with_wiki_count
            - (pooled_total - selected_total) / without_wiki_count,
        )
        extreme_assignments += difference >= observed - 1e-12

    total_assignments = math.comb(len(pooled), with_wiki_count)
    return (
        extreme_assignments / total_assignments,
        extreme_assignments,
        total_assignments,
    )


def primary_comparison(
    approaches: Sequence[Approach],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> PrimaryComparison | None:
    """Calculate the primary comparison when both agent conditions are present."""
    by_key = {approach.key: approach for approach in approaches}
    if "wiki_agent" not in by_key or "no_wiki_agent" not in by_key:
        return None

    with_wiki_scores = campaign_scores(by_key["wiki_agent"].campaigns)
    without_wiki_scores = campaign_scores(by_key["no_wiki_agent"].campaigns)
    with_wiki_mean = statistics.fmean(with_wiki_scores)
    without_wiki_mean = statistics.fmean(without_wiki_scores)
    permutation_p_value, extreme_assignments, total_assignments = (
        exact_permutation_test(with_wiki_scores, without_wiki_scores)
    )
    return PrimaryComparison(
        with_wiki_scores=with_wiki_scores,
        without_wiki_scores=without_wiki_scores,
        with_wiki_mean=with_wiki_mean,
        without_wiki_mean=without_wiki_mean,
        mean_difference=with_wiki_mean - without_wiki_mean,
        bootstrap_interval=bootstrap_mean_difference(
            with_wiki_scores,
            without_wiki_scores,
            resamples=resamples,
            seed=seed,
        ),
        permutation_p_value=permutation_p_value,
        extreme_assignments=extreme_assignments,
        total_assignments=total_assignments,
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Calculate a linearly interpolated percentile from sorted values."""
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def bootstrap_confidence_band(
    campaigns: Sequence[Sequence[float]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Calculate a deterministic pointwise 95% bootstrap confidence band.

    The function resamples whole campaigns. It returns ``None`` when one campaign
    cannot describe cross-campaign uncertainty.
    """
    campaign_count = len(campaigns)
    if campaign_count == 1:
        return None
    if campaign_count == 0:
        raise ValueError("At least one campaign is required")
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be at least 1")
    width = len(campaigns[0])
    if width == 0 or any(len(campaign) != width for campaign in campaigns):
        raise ValueError("Campaign trajectories must have the same nonzero length")

    rng = random.Random(  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        seed,
    )
    distributions = [[] for _ in range(width)]
    for _ in range(resamples):
        selected = [rng.randrange(campaign_count) for _ in range(campaign_count)]
        for sample, distribution in enumerate(distributions):
            distribution.append(
                statistics.fmean(campaigns[index][sample] for index in selected),
            )

    lower: list[float] = []
    upper: list[float] = []
    for distribution in distributions:
        distribution.sort()
        lower.append(_percentile(distribution, 0.025))
        upper.append(_percentile(distribution, 0.975))
    return tuple(lower), tuple(upper)


def _draw_approach(
    axes: Axes,
    approach: Approach,
    *,
    color: str,
    seed: int,
    linestyle: str | None = None,
) -> None:
    """Draw the summary trajectory for one approach."""
    sample_numbers = range(len(approach.campaigns[0]) + 1)
    campaign_count = len(approach.campaigns)
    band = bootstrap_confidence_band(approach.campaigns, seed=seed)
    if band is not None:
        lower, upper = band
        axes.fill_between(
            sample_numbers,
            (0.0, *lower),
            (0.0, *upper),
            color=color,
            alpha=0.16 if _is_agent_approach(approach) else 0.07,
            linewidth=0,
            zorder=1,
        )

        selected_linestyle = "-" if linestyle is None else linestyle
        label = f"{approach.name} (n = {campaign_count})"
    else:
        selected_linestyle = "--" if linestyle is None else linestyle
        label = f"{approach.name} (single run)"

    trajectory = (0.0, *mean_trajectory(approach.campaigns))
    axes.plot(
        sample_numbers,
        trajectory,
        color=color,
        linewidth=2.7 if _is_agent_approach(approach) else 1.5,
        alpha=1.0 if _is_agent_approach(approach) else 0.72,
        linestyle=selected_linestyle,
        label=label,
        zorder=2,
    )
    if campaign_count > 1:
        label_offset = {
            "no_wiki_agent": 4,
            "wiki_agent": -5,
        }.get(approach.key, 0)
        axes.annotate(
            f"{trajectory[-1]:.1f}",
            xy=(len(trajectory) - 1, trajectory[-1]),
            xytext=(5, label_offset),
            textcoords="offset points",
            color=color,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
            clip_on=False,
        )


def _style_axes(axes: Axes) -> None:
    """Apply the shared axis style."""
    axes.grid(axis="y", color="#D8D8D8", linewidth=0.65)
    axes.set_axisbelow(True)
    axes.spines[["top", "right"]].set_visible(False)


def _figure_path(output: Path, name: str) -> Path:
    """Create a companion figure path next to the main output."""
    return output.with_name(f"{name}{output.suffix}")


def _draw_trajectory_summary(  # ruff: ignore[too-many-arguments] - Plot style options are explicit.
    axes: Axes,
    approach: Approach,
    trajectories: Sequence[Sequence[float]],
    *,
    color: str,
    seed: int,
    linestyle: str | None = None,
) -> None:
    """Draw a mean trajectory, its uncertainty, or a single-run line."""
    sample_numbers = range(1, len(trajectories[0]) + 1)
    band = bootstrap_confidence_band(trajectories, seed=seed)
    if band is None:
        selected_linestyle = "--" if linestyle is None else linestyle
        label = f"{approach.name} (single run)"
    else:
        lower, upper = band
        axes.fill_between(
            sample_numbers,
            lower,
            upper,
            color=color,
            alpha=0.16 if _is_agent_approach(approach) else 0.07,
            linewidth=0,
        )
        selected_linestyle = "-" if linestyle is None else linestyle
        label = f"{approach.name} (n = {len(trajectories)})"
    axes.plot(
        sample_numbers,
        mean_trajectory(trajectories),
        color=color,
        linewidth=2.5 if _is_agent_approach(approach) else 1.4,
        alpha=1.0 if _is_agent_approach(approach) else 0.7,
        linestyle=selected_linestyle,
        label=label,
    )


def _plot_final_hit_rate_distribution(
    approaches: Sequence[Approach],
    output: Path,
    *,
    threshold: float,
) -> None:
    """Plot campaign-level hit rates at the final measurement budget."""
    figure, axes = plt.subplots(figsize=FULL_WIDTH_FIGURE, constrained_layout=True)
    rng = random.Random(  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        BOOTSTRAP_SEED,
    )
    for index, approach in enumerate(approaches, start=1):
        color = _approach_color(approach, index - 1)
        values = [
            campaign[-1] / len(campaign) * 100.0 for campaign in approach.campaigns
        ]
        if len(values) > 1:
            box = axes.boxplot(
                values,
                positions=[index],
                widths=0.48,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.4},
            )
            box["boxes"][0].set_facecolor(color)
            box["boxes"][0].set_alpha(0.22)
            for item in (*box["whiskers"], *box["caps"]):
                item.set_color(color)
            x_values = [index + rng.uniform(-0.13, 0.13) for _ in values]
            axes.scatter(x_values, values, color=color, s=18, alpha=0.48, zorder=3)
        else:
            axes.scatter(
                [index],
                values,
                color=color,
                marker="D",
                edgecolor="black",
                linewidth=0.6,
                s=62,
                zorder=3,
            )
    axes.set_xticks(
        range(1, len(approaches) + 1),
        [approach.name for approach in approaches],
        rotation=15,
        ha="right",
    )
    axes.set_ylabel(f"Final hit rate (< {threshold:g} mN/m)")
    axes.set_ylim(0, 100)
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    _style_axes(axes)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_best_surface_tension(
    approaches: Sequence[Approach],
    output: Path,
) -> None:
    """Plot the best surface tension found by each measurement number."""
    figure, axes = plt.subplots(figsize=FULL_WIDTH_FIGURE, constrained_layout=True)
    for index, approach in enumerate(approaches):
        trajectories: list[tuple[float, ...]] = []
        for campaign in approach.raw_campaigns:
            best = math.inf
            trajectory: list[float] = []
            for measurement in campaign.measurements:
                best = min(best, measurement.surface_tension)
                trajectory.append(best)
            trajectories.append(tuple(trajectory))
        _draw_trajectory_summary(
            axes,
            approach,
            trajectories,
            color=_approach_color(approach, index),
            seed=BOOTSTRAP_SEED + index,
            linestyle=_approach_linestyle(approach, index),
        )
    axes.set_xlabel("Successful measurements")
    axes.set_ylabel("Best surface tension found (mN/m)")
    _style_axes(axes)
    axes.legend(frameon=False, loc="best")
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _threshold_grid(approaches: Sequence[Approach]) -> tuple[float, ...]:
    """Create a shared threshold grid from the observed value range."""
    values = [
        measurement.surface_tension
        for approach in approaches
        for campaign in approach.raw_campaigns
        for measurement in campaign.measurements
    ]
    lower = math.floor(min(values))
    upper = math.ceil(max(values))
    width = upper - lower
    return tuple(lower + width * index / 100 for index in range(101))


def _plot_threshold_sensitivity(
    approaches: Sequence[Approach],
    output: Path,
    *,
    selected_threshold: float,
) -> None:
    """Plot final hit rates over the observed surface-tension thresholds."""
    thresholds = _threshold_grid(approaches)
    figure, axes = plt.subplots(figsize=FULL_WIDTH_FIGURE, constrained_layout=True)
    for index, approach in enumerate(approaches):
        trajectories = [
            tuple(
                sum(
                    measurement.surface_tension < threshold
                    for measurement in campaign.measurements
                )
                / len(campaign.measurements)
                * 100
                for threshold in thresholds
            )
            for campaign in approach.raw_campaigns
        ]
        color = _approach_color(approach, index)
        band = bootstrap_confidence_band(
            trajectories,
            seed=BOOTSTRAP_SEED + index,
        )
        if band is not None:
            axes.fill_between(
                thresholds,
                band[0],
                band[1],
                color=color,
                alpha=0.16 if _is_agent_approach(approach) else 0.07,
                linewidth=0,
            )
        linestyle = _approach_linestyle(approach, index)
        suffix = (
            "single run" if len(trajectories) == 1 else f"n = {len(trajectories)}"
        )
        axes.plot(
            thresholds,
            mean_trajectory(trajectories),
            color=color,
            linewidth=2.5 if _is_agent_approach(approach) else 1.4,
            alpha=1.0 if _is_agent_approach(approach) else 0.7,
            linestyle=linestyle,
            label=f"{approach.name} ({suffix})",
        )
    axes.axvline(
        selected_threshold,
        color="#666666",
        linestyle=":",
        linewidth=1.2,
        label=f"Selected threshold ({selected_threshold:g})",
    )
    axes.set_xlabel("Exclusive hit threshold (mN/m)")
    axes.set_ylabel("Final hit rate")
    axes.set_ylim(0, 100)
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    _style_axes(axes)
    axes.legend(frameon=False, loc="best")
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _attempt_hit_trajectory(
    attempt_values: Sequence[float | None],
    threshold: float,
) -> tuple[float, ...]:
    """Calculate cumulative hit counts over all lookup attempts."""
    hits = 0
    trajectory: list[float] = []
    for value in attempt_values:
        hits += value is not None and value < threshold
        trajectory.append(float(hits))
    return tuple(trajectory)


def _plot_attempt_performance(
    approaches: Sequence[Approach],
    output: Path,
    *,
    threshold: float,
) -> None:
    """Plot cumulative hits against successful and failed attempts."""
    figure, axes = plt.subplots(figsize=FULL_WIDTH_FIGURE, constrained_layout=True)
    for index, approach in enumerate(approaches):
        trajectories = [
            _attempt_hit_trajectory(campaign.attempt_values, threshold)
            for campaign in approach.raw_campaigns
        ]
        common_width = min(len(trajectory) for trajectory in trajectories)
        trajectories = [trajectory[:common_width] for trajectory in trajectories]
        _draw_trajectory_summary(
            axes,
            approach,
            trajectories,
            color=_approach_color(approach, index),
            seed=BOOTSTRAP_SEED + index,
            linestyle=_approach_linestyle(approach, index),
        )
    axes.set_xlabel("Lookup attempts (failed lookups included)")
    axes.set_ylabel(f"Cumulative hits (< {threshold:g} mN/m)")
    _style_axes(axes)
    axes.legend(frameon=False, loc="best")
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _attempt_to_hit_target(
    attempt_values: Sequence[float | None],
    threshold: float,
    target: int,
) -> int | None:
    """Return the attempt that reaches a cumulative hit target."""
    hits = 0
    for attempt, value in enumerate(attempt_values, start=1):
        hits += value is not None and value < threshold
        if hits >= target:
            return attempt
    return None


def _plot_time_to_hit_targets(
    approaches: Sequence[Approach],
    output: Path,
    *,
    budget: int,
    threshold: float,
) -> None:
    """Plot the campaign fraction that reaches selected hit targets."""
    targets = sorted({max(1, budget // 10), max(1, budget // 4), max(1, budget // 2)})
    figure, axes_row = plt.subplots(
        1,
        len(targets),
        figsize=(7.0, 3.8),
        constrained_layout=True,
        sharey=True,
    )
    axes_list = [axes_row] if len(targets) == 1 else list(axes_row)
    maximum_attempts = max(
        len(campaign.attempt_values)
        for approach in approaches
        for campaign in approach.raw_campaigns
    )
    attempt_numbers = range(1, maximum_attempts + 1)
    for axes, target in zip(axes_list, targets, strict=True):
        for index, approach in enumerate(approaches):
            reach_attempts = [
                _attempt_to_hit_target(campaign.attempt_values, threshold, target)
                for campaign in approach.raw_campaigns
            ]
            fractions = [
                sum(reached is not None and reached <= attempt for reached in reach_attempts)
                / len(reach_attempts)
                * 100
                for attempt in attempt_numbers
            ]
            linestyle = _approach_linestyle(approach, index)
            axes.step(
                attempt_numbers,
                fractions,
                where="post",
                color=_approach_color(approach, index),
                linewidth=2.5 if _is_agent_approach(approach) else 1.4,
                alpha=1.0 if _is_agent_approach(approach) else 0.7,
                linestyle=linestyle,
                label=approach.name,
            )
        axes.set_title(
            f"Target: {target} hits",
            fontfamily="serif",
            fontweight="bold",
        )
        axes.set_xlabel("Lookup attempts")
        axes.set_ylim(0, 100)
        axes.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        _style_axes(axes)
    axes_list[0].set_ylabel("Campaigns reaching target")
    axes_list[-1].legend(
        frameon=False,
        loc="upper left",
        ncol=1,
        fontsize=10,
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _mean_ph_metric(
    approach: Approach,
    ph: float,
    *,
    threshold: float,
    hit_rate: bool,
) -> float:
    """Calculate a mean campaign-level pH selection share or hit rate."""
    values: list[float] = []
    for campaign in approach.raw_campaigns:
        selected = [measurement for measurement in campaign.measurements if measurement.ph == ph]
        if hit_rate:
            if selected:
                values.append(
                    sum(measurement.surface_tension < threshold for measurement in selected)
                    / len(selected)
                    * 100,
                )
        else:
            values.append(len(selected) / len(campaign.measurements) * 100)
    return statistics.fmean(values) if values else math.nan


def _plot_ph_performance(
    approaches: Sequence[Approach],
    output: Path,
    *,
    threshold: float,
) -> None:
    """Plot pH selection shares and conditional hit rates."""
    ph_values = sorted(
        {
            measurement.ph
            for approach in approaches
            for campaign in approach.raw_campaigns
            for measurement in campaign.measurements
            if measurement.ph is not None
        },
    )
    figure, (coverage_axes, hit_axes) = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.8),
        constrained_layout=True,
        sharey=True,
    )
    group_width = 0.8
    bar_width = group_width / len(approaches)
    base_positions = list(range(len(ph_values)))
    for index, approach in enumerate(approaches):
        positions = [
            position - group_width / 2 + bar_width * (index + 0.5)
            for position in base_positions
        ]
        color = _approach_color(approach, index)
        coverage_axes.bar(
            positions,
            [
                _mean_ph_metric(approach, ph, threshold=threshold, hit_rate=False)
                for ph in ph_values
            ],
            width=bar_width,
            color=color,
            label=approach.name,
        )
        hit_axes.bar(
            positions,
            [
                _mean_ph_metric(approach, ph, threshold=threshold, hit_rate=True)
                for ph in ph_values
            ],
            width=bar_width,
            color=color,
        )
    for axes in (coverage_axes, hit_axes):
        axes.set_xticks(base_positions, [f"{ph:g}" for ph in ph_values])
        axes.set_xlabel("pH")
        axes.set_ylim(0, 100)
        axes.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        _style_axes(axes)
    coverage_axes.set_title(
        "Selection share",
        fontfamily="serif",
        fontweight="bold",
    )
    coverage_axes.set_ylabel("Mean campaign percentage")
    hit_axes.set_title(
        f"Hit rate among measurements (< {threshold:g} mN/m)\n"
        "Blank: no measurements",
        fontfamily="serif",
        fontweight="bold",
    )
    coverage_axes.legend(frameon=False, fontsize=10)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _required_approaches(approaches: Sequence[Approach]) -> dict[str, Approach]:
    """Return the four approaches that the manuscript figures require."""
    by_key = {approach.key: approach for approach in approaches}
    missing = [key for key in KNOWN_APPROACH_ORDER if key not in by_key]
    if missing:
        raise CampaignDataError(
            "Missing approaches required for manuscript figures: "
            + ", ".join(missing),
        )
    return by_key


def _bootstrap_difference_band(
    with_wiki: Sequence[Sequence[float]],
    without_wiki: Sequence[Sequence[float]],
    *,
    seed: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Calculate a pointwise interval for the difference between means."""
    rng = random.Random(  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        seed,
    )
    draws = [[] for _ in with_wiki[0]]
    for _ in range(BOOTSTRAP_RESAMPLES):
        with_rows = [rng.randrange(len(with_wiki)) for _ in with_wiki]
        without_rows = [rng.randrange(len(without_wiki)) for _ in without_wiki]
        for index, values in enumerate(draws):
            values.append(
                statistics.fmean(with_wiki[row][index] for row in with_rows)
                - statistics.fmean(
                    without_wiki[row][index] for row in without_rows
                ),
            )
    for values in draws:
        values.sort()
    return (
        tuple(_percentile(values, 0.025) for values in draws),
        tuple(_percentile(values, 0.975) for values in draws),
    )


def _plot_manuscript_primary(  # ruff: ignore[too-many-locals]
    approaches: Sequence[Approach],
    output: Path,
    *,
    budget: int,
) -> None:
    """Plot cumulative hits and the direct between-agent contrast."""
    by_key = _required_approaches(approaches)
    trajectories = {
        key: tuple((0.0, *campaign) for campaign in by_key[key].campaigns)
        for key in KNOWN_APPROACH_ORDER
    }
    figure, (main_axes, difference_axes) = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.7),
        gridspec_kw={"height_ratios": (3.2, 1.15), "hspace": 0.08},
        sharex=True,
    )
    x_values = range(budget + 1)
    final_means: dict[str, float] = {}
    for index, key in enumerate(KNOWN_APPROACH_ORDER):
        approach = by_key[key]
        mean = mean_trajectory(trajectories[key])
        final_means[key] = mean[-1]
        band = bootstrap_confidence_band(
            trajectories[key],
            seed=BOOTSTRAP_SEED + index,
        )
        if band is None:
            lower = upper = mean
        else:
            lower, upper = band
        agent = _is_agent_approach(approach)
        main_axes.fill_between(
            x_values,
            lower,
            upper,
            color=_approach_color(approach, index),
            alpha=0.16 if agent else 0.07,
            linewidth=0,
        )
        main_axes.plot(
            x_values,
            mean,
            color=_approach_color(approach, index),
            linestyle=_approach_linestyle(approach, index),
            linewidth=2.7 if agent else 1.5,
            alpha=1.0 if agent else 0.72,
            label=f"{approach.name} (n = {len(trajectories[key])})",
        )

    checkpoints = tuple(value for value in (25, 50, 100) if value <= budget)
    for measurement in checkpoints:
        main_axes.axvline(
            measurement,
            color="#B8B8B8",
            linewidth=0.7,
            linestyle=":",
        )
    main_axes.set_ylabel("Cumulative hits")
    main_axes.set_ylim(bottom=0)
    handles, labels = main_axes.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=10,
    )
    label_offsets = {
        "no_wiki_agent": 4,
        "wiki_agent": -5,
        "random": 0,
        "active_learning": 0,
    }
    for index, key in enumerate(KNOWN_APPROACH_ORDER):
        approach = by_key[key]
        final_mean = final_means[key]
        main_axes.annotate(
            f"{final_mean:.1f}",
            (budget, final_mean),
            xytext=(5, label_offsets[key]),
            textcoords="offset points",
            color=_approach_color(approach, index),
            fontsize=10,
            fontweight="bold" if _is_agent_approach(approach) else "normal",
            ha="left",
            va="center",
            clip_on=False,
        )
    _style_axes(main_axes)

    with_wiki = trajectories["wiki_agent"]
    without_wiki = trajectories["no_wiki_agent"]
    difference = tuple(
        with_value - without_value
        for with_value, without_value in zip(
            mean_trajectory(with_wiki),
            mean_trajectory(without_wiki),
            strict=True,
        )
    )
    lower, upper = _bootstrap_difference_band(
        with_wiki,
        without_wiki,
        seed=BOOTSTRAP_SEED,
    )
    difference_axes.fill_between(
        x_values,
        lower,
        upper,
        color="#6F4E7C",
        alpha=0.18,
        linewidth=0,
    )
    difference_axes.plot(x_values, difference, color="#6F4E7C", linewidth=2.0)
    difference_axes.axhline(0, color="#555555", linewidth=0.9)
    for measurement in checkpoints:
        difference_axes.axvline(
            measurement,
            color="#B8B8B8",
            linewidth=0.7,
            linestyle=":",
        )
        difference_axes.annotate(
            f"{difference[measurement]:+.1f}",
            (measurement, difference[measurement]),
            xytext=(-3 if measurement == budget else 3, -10),
            textcoords="offset points",
            ha="right" if measurement == budget else "left",
            fontsize=10,
            color="#53365F",
        )
    difference_axes.set_xlabel("Successful measurements")
    difference_axes.set_ylabel("With Wiki\nminus without")
    difference_axes.set_xlim(0, budget)
    _style_axes(difference_axes)
    figure.subplots_adjust(bottom=0.22)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _bootstrap_mean_interval(values: Sequence[float], seed: int) -> tuple[float, float]:
    """Calculate a percentile interval for a campaign mean."""
    rng = random.Random(  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        seed,
    )
    draws = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    draws.sort()
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def _plot_manuscript_final_counts(
    approaches: Sequence[Approach],
    output: Path,
    *,
    budget: int,
) -> None:
    """Plot agent outcomes and compact computational references."""
    by_key = _required_approaches(approaches)
    figure, (agent_axes, reference_axes) = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.4),
        gridspec_kw={"width_ratios": (1.3, 1.0)},
    )
    rng = random.Random(  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        BOOTSTRAP_SEED,
    )
    for position, key in enumerate(("no_wiki_agent", "wiki_agent"), start=1):
        approach = by_key[key]
        values = tuple(campaign[-1] for campaign in approach.campaigns)
        jitter = [position + rng.uniform(-0.10, 0.10) for _ in values]
        color = APPROACH_COLORS[key]
        agent_axes.scatter(jitter, values, color=color, alpha=0.72, s=26, zorder=2)
        mean = statistics.fmean(values)
        lower, upper = _bootstrap_mean_interval(values, BOOTSTRAP_SEED + position)
        agent_axes.errorbar(
            position,
            mean,
            yerr=((mean - lower,), (upper - mean,)),
            fmt="D",
            color="#222222",
            markerfacecolor=color,
            markersize=6,
            capsize=4,
            linewidth=1.4,
            zorder=3,
        )
    agent_axes.set_xticks((1, 2), ("Without Wiki", "With Wiki"))
    agent_axes.set_ylabel(f"Final hits out of {budget}")
    agent_axes.set_title("Agent campaigns", fontweight="bold")
    _style_axes(agent_axes)

    for position, key in enumerate(("random", "active_learning"), start=1):
        values = tuple(campaign[-1] for campaign in by_key[key].campaigns)
        mean = statistics.fmean(values)
        lower, upper = _bootstrap_mean_interval(
            values,
            BOOTSTRAP_SEED + 10 + position,
        )
        reference_axes.errorbar(
            position,
            mean,
            yerr=((mean - lower,), (upper - mean,)),
            fmt="o",
            color=APPROACH_COLORS[key],
            capsize=4,
            linewidth=1.5,
        )
    reference_axes.set_xticks((1, 2), ("Random", "Active\nlearning"))
    reference_axes.set_title("Computational references", fontweight="bold")
    reference_axes.tick_params(labelleft=False)
    _style_axes(reference_axes)
    combined = [
        campaign[-1]
        for approach in by_key.values()
        for campaign in approach.campaigns
    ]
    lower_limit = max(0, min(combined) - 3)
    upper_limit = max(combined) + 3
    agent_axes.set_ylim(lower_limit, upper_limit)
    reference_axes.set_ylim(lower_limit, upper_limit)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_manuscript_threshold_difference(
    approaches: Sequence[Approach],
    output: Path,
    *,
    threshold: float,
) -> None:
    """Plot the between-agent contrast over the fixed threshold range."""
    by_key = _required_approaches(approaches)
    thresholds = tuple(35.0 + index * 0.25 for index in range(61))

    def threshold_rows(key: str) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(
                float(
                    sum(
                        measurement.surface_tension < candidate
                        for measurement in campaign.measurements
                    ),
                )
                for candidate in thresholds
            )
            for campaign in by_key[key].raw_campaigns
        )

    with_rows = threshold_rows("wiki_agent")
    without_rows = threshold_rows("no_wiki_agent")
    difference = tuple(
        with_value - without_value
        for with_value, without_value in zip(
            mean_trajectory(with_rows),
            mean_trajectory(without_rows),
            strict=True,
        )
    )
    lower, upper = _bootstrap_difference_band(
        with_rows,
        without_rows,
        seed=BOOTSTRAP_SEED,
    )
    figure, axes = plt.subplots(figsize=(7.0, 3.6))
    axes.fill_between(
        thresholds,
        lower,
        upper,
        color="#6F4E7C",
        alpha=0.18,
        linewidth=0,
    )
    axes.plot(thresholds, difference, color="#6F4E7C", linewidth=2.3)
    axes.axhline(0, color="#555555", linewidth=0.9)
    axes.axvline(threshold, color="#333333", linewidth=1.0, linestyle=":")
    axes.annotate(
        "Fixed threshold",
        (threshold, axes.get_ylim()[1]),
        xytext=(4, -3),
        textcoords="offset points",
        va="top",
        fontsize=10,
    )
    axes.set_xlabel("Exclusive hit threshold (mN/m)")
    axes.set_ylabel("Final hits, with Wiki minus without Wiki")
    axes.set_xlim(35, 50)
    _style_axes(axes)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_manuscript_best(
    approaches: Sequence[Approach],
    output: Path,
    *,
    budget: int,
) -> None:
    """Plot median running minima with interquartile bands."""
    by_key = _required_approaches(approaches)
    figure, axes = plt.subplots(figsize=(7.0, 3.8))
    x_values = range(1, budget + 1)
    for index, key in enumerate(KNOWN_APPROACH_ORDER):
        approach = by_key[key]
        rows = []
        for campaign in approach.raw_campaigns:
            running = []
            best = math.inf
            for measurement in campaign.measurements:
                best = min(best, measurement.surface_tension)
                running.append(best)
            rows.append(tuple(running))
        median = tuple(
            statistics.median(row[sample - 1] for row in rows)
            for sample in x_values
        )
        lower = tuple(
            _percentile(sorted(row[sample - 1] for row in rows), 0.25)
            for sample in x_values
        )
        upper = tuple(
            _percentile(sorted(row[sample - 1] for row in rows), 0.75)
            for sample in x_values
        )
        agent = _is_agent_approach(approach)
        color = _approach_color(approach, index)
        axes.fill_between(
            x_values,
            lower,
            upper,
            step="post",
            color=color,
            alpha=0.15 if agent else 0.06,
            linewidth=0,
        )
        axes.step(
            x_values,
            median,
            where="post",
            color=color,
            linestyle=_approach_linestyle(approach, index),
            linewidth=2.5 if agent else 1.4,
            alpha=1.0 if agent else 0.7,
            label=f"{approach.name} (n = {len(rows)})",
        )
    axes.annotate(
        "Lower is better",
        xy=(0.98, 0.96),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
    )
    axes.set_xlabel("Successful measurements")
    axes.set_ylabel("Best surface tension (mN/m)")
    axes.set_xlim(1, budget)
    handles, labels = axes.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=10,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    _style_axes(axes)
    figure.subplots_adjust(bottom=0.24)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_manuscript_selection_mechanism(  # ruff: ignore[too-many-locals]
    approaches: Sequence[Approach],
    output: Path,
    *,
    threshold: float,
) -> None:
    """Plot motif selection shares and their associated hit counts."""
    by_key = _required_approaches(approaches)
    motifs = (
        ("Proline-containing", lambda sequence: "P" in sequence),
        (
            "Lysine plus aromatic",
            lambda sequence: "K" in sequence
            and any(residue in sequence for residue in "FWY"),
        ),
    )
    windows = ((0, 20), (20, 40), (40, 60))
    window_labels = ("1-20", "21-40", "41-60")
    figure, axes_grid = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    rng = random.Random(  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        BOOTSTRAP_SEED,
    )
    markers = {"no_wiki_agent": "o", "wiki_agent": "^"}
    for row, (motif_name, matches) in enumerate(  # ruff: ignore[too-many-nested-blocks]
        motifs,
    ):
        for column, metric in enumerate(("Selection share", "Hits among selections")):
            axes = axes_grid[row][column]
            for approach_index, key in enumerate(
                ("no_wiki_agent", "wiki_agent"),
            ):
                offset = -0.15 if approach_index == 0 else 0.15
                for window_index, (start, stop) in enumerate(windows):
                    values = []
                    for campaign in by_key[key].raw_campaigns:
                        selected = [
                            measurement.surface_tension
                            for measurement in campaign.measurements[start:stop]
                            if matches(measurement.sequence)
                        ]
                        if column == 0:
                            values.append(len(selected) / (stop - start) * 100.0)
                        else:
                            values.append(
                                float(sum(value < threshold for value in selected)),
                            )
                    position = window_index + offset
                    jitter = [
                        position + rng.uniform(-0.045, 0.045) for _ in values
                    ]
                    axes.scatter(
                        jitter,
                        values,
                        color=APPROACH_COLORS[key],
                        marker=markers[key],
                        alpha=0.42,
                        s=18,
                        zorder=2,
                    )
                    mean = statistics.fmean(values)
                    lower, upper = _bootstrap_mean_interval(
                        values,
                        BOOTSTRAP_SEED
                        + row * 20
                        + column * 10
                        + approach_index * 3
                        + window_index,
                    )
                    axes.errorbar(
                        position,
                        mean,
                        yerr=((mean - lower,), (upper - mean,)),
                        fmt=markers[key],
                        color=APPROACH_COLORS[key],
                        markeredgecolor="#222222",
                        markeredgewidth=0.6,
                        markersize=6,
                        capsize=3,
                        linewidth=1.2,
                        label=(
                            by_key[key].name
                            if row == 0 and column == 0 and window_index == 0
                            else None
                        ),
                        zorder=3,
                    )
            if row == 0:
                axes.set_title(metric, fontweight="bold")
            if column == 0:
                axes.set_ylabel(f"{motif_name}\nCampaign percentage")
            else:
                axes.set_ylabel(f"{motif_name}\nHit count")
            axes.set_xticks(range(3), window_labels)
            _style_axes(axes)
    for axes in axes_grid[-1]:
        axes.set_xlabel("Successful-measurement window")
    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.subplots_adjust(bottom=0.17, hspace=0.34, wspace=0.29)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_additional_figures(
    approaches: Sequence[Approach],
    output: Path,
    *,
    budget: int,
    threshold: float,
) -> tuple[Path, ...]:
    """Write all companion figures and return their paths."""
    paths = tuple(_figure_path(output, name) for name in ADDITIONAL_FIGURE_NAMES)
    _plot_manuscript_final_counts(approaches, paths[0], budget=budget)
    _plot_manuscript_threshold_difference(
        approaches,
        paths[1],
        threshold=threshold,
    )
    _plot_manuscript_best(approaches, paths[2], budget=budget)
    _plot_manuscript_selection_mechanism(
        approaches,
        paths[3],
        threshold=threshold,
    )
    return paths


def plot_observations(
    input_dir: Path,
    output: Path,
    *,
    budget: int,
    threshold: float,
) -> tuple[Approach, ...]:
    """Load campaign data and write the comparison figure.

    Args:
        input_dir: The directory that contains approach subdirectories.
        output: The figure path. Its extension selects the output format.
        budget: The successful-measurement budget per campaign.
        threshold: The exclusive hit threshold in mN/m.

    Returns:
        The approaches included in the figure.

    """
    approaches = discover_approaches(input_dir, budget=budget, threshold=threshold)
    output.parent.mkdir(parents=True, exist_ok=True)

    style = mpl.RcParams(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Liberation Serif",
                "TeX Gyre Termes",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    with plt.rc_context(style):
        _plot_manuscript_primary(approaches, output, budget=budget)
        _plot_additional_figures(
            approaches,
            output,
            budget=budget,
            threshold=threshold,
        )
    return approaches


def _positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the observation-plot command-line interface."""
    parser = argparse.ArgumentParser(
        description="Plot cumulative hits across sampling approaches.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=_positive_integer, default=DEFAULT_BUDGET)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    try:
        approaches = plot_observations(
            args.input_dir,
            args.output,
            budget=args.budget,
            threshold=args.threshold,
        )
    except (CampaignDataError, OSError, ValueError) as exc:
        parser.error(str(exc))
    for approach in approaches:
        campaign_count = len(approach.campaigns)
        suffix = "campaign" if campaign_count == 1 else "campaigns"
        sys.stdout.write(f"{approach.name}: {campaign_count} {suffix}\n")
    comparison = primary_comparison(approaches)
    if comparison is not None:
        lower, upper = comparison.bootstrap_interval
        sys.stdout.write("Primary Wiki-access comparison (Wiki - no Wiki):\n")
        sys.stdout.write(
            f"  Mean scores: {comparison.with_wiki_mean:.3f} vs "
            f"{comparison.without_wiki_mean:.3f} hits\n",
        )
        sys.stdout.write(
            f"  Mean difference: {comparison.mean_difference:.3f} hits\n",
        )
        sys.stdout.write(
            f"  95% percentile bootstrap CI: [{lower:.3f}, {upper:.3f}] "
            f"({BOOTSTRAP_RESAMPLES:,} resamples, seed {BOOTSTRAP_SEED})\n",
        )
        sys.stdout.write(
            "  Exact two-sided permutation test: "
            f"p={comparison.permutation_p_value:.6f} "
            f"({comparison.extreme_assignments:,}/"
            f"{comparison.total_assignments:,} assignments)\n",
        )
    sys.stdout.write(f"Wrote {args.output}\n")
    for name in ADDITIONAL_FIGURE_NAMES:
        sys.stdout.write(f"Wrote {_figure_path(args.output, name)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
