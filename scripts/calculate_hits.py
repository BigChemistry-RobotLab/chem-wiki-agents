"""Report surface-tension campaign performance from a JSONL ledger."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class Measurement:
    """Store one usable campaign measurement."""

    sequence: str | None
    ph: float | None
    surface_tension: float

    @property
    def condition(self) -> tuple[str | None, float | None]:
        """The request condition."""
        return self.sequence, self.ph


@dataclass(frozen=True, slots=True)
class CampaignAnalysis:
    """Store campaign measurements and ledger-quality counts."""

    path: Path
    threshold: float
    physical_lines: int
    blank_lines: int
    invalid_json_lines: int
    missing_measurement_lines: int
    measurements: tuple[Measurement, ...]

    @property
    def hits(self) -> tuple[Measurement, ...]:
        """The measurements below the threshold."""
        return tuple(
            measurement
            for measurement in self.measurements
            if measurement.surface_tension < self.threshold
        )

    @property
    def unique_condition_count(self) -> int:
        """The number of distinct request conditions."""
        return len({measurement.condition for measurement in self.measurements})

    @property
    def duplicate_count(self) -> int:
        """The number of repeated request conditions."""
        return len(self.measurements) - self.unique_condition_count


def _as_object(value: object) -> dict[str, object] | None:
    """Narrow a JSON value to an object with string keys."""
    if not isinstance(value, dict):
        return None
    return cast("dict[str, object]", value)


def _finite_number(value: object) -> float | None:
    """Return a finite JSON number as a float."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _measurement_from_record(record: object) -> Measurement | None:
    """Extract one measurement from a decoded ledger record."""
    record_data = _as_object(record)
    if record_data is None:
        return None
    result = _as_object(record_data.get("result"))
    values = _as_object(result.get("values")) if result is not None else None
    if values is None:
        return None
    surface_tension = _finite_number(values.get("surface_tension"))
    if surface_tension is None:
        return None

    request = _as_object(record_data.get("request"))
    sequence_value = request.get("sequence") if request is not None else None
    sequence = sequence_value if isinstance(sequence_value, str) else None
    ph = _finite_number(request.get("ph")) if request is not None else None
    return Measurement(sequence=sequence, ph=ph, surface_tension=surface_tension)


def analyze_surface_tension(
    file_path: str | Path,
    threshold: float,
) -> CampaignAnalysis:
    """Analyze a campaign observation ledger.

    Args:
        file_path: The JSONL ledger path.
        threshold: The exclusive hit threshold.

    Returns:
        Campaign measurements and ledger-quality counts.

    Raises:
        ValueError: If the threshold is not finite.

    """
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")

    path = Path(file_path)
    physical_lines = 0
    blank_lines = 0
    invalid_json_lines = 0
    missing_measurement_lines = 0
    measurements: list[Measurement] = []

    with path.open("r", encoding="utf-8") as ledger:
        for line in ledger:
            physical_lines += 1
            stripped_line = line.strip()
            if not stripped_line:
                blank_lines += 1
                continue
            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError:
                invalid_json_lines += 1
                continue
            measurement = _measurement_from_record(record)
            if measurement is None:
                missing_measurement_lines += 1
                continue
            measurements.append(measurement)

    return CampaignAnalysis(
        path=path,
        threshold=threshold,
        physical_lines=physical_lines,
        blank_lines=blank_lines,
        invalid_json_lines=invalid_json_lines,
        missing_measurement_lines=missing_measurement_lines,
        measurements=tuple(measurements),
    )


def _describe(values: Sequence[float]) -> list[str]:
    """Format distribution statistics for a nonempty value sequence."""
    lines = [
        f"  Minimum:              {min(values):.6f}",
        f"  Mean:                 {statistics.fmean(values):.6f}",
        f"  Median:               {statistics.median(values):.6f}",
        f"  Maximum:              {max(values):.6f}",
    ]
    if len(values) > 1:
        lines.append(f"  Standard deviation:   {statistics.stdev(values):.6f}")
    return lines


def _format_condition(measurement: Measurement) -> str:
    """Format one measured condition."""
    sequence = measurement.sequence or "unknown"
    ph = f"{measurement.ph:g}" if measurement.ph is not None else "unknown"
    return (
        f"sequence={sequence}, pH={ph}, "
        f"surface_tension={measurement.surface_tension:.6f}"
    )


def format_report(analysis: CampaignAnalysis, *, top: int = 5) -> str:
    """Create a readable campaign report.

    Args:
        analysis: The analyzed campaign.
        top: The maximum number of ranked conditions to show.

    Returns:
        A newline-terminated text report.

    Raises:
        ValueError: If top is negative.

    """
    if top < 0:
        raise ValueError("Top count cannot be negative")

    measurements = analysis.measurements
    hits = analysis.hits
    measurement_count = len(measurements)
    hit_rate = len(hits) / measurement_count * 100 if measurements else 0.0
    nonblank_lines = analysis.physical_lines - analysis.blank_lines
    lines = [
        "Surface-tension campaign report",
        f"Input: {analysis.path}",
        f"Hit criterion: surface_tension < {analysis.threshold:g} mN/m",
        "",
        "Coverage",
        f"  Usable measurements:  {measurement_count} / {nonblank_lines} records",
        f"  Unique conditions:    {analysis.unique_condition_count}",
        f"  Repeated conditions:  {analysis.duplicate_count}",
        f"  Hits:                 {len(hits)} ({hit_rate:.1f}%)",
    ]

    skipped_count = analysis.invalid_json_lines + analysis.missing_measurement_lines
    if analysis.blank_lines or skipped_count:
        lines.extend(
            [
                "",
                "Ledger quality",
                f"  Blank lines:           {analysis.blank_lines}",
                f"  Invalid JSON:          {analysis.invalid_json_lines}",
                f"  Missing/invalid value: {analysis.missing_measurement_lines}",
            ],
        )

    if measurements:
        values = [measurement.surface_tension for measurement in measurements]
        lines.extend(["", "All measurements (mN/m)", *_describe(values)])
        if hits:
            hit_values = [measurement.surface_tension for measurement in hits]
            lines.extend(["", "Hits (mN/m)", *_describe(hit_values)])

        ranked = sorted(
            measurements,
            key=lambda measurement: (
                measurement.surface_tension,
                measurement.sequence or "",
                measurement.ph if measurement.ph is not None else math.inf,
            ),
        )
        shown_count = min(top, len(ranked))
        if shown_count:
            lines.extend(["", f"Top {shown_count} conditions"])
            lines.extend(
                f"  {rank}. {_format_condition(measurement)}"
                for rank, measurement in enumerate(ranked[:shown_count], start=1)
            )
    else:
        lines.extend(["", "No usable surface-tension measurements were found."])

    return "\n".join(lines) + "\n"


def _nonnegative_integer(value: str) -> int:
    """Parse a nonnegative command-line integer."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    """Run the campaign-report command-line interface."""
    parser = argparse.ArgumentParser(
        description="Report surface-tension campaign performance from JSONL.",
    )
    parser.add_argument("input_file", type=Path, help="JSONL observation ledger.")
    parser.add_argument(
        "threshold",
        type=float,
        help="Exclusive surface-tension hit threshold in mN/m.",
    )
    parser.add_argument(
        "--top",
        type=_nonnegative_integer,
        default=5,
        help="Number of best conditions to list (default: 5; 0 hides the list).",
    )
    args = parser.parse_args(argv)

    try:
        analysis = analyze_surface_tension(args.input_file, args.threshold)
        report = format_report(analysis, top=args.top)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    sys.stdout.write(report)


if __name__ == "__main__":
    main()
