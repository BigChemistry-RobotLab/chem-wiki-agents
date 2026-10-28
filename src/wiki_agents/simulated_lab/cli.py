"""Command-line entry point for the simulated lab."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import uvicorn

from wiki_agents.simulated_lab.lab import load_lab
from wiki_agents.simulated_lab.server import create_app

if TYPE_CHECKING:
    from collections.abc import Sequence


def _mapping(value: str) -> tuple[str, str]:
    """Parse a NAME=VALUE command-line mapping."""
    name, separator, target = value.partition("=")
    if not separator or not name or not target:
        raise argparse.ArgumentTypeError("must use non-empty NAME=VALUE syntax")
    return name, target


def build_parser() -> argparse.ArgumentParser:
    """Build the start-lab argument parser."""
    parser = argparse.ArgumentParser(description="Start a CSV-backed simulated lab")
    parser.add_argument("dataset", help="Experiment dataset CSV")
    parser.add_argument("--input", action="append", default=[], dest="inputs")
    parser.add_argument("--output", action="append", default=[], dest="outputs")
    parser.add_argument("--alias", action="append", default=[], type=_mapping)
    parser.add_argument(
        "--column-type",
        action="append",
        default=[],
        type=_mapping,
        metavar="COLUMN=TYPE",
    )
    parser.add_argument("--materials", help="Optional materials CSV")
    parser.add_argument(
        "--material-field",
        action="append",
        default=[],
        dest="material_fields",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--max-batch-size", default=100, type=int)
    return parser


def _unique_mapping(
    parser: argparse.ArgumentParser,
    values: list[tuple[str, str]],
    label: str,
) -> dict[str, str]:
    """Convert repeated mappings and reject duplicate names."""
    result: dict[str, str] = {}
    for name, value in values:
        if name in result:
            parser.error(f"duplicate {label} mapping: {name}")
        result[name] = value
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Load a lab configuration and run Uvicorn."""
    parser = build_parser()
    args = parser.parse_args(argv)
    aliases = _unique_mapping(parser, args.alias, "alias")
    column_types = _unique_mapping(parser, args.column_type, "column type")
    inputs = args.inputs
    outputs = args.outputs
    if not inputs and not outputs:
        inputs = ["sequence", "pH"]
        outputs = ["surface_tension(mN/m)"]
        column_types = {"pH": "number", **column_types}
        aliases = {
            "ph": "pH",
            "surface_tension": "surface_tension(mN/m)",
            **aliases,
        }
    try:
        lab = load_lab(
            args.dataset,
            inputs=inputs,
            outputs=outputs,
            aliases=aliases,
            column_types=column_types,
            materials_path=args.materials,
            material_fields=args.material_fields,
            max_batch_size=args.max_batch_size,
        )
    except ValueError as exc:
        parser.error(str(exc))
    uvicorn.run(create_app(lab), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
