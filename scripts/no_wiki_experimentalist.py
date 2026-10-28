"""Run a persistent experimental campaign without Wiki access."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from wiki_agents.logging_config import configure_logging
from wiki_agents.orchestrator.experimentalist import (
    ExperimentalistOrchestrator,
    ExperimentalistOrchestratorConfig,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class NoWikiExperimentalistOrchestrator(ExperimentalistOrchestrator):
    """Name the shared orchestrator for the no-Wiki command."""


def build_parser() -> argparse.ArgumentParser:
    """Build the no-Wiki experimentalist argument parser."""
    parser = argparse.ArgumentParser(
        description="Run a persistent experimental campaign without Wiki access",
    )
    parser.add_argument("campaign_path", type=Path, help="Campaign artifact directory")
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--task", help="Campaign objective text")
    task.add_argument("--task-file", type=Path, help="UTF-8 campaign objective file")
    parser.add_argument("--lab-url", help="Lab API URL; defaults to LAB_API_URL")
    parser.add_argument("--request-budget", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--researcher-turns", type=int, default=30)
    parser.add_argument("--researcher-model", default="openai/gpt-5.6-luna")
    parser.add_argument("--curator-model", default="openai/gpt-5.6-luna")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse configuration and run a no-Wiki experimental campaign."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        task = (
            args.task
            if args.task is not None
            else args.task_file.read_text(encoding="utf-8")
        )
        config = ExperimentalistOrchestratorConfig(
            wiki_path=None,
            campaign_path=args.campaign_path,
            task=task,
            lab_url=args.lab_url,
            request_budget=args.request_budget,
            batch_size=args.batch_size,
            researcher_turns=args.researcher_turns,
            researcher_model=args.researcher_model,
            curator_model=args.curator_model,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    configure_logging()
    NoWikiExperimentalistOrchestrator(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
