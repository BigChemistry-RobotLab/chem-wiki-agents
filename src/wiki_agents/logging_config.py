"""Shared logging configuration for wiki-agents entry points.

The module deliberately avoids the name ``logging.py`` so executing a module from
this package as a script cannot shadow Python's standard-library ``logging`` module.
"""

from __future__ import annotations

import logging
import os
from typing import TextIO, override

DEFAULT_FORMAT = "%(asctime)s  %(levelname)-8s %(name)s  %(message)s"
DEFAULT_DATE_FORMAT = "%H:%M:%S"

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_LEVEL_COLORS = {
    logging.DEBUG: _DIM,
    logging.INFO: "\x1b[36m",
    logging.WARNING: "\x1b[33m",
    logging.ERROR: "\x1b[31m",
    logging.CRITICAL: "\x1b[1;31m",
}
_COLOR_FORMAT = (
    f"{_DIM}%(asctime)s{_RESET}  "
    f"%(level_color)s%(levelname)-8s{_RESET} "
    "%(name)s  %(message)s"
)

# These libraries log routine network and tracing activity at INFO. Their warnings
# and errors remain visible unless the caller explicitly chooses another level.
QUIET_LOGGERS = (
    "httpx",
    "httpx2",
    "httpcore",
    "weave",
    "wandb",
)


class ProjectLogFilter(logging.Filter):
    """Allow project records and significant dependency records."""

    def __init__(self, third_party_level: str | int) -> None:
        super().__init__()
        self.third_party_level = _level_number(third_party_level)

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Keep all project logs and dependency logs at the configured level."""
        is_project_log = record.name == "wiki_agents" or record.name.startswith(
            "wiki_agents.",
        )
        return is_project_log or record.levelno >= self.third_party_level


class ConsoleFormatter(logging.Formatter):
    """Render readable console logs without leaking formatting into log records."""

    def __init__(self, *, color: bool = False) -> None:
        super().__init__(
            _COLOR_FORMAT if color else DEFAULT_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )
        self.color = color

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Use a concise app logger name and optionally color the severity."""
        original_name = record.name
        record_attributes = vars(record)
        had_level_color = "level_color" in record_attributes
        original_level_color = record_attributes.get("level_color")
        try:
            record.name = _display_name(record.name)
            record_attributes["level_color"] = (
                _LEVEL_COLORS.get(record.levelno, "") if self.color else ""
            )
            return super().format(record)
        finally:
            record.name = original_name
            if had_level_color:
                record_attributes["level_color"] = original_level_color
            else:
                del record_attributes["level_color"]


def configure_logging(
    level: str | int | None = None,
    *,
    third_party_level: str | int | None = None,
    color: bool | None = None,
) -> None:
    """Configure polished application logs and quiet chatty dependencies.

    ``LOG_LEVEL`` controls wiki-agents and the root logger (default: ``INFO``).
    ``THIRD_PARTY_LOG_LEVEL`` independently controls routine dependency output
    (default: ``WARNING``). Color is enabled automatically for interactive
    terminals. ``NO_COLOR`` disables it and ``FORCE_COLOR`` enables it when output
    is redirected; an explicit ``color`` argument takes precedence over both.
    """
    application_level = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    dependency_level = (
        third_party_level
        if third_party_level is not None
        else os.getenv("THIRD_PARTY_LOG_LEVEL", "WARNING")
    )

    handler = logging.StreamHandler()
    handler.addFilter(ProjectLogFilter(dependency_level))
    handler.setFormatter(
        ConsoleFormatter(color=_color_enabled(handler.stream, requested=color)),
    )
    logging.basicConfig(
        level=application_level,
        handlers=[handler],
        force=True,
    )
    logging.getLogger("wiki_agents").setLevel(application_level)

    for logger_name in QUIET_LOGGERS:
        logging.getLogger(logger_name).setLevel(dependency_level)


def _level_number(level: str | int) -> int:
    """Convert a standard logging level to its numeric value."""
    if isinstance(level, int):
        return level
    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        msg = f"Unknown logging level: {level!r}"
        raise ValueError(msg)
    return numeric_level


def _display_name(name: str) -> str:
    """Shorten application logger names while leaving dependencies identifiable."""
    if name == "wiki_agents":
        return "wiki-agent"
    if name.startswith("wiki_agents."):
        return name.removeprefix("wiki_agents.")
    return name


def _color_enabled(stream: TextIO, *, requested: bool | None) -> bool:
    """Resolve explicit, conventional environment, and terminal color settings."""
    if requested is not None:
        return requested
    if "NO_COLOR" in os.environ:
        return False
    if os.getenv("FORCE_COLOR", "") not in {"", "0"}:
        return True
    return stream.isatty() if hasattr(stream, "isatty") else False
