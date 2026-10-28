import logging
from io import StringIO

import pytest

from wiki_agents.logging_config import (
    QUIET_LOGGERS,
    ConsoleFormatter,
    ProjectLogFilter,
    configure_logging,
)


def test_configure_logging_keeps_app_info_and_quiets_dependencies(
    monkeypatch,
) -> None:
    basic_config_calls = []
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("THIRD_PARTY_LOG_LEVEL", raising=False)
    monkeypatch.setattr(
        logging,
        "basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )

    affected_loggers = ("wiki_agents", *QUIET_LOGGERS)
    for logger_name in affected_loggers:
        logger = logging.getLogger(logger_name)
        monkeypatch.setattr(logger, "level", logger.level)

    configure_logging()

    assert basic_config_calls[0]["level"] == "INFO"
    assert basic_config_calls[0]["force"] is True
    formatter = basic_config_calls[0]["handlers"][0].formatter
    assert isinstance(formatter, ConsoleFormatter)
    assert logging.getLogger("wiki_agents").level == logging.INFO
    assert all(
        logging.getLogger(logger_name).level == logging.WARNING
        for logger_name in QUIET_LOGGERS
    )


def test_configure_logging_honors_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("THIRD_PARTY_LOG_LEVEL", "ERROR")
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)

    affected_loggers = ("wiki_agents", *QUIET_LOGGERS)
    for logger_name in affected_loggers:
        logger = logging.getLogger(logger_name)
        monkeypatch.setattr(logger, "level", logger.level)

    configure_logging()

    assert logging.getLogger("wiki_agents").level == logging.DEBUG
    assert all(
        logging.getLogger(logger_name).level == logging.ERROR
        for logger_name in QUIET_LOGGERS
    )


def test_explicit_levels_take_precedence_over_environment(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("THIRD_PARTY_LOG_LEVEL", "CRITICAL")
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)

    affected_loggers = ("wiki_agents", *QUIET_LOGGERS)
    for logger_name in affected_loggers:
        logger = logging.getLogger(logger_name)
        monkeypatch.setattr(logger, "level", logger.level)

    configure_logging(level="DEBUG", third_party_level="INFO")

    assert logging.getLogger("wiki_agents").level == logging.DEBUG
    assert all(logging.getLogger(name).level == logging.INFO for name in QUIET_LOGGERS)


def test_project_log_filter_keeps_app_info_and_dependency_warnings() -> None:
    log_filter = ProjectLogFilter("WARNING")

    assert log_filter.filter(_record("wiki_agents.orchestrator", logging.INFO))
    assert not log_filter.filter(_record("httpx2", logging.INFO))
    assert log_filter.filter(_record("httpx2", logging.WARNING))


def test_console_formatter_is_compact_and_preserves_log_record() -> None:
    record = logging.LogRecord(
        "wiki_agents.orchestrator",
        logging.INFO,
        __file__,
        1,
        "Processed %d pages",
        (3,),
        None,
    )

    output = ConsoleFormatter().format(record)

    assert output.endswith("INFO     orchestrator  Processed 3 pages")
    assert record.name == "wiki_agents.orchestrator"
    assert not hasattr(record, "level_color")


def test_console_formatter_colors_level_when_requested() -> None:
    record = logging.LogRecord(
        "dependency.client",
        logging.WARNING,
        __file__,
        1,
        "Request was retried",
        (),
        None,
    )

    output = ConsoleFormatter(color=True).format(record)

    assert "\x1b[33mWARNING " in output
    assert "dependency.client  Request was retried" in output


@pytest.mark.parametrize(
    ("environment", "isatty", "expected"),
    [
        ({}, False, False),
        ({}, True, True),
        ({"FORCE_COLOR": "1"}, False, True),
        ({"FORCE_COLOR": "0"}, True, True),
        ({"NO_COLOR": "1", "FORCE_COLOR": "1"}, True, False),
    ],
)
def test_configure_logging_resolves_color_environment(
    monkeypatch,
    environment,
    isatty,
    expected,
) -> None:
    basic_config_calls = []
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    stream = StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: isatty)
    original_handler = logging.StreamHandler
    monkeypatch.setattr(logging, "StreamHandler", lambda: original_handler(stream))
    monkeypatch.setattr(
        logging,
        "basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )

    configure_logging()

    formatter = basic_config_calls[0]["handlers"][0].formatter
    assert isinstance(formatter, ConsoleFormatter)
    assert formatter.color is expected


def test_configure_logging_explicit_color_overrides_environment(monkeypatch) -> None:
    basic_config_calls = []
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr(
        logging,
        "basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )

    configure_logging(color=False)

    formatter = basic_config_calls[0]["handlers"][0].formatter
    assert isinstance(formatter, ConsoleFormatter)
    assert not formatter.color


def _record(name: str, level: int) -> logging.LogRecord:
    """Create a minimal record for logging filter tests."""
    return logging.LogRecord(name, level, __file__, 1, "message", (), None)
