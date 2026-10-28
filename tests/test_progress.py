import logging

import pytest

from wiki_agents.progress import ProgressReporter


def test_disabled_progress_reporter_does_not_log(caplog) -> None:
    reporter = ProgressReporter(enabled=False)

    with reporter.task("processing items", 2) as progress:
        progress.advance()
        progress.advance()

    assert not caplog.records


def test_progress_reporter_logs_counts_and_elapsed_time(caplog) -> None:
    caplog.set_level(logging.INFO, logger="wiki_agents.progress")
    reporter = ProgressReporter(enabled=True, heartbeat_seconds=60)

    with reporter.task("processing items", 2) as progress:
        progress.advance()
        progress.advance()

    messages = [record.getMessage() for record in caplog.records]
    assert messages[0].startswith(
        "Progress: processing items 0/2 (0.0%), elapsed ",
    )
    assert any("processing items 1/2 (50.0%)" in message for message in messages)
    assert any("processing items 2/2 (100.0%)" in message for message in messages)


def test_progress_reporter_includes_checkpointed_work(caplog) -> None:
    caplog.set_level(logging.INFO, logger="wiki_agents.progress")
    reporter = ProgressReporter(enabled=True, heartbeat_seconds=60)

    with reporter.task("resuming items", 10, completed=7) as progress:
        progress.update(9)
        progress.update(10)

    messages = [record.getMessage() for record in caplog.records]
    assert "resuming items 7/10 (70.0%)" in messages[0]
    assert any("resuming items 10/10 (100.0%)" in message for message in messages)


def test_progress_reporter_uses_updated_description(caplog) -> None:
    caplog.set_level(logging.INFO, logger="wiki_agents.progress")
    reporter = ProgressReporter(enabled=True, heartbeat_seconds=60)

    with reporter.task("processing [accepted 0, failed 0]", 1) as progress:
        progress.set_description("processing [accepted 0, failed 1]")
        progress.advance()

    messages = [record.getMessage() for record in caplog.records]
    assert messages[-1].startswith(
        "Progress: processing [accepted 0, failed 1] 1/1 (100.0%), elapsed ",
    )


@pytest.mark.parametrize(("total", "completed"), [(-1, 0), (1, -1), (1, 2)])
def test_progress_reporter_rejects_invalid_bounds(total, completed) -> None:
    reporter = ProgressReporter(enabled=True)

    with pytest.raises(ValueError, match=r"progress|completed"):
        reporter.task("invalid", total, completed=completed)
