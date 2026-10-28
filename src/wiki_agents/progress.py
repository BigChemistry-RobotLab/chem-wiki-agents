"""Report opt-in progress for long-running user operations."""

from __future__ import annotations

import logging
from threading import Event, Lock, Thread, current_thread
from time import monotonic
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from types import TracebackType

_MIN_UPDATE_SECONDS = 5.0


class ProgressReporter:
    """Create progress tasks that write periodic application logs."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        logger: logging.Logger | None = None,
        heartbeat_seconds: float = 30.0,
    ) -> None:
        """Create a reporter.

        Args:
            enabled: Whether tasks write progress logs.
            logger: Logger that receives progress records.
            heartbeat_seconds: Maximum time between active task records.

        """
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.enabled = enabled
        self._logger = logger or logging.getLogger("wiki_agents.progress")
        self._heartbeat_seconds = heartbeat_seconds

    def task(
        self,
        description: str,
        total: int,
        *,
        completed: int = 0,
    ) -> ProgressTask:
        """Create and start one progress task.

        Args:
            description: Present-participle description of the operation.
            total: Total number of work items.
            completed: Number of items already complete.

        Returns:
            A started task that supports context manager use.

        """
        return ProgressTask(
            description,
            total,
            completed=completed,
            enabled=self.enabled,
            logger=self._logger,
            heartbeat_seconds=self._heartbeat_seconds,
        )


class ProgressTask:
    """Track one bounded operation and emit thread-safe progress records."""

    def __init__(  # ruff: ignore[too-many-arguments] - Task state has fixed reporting inputs.
        self,
        description: str,
        total: int,
        *,
        completed: int,
        enabled: bool,
        logger: logging.Logger,
        heartbeat_seconds: float,
    ) -> None:
        """Start one bounded progress task."""
        if total < 0:
            raise ValueError("progress total must not be negative")
        if not 0 <= completed <= total:
            raise ValueError("completed progress must be between zero and total")
        self.description = description
        self.total = total
        self._completed = completed
        self._enabled = enabled
        self._logger = logger
        self._heartbeat_seconds = heartbeat_seconds
        self._started_at = monotonic()
        self._last_logged_at = self._started_at
        self._last_logged_completed = completed
        self._lock = Lock()
        self._stop = Event()
        self._closed = False
        self._thread: Thread | None = None
        if enabled:
            self._log_locked()
            if completed < total:
                self._thread = Thread(
                    target=self._heartbeat,
                    name="wiki-progress",
                    daemon=True,
                )
                self._thread.start()

    def __enter__(self) -> Self:
        """Return this active task."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the task and report its final state."""
        self.close()

    def advance(self, amount: int = 1) -> None:
        """Add completed work and report useful milestones.

        Args:
            amount: Positive number of newly completed items.

        """
        if amount < 1:
            raise ValueError("progress advance must be positive")
        if not self._enabled:
            return
        with self._lock:
            if self._closed:
                return
            self._completed = min(self.total, self._completed + amount)
            now = monotonic()
            milestone = max(1, self.total // 100)
            if (
                self._completed == self.total
                or self._completed - self._last_logged_completed >= milestone
                or now - self._last_logged_at >= _MIN_UPDATE_SECONDS
            ):
                self._log_locked(now)

    def update(self, completed: int) -> None:
        """Set the absolute completed count.

        Args:
            completed: Current completed item count.

        """
        if completed < 0:
            raise ValueError("completed progress must not be negative")
        if not self._enabled:
            return
        with self._lock:
            amount = min(completed, self.total) - self._completed
        if amount > 0:
            self.advance(amount)

    def set_description(self, description: str) -> None:
        """Set the description that the next progress record uses."""
        with self._lock:
            self.description = description

    def close(self) -> None:
        """Stop heartbeat reporting and write a final record when required."""
        if not self._enabled:
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=1.0)
        with self._lock:
            if self._closed:
                return
            if self._completed != self._last_logged_completed:
                self._log_locked()
            self._closed = True

    def _heartbeat(self) -> None:
        """Write a progress record while the current work item is active."""
        while not self._stop.wait(self._heartbeat_seconds):
            with self._lock:
                if self._closed:
                    return
                self._log_locked()

    def _log_locked(self, now: float | None = None) -> None:
        """Write one record while the task lock is held."""
        current_time = monotonic() if now is None else now
        percent = 100.0 if self.total == 0 else self._completed / self.total * 100
        self._logger.info(
            "Progress: %s %d/%d (%.1f%%), elapsed %.1fs",
            self.description,
            self._completed,
            self.total,
            percent,
            current_time - self._started_at,
        )
        self._last_logged_at = current_time
        self._last_logged_completed = self._completed
