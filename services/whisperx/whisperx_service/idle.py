"""Idle-timeout management for ML engine singletons.

Provides ``IdleMixin`` (adds idle tracking + a status phase to an
engine class) and ``idle_checker`` (async context manager that polls
for idle timeout in a background task).

The status phase vocabulary mirrors llama-server's router mode:

- ``unloaded`` — never loaded or explicitly unloaded
- ``loading``  — currently being loaded
- ``loaded``   — resident and ready to serve requests
- ``sleeping`` — auto-unloaded due to idle timeout; will reload lazily

Usage in engine.py::

    class MyEngine(IdleMixin):
        @property
        def is_loaded(self) -> bool: ...
        def unload(self) -> None: ...          # manual unload → unloaded

        def load(self, name: str) -> None:
            self.mark_loading()
            try:
                ...  # actually load weights
            except Exception:
                self.mark_unloaded()
                raise
            self.mark_loaded()
            self.touch()

        def do_inference(self, ...) -> ...:
            self.touch()                        # reset idle timer
            ...

Usage in app.py lifespan::

    async with idle_checker(engine, settings.model_idle_timeout):
        yield
    engine.unload()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

_IDLE_CHECK_INTERVAL = 30  # seconds between idle checks

StatusPhase = Literal["unloaded", "loading", "loaded", "sleeping"]


class IdleMixin:
    """Mixin that adds idle-timeout tracking and a status phase.

    The host class must define:

    - ``is_loaded`` — bool property indicating whether any model is loaded
    - ``unload()`` — method to release all loaded models

    Call ``touch()`` after each inference to reset the idle timer.
    Call ``mark_loading()`` / ``mark_loaded()`` / ``mark_unloaded()``
    around load/unload transitions so that ``status_phase`` stays in
    sync with reality.

    The distinction between ``unloaded`` (manual / never loaded) and
    ``sleeping`` (auto-unloaded due to idle) is driven by whether the
    last unload came from ``unload_if_idle`` or from an explicit call.
    """

    _last_used: float = 0.0
    _status_phase: StatusPhase = "unloaded"

    # -- Required interface (provided by host class) --------------------------
    # These raise NotImplementedError so that forgetting to override them
    # in the host class produces a clear error rather than a silent bug.

    @property
    def is_loaded(self) -> bool:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    # -- Status phase ---------------------------------------------------------

    @property
    def status_phase(self) -> StatusPhase:
        """Current lifecycle phase of the engine.

        Returns one of ``unloaded``, ``loading``, ``loaded``,
        ``sleeping``.  Updated via ``mark_loading`` / ``mark_loaded`` /
        ``mark_unloaded`` and ``unload_if_idle``.
        """
        return self._status_phase

    def mark_loading(self) -> None:
        """Record that a load is in progress."""
        self._status_phase = "loading"

    def mark_loaded(self) -> None:
        """Record that a load has completed successfully."""
        self._status_phase = "loaded"

    def mark_unloaded(self) -> None:
        """Record that the engine is explicitly unloaded (not sleeping)."""
        self._status_phase = "unloaded"
        self._last_used = 0.0

    def mark_sleeping(self) -> None:
        """Record that the engine was auto-unloaded by the idle checker."""
        self._status_phase = "sleeping"
        self._last_used = 0.0

    # -- Idle tracking --------------------------------------------------------

    def touch(self) -> None:
        """Reset the idle timer.  Call this after each inference."""
        self._last_used = time.monotonic()

    def idle_seconds(self) -> float:
        """Seconds since the last ``touch()``, or 0 if no model is loaded."""
        if not self.is_loaded or self._last_used == 0.0:
            return 0.0
        return time.monotonic() - self._last_used

    def unload_if_idle(self, timeout: float) -> bool:
        """Unload models if idle for at least *timeout* seconds.

        Transitions the engine into the ``sleeping`` phase on success
        (not ``unloaded``) so callers can distinguish automatic from
        manual unloads.  Returns ``True`` if models were unloaded.

        A *timeout* of ``<= 0`` disables idle unloading (matches
        llama-server's ``--sleep-idle-seconds`` convention).
        """
        if timeout <= 0 or not self.is_loaded:
            return False
        if self.idle_seconds() >= timeout:
            logger.info(
                "Idle for %.0fs (timeout=%ds), unloading models.",
                self.idle_seconds(), timeout,
            )
            self.unload()
            self.mark_sleeping()
            return True
        return False


@asynccontextmanager
async def idle_checker(engine: IdleMixin, timeout: int) -> AsyncIterator[None]:
    """Async context manager that polls *engine* for idle timeout.

    Starts a background ``asyncio`` task that calls
    ``engine.unload_if_idle(timeout)`` every 30 seconds.  The task is
    cancelled and awaited on exit.  If *timeout* is ``<= 0`` no task
    is started (matches llama-server's ``--sleep-idle-seconds``
    convention where ``-1`` / ``0`` disables the feature).
    """
    task: asyncio.Task[None] | None = None

    if timeout > 0:

        async def _poll() -> None:
            while True:
                await asyncio.sleep(_IDLE_CHECK_INTERVAL)
                engine.unload_if_idle(timeout)

        task = asyncio.create_task(_poll())

    try:
        yield
    finally:
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
