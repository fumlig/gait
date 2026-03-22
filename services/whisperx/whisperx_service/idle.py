"""Idle-timeout management for ML engine singletons.

Provides ``IdleMixin`` (adds idle tracking to an engine class) and
``idle_checker`` (async context manager that polls for idle timeout
in a background task).

Usage in engine.py::

    class MyEngine(IdleMixin):
        @property
        def is_loaded(self) -> bool: ...
        def unload(self) -> None: ...

        def do_inference(self, ...) -> ...:
            self.touch()           # reset idle timer on each request
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

_IDLE_CHECK_INTERVAL = 30  # seconds between idle checks


class IdleMixin:
    """Mixin that adds idle-timeout tracking to an engine class.

    The host class must define:

    - ``is_loaded`` — bool property indicating whether any model is loaded
    - ``unload()`` — method to release all loaded models

    Call ``touch()`` after each inference to reset the idle timer.
    """

    _last_used: float = 0.0

    # -- Required interface (provided by host class) --------------------------
    # These raise NotImplementedError so that forgetting to override them
    # in the host class produces a clear error rather than a silent bug.

    @property
    def is_loaded(self) -> bool:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

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

        Returns ``True`` if models were unloaded.
        """
        if timeout <= 0 or not self.is_loaded:
            return False
        if self.idle_seconds() >= timeout:
            logger.info(
                "Idle for %.0fs (timeout=%ds), unloading models.",
                self.idle_seconds(), timeout,
            )
            self.unload()
            return True
        return False


@asynccontextmanager
async def idle_checker(engine: IdleMixin, timeout: int) -> AsyncIterator[None]:
    """Async context manager that polls *engine* for idle timeout.

    Starts a background ``asyncio`` task that calls
    ``engine.unload_if_idle(timeout)`` every 30 seconds.  The task is
    cancelled and awaited on exit.  If *timeout* is ``<= 0`` no task
    is started.
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
