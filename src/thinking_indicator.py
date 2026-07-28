"""Asynchroner Konsolen-Indikator fuer Wartezeiten (Nachdenken)."""

from __future__ import annotations

import asyncio
import sys
from typing import TextIO


class ThinkingIndicator:
    """Zeigt eine animierte Statuszeile, solange das Modell arbeitet."""

    def __init__(
        self,
        *,
        label: str = "Nachdenken",
        stream: TextIO | None = None,
        interval: float = 0.4,
    ) -> None:
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.interval = interval
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Startet die Animation, falls sie nicht bereits laeuft."""
        if self.running:
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._animate(), name="thinking-indicator")

    async def stop(self) -> None:
        """Beendet die Animation und loescht die Statuszeile."""
        if self._task is None:
            return
        self._stop.set()
        try:
            await self._task
        finally:
            self._task = None

    async def __aenter__(self) -> ThinkingIndicator:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()

    async def _animate(self) -> None:
        frames = (".  ", ".. ", "...", "   ")
        index = 0
        width = len(self.label) + 4
        try:
            while not self._stop.is_set():
                frame = frames[index % len(frames)]
                print(f"\r{self.label}{frame}", end="", flush=True, file=self.stream)
                index += 1
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
                except TimeoutError:
                    continue
        finally:
            print("\r" + (" " * width) + "\r", end="", flush=True, file=self.stream)
