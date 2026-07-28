"""Unit-Tests fuer den asynchronen Nachdenken-Indikator."""

from __future__ import annotations

import asyncio
from io import StringIO

import pytest

from src.thinking_indicator import ThinkingIndicator


@pytest.mark.asyncio
async def test_indicator_writes_and_clears_line() -> None:
    buffer = StringIO()
    indicator = ThinkingIndicator(label="Nachdenken", stream=buffer, interval=0.05)

    await indicator.start()
    await asyncio.sleep(0.12)
    assert indicator.running is True
    await indicator.stop()

    assert indicator.running is False
    output = buffer.getvalue()
    assert "Nachdenken" in output
    assert "." in output


@pytest.mark.asyncio
async def test_start_is_idempotent() -> None:
    buffer = StringIO()
    indicator = ThinkingIndicator(label="Warten", stream=buffer, interval=0.05)

    await indicator.start()
    await indicator.start()
    assert indicator.running is True
    await indicator.stop()
    await indicator.stop()
    assert indicator.running is False


@pytest.mark.asyncio
async def test_context_manager_stops_on_exit() -> None:
    buffer = StringIO()
    async with ThinkingIndicator(label="Denk", stream=buffer, interval=0.05) as indicator:
        assert indicator.running is True
        await asyncio.sleep(0.08)
    assert indicator.running is False
