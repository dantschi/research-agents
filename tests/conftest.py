"""Gemeinsame Test-Fixtures und Fake-Chat-Client fuer Magentic-Tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Awaitable, Mapping, Sequence
from typing import Any

import pytest
from agent_framework import (
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)


class FakeChatClient(BaseChatClient):
    """Deterministischer Chat-Client ohne Netzwerkzugriff.

    Liefert Magentic-kompatible Progress-Ledger-JSON-Antworten und
    Teilnehmerbeitraege fuer Workflow-Streaming-Tests.
    """

    def __init__(self, *, fail_times: int = 0, fail_exception: Exception | None = None) -> None:
        super().__init__()
        self.calls = 0
        self.ledger_calls = 0
        self.participant_texts: list[str] = []
        self._fail_times = fail_times
        self._fail_exception = fail_exception

    def _make_text(self, messages: Sequence[Message]) -> str:
        self.calls += 1
        if self._fail_times > 0 and self._fail_exception is not None:
            self._fail_times -= 1
            raise self._fail_exception

        joined = "\n".join(
            (m.text if getattr(m, "text", None) else str(m)) for m in messages
        ).lower()

        if "is_request_satisfied" in joined:
            self.ledger_calls += 1
            done = self.ledger_calls >= 2
            speaker = "Skeptiker" if self.ledger_calls == 1 else "Visionaer"
            return json.dumps(
                {
                    "is_request_satisfied": {
                        "reason": "Genug Debatte" if done else "Weitere Debatte noetig",
                        "answer": done,
                    },
                    "is_in_loop": {"reason": "Kein Loop", "answer": False},
                    "is_progress_being_made": {"reason": "Fortschritt ok", "answer": True},
                    "next_speaker": {"reason": "Naechster Sprecher", "answer": speaker},
                    "instruction_or_question": {
                        "reason": "Auftrag",
                        "answer": "Teile deine wissenschaftliche Perspektive mit.",
                    },
                }
            )

        if "final answer" in joined or "prepare a final answer" in joined:
            return "FINAL: Debatte abgeschlossen – ausgewogene Forschungsbilanz."

        text = f"Beitrag #{self.calls}: gewagte, aber validierte These zum Forschungsthema."
        self.participant_texts.append(text)
        return text

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        text = self._make_text(messages)

        if stream:

            async def _stream() -> AsyncIterable[ChatResponseUpdate]:
                yield ChatResponseUpdate(
                    contents=[Content.from_text(text)],
                    role="assistant",
                    message_id=f"msg-{self.calls}",
                )

            return ResponseStream(
                _stream(),
                finalizer=lambda updates: ChatResponse.from_updates(list(updates)),
            )

        async def _get_response() -> ChatResponse:
            return ChatResponse(
                messages=[Message(role="assistant", contents=[Content.from_text(text)])]
            )

        return _get_response()


@pytest.fixture
def fake_client() -> FakeChatClient:
    return FakeChatClient()
