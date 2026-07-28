"""Unit-Tests fuer das Forschungs-Debattier-Team (TDD)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_framework import Agent, AgentContext, AgentResponse, Content, Message, Workflow
from google.genai.errors import ClientError, ServerError

from src.research_team import (
    CONTENT_FILTER_FALLBACK,
    MAX_RESET_COUNT,
    MAX_ROUND_COUNT,
    MAX_STALL_COUNT,
    GeminiAuthError,
    GeminiTransientError,
    ModelUnavailableError,
    ResearchTeam,
    build_research_team,
    create_manager,
    create_skeptic,
    create_visionary,
    format_model_unavailable_message,
    format_workflow_failure_message,
    gemini_resilience_middleware,
    is_auth_error,
    is_model_unavailable_error,
    is_transient_error,
    is_workflow_failure_event,
)
from tests.conftest import FakeChatClient


class TestAgentCreation:
    def test_visionary_agent_is_instantiated_correctly(self, fake_client: FakeChatClient) -> None:
        agent = create_visionary(fake_client)

        assert isinstance(agent, Agent)
        assert agent.name == "Visionaer"
        assert agent.description is not None
        instructions = agent.default_options.get("instructions", "")
        assert "gewagte" in instructions.lower() or "these" in instructions.lower()

    def test_skeptic_agent_is_instantiated_correctly(self, fake_client: FakeChatClient) -> None:
        agent = create_skeptic(fake_client)

        assert isinstance(agent, Agent)
        assert agent.name == "Skeptiker"
        assert agent.description is not None
        instructions = agent.default_options.get("instructions", "")
        assert "valid" in instructions.lower() or "wissenschaft" in instructions.lower()

    def test_manager_agent_is_instantiated_correctly(self, fake_client: FakeChatClient) -> None:
        agent = create_manager(fake_client)

        assert isinstance(agent, Agent)
        assert agent.name == "Manager"
        assert agent.description is not None
        instructions = agent.default_options.get("instructions", "")
        assert "koordin" in instructions.lower() or "coord" in instructions.lower()

    def test_agents_have_resilience_middleware(self, fake_client: FakeChatClient) -> None:
        agents = [
            create_visionary(fake_client),
            create_skeptic(fake_client),
            create_manager(fake_client),
        ]
        for agent in agents:
            assert agent.middleware is not None
            assert gemini_resilience_middleware in agent.middleware


class TestWorkflowConstruction:
    def test_magentic_builder_produces_valid_workflow(self, fake_client: FakeChatClient) -> None:
        team = build_research_team(fake_client)

        assert isinstance(team, ResearchTeam)
        assert isinstance(team.workflow, Workflow)
        assert team.visionary.name == "Visionaer"
        assert team.skeptic.name == "Skeptiker"
        assert team.manager.name == "Manager"

    def test_workflow_has_guardrails(self, fake_client: FakeChatClient) -> None:
        team = build_research_team(fake_client)

        assert team.max_round_count == MAX_ROUND_COUNT
        assert team.max_stall_count == MAX_STALL_COUNT
        assert team.max_reset_count == MAX_RESET_COUNT
        assert MAX_ROUND_COUNT > 0
        assert MAX_STALL_COUNT > 0


class TestGeminiResponseProcessing:
    @pytest.mark.asyncio
    async def test_mocked_gemini_response_flows_through_group_context(
        self, fake_client: FakeChatClient
    ) -> None:
        team = build_research_team(fake_client)

        events: list[Any] = []
        texts: list[str] = []
        async for event in team.workflow.run(
            "Bewerte die Hypothese, dass LLMs emergentes Bewusstsein entwickeln.",
            stream=True,
        ):
            events.append(event)
            if event.type in ("intermediate", "output") and event.data is not None:
                text = getattr(event.data, "text", None) or str(event.data)
                if text:
                    texts.append(text)

        assert len(events) > 0
        assert fake_client.calls > 0
        assert any("Beitrag" in t or "FINAL" in t or "These" in t for t in texts)
        assert any(e.type in ("intermediate", "output", "magentic_orchestrator") for e in events)


class TestResilienceMiddleware:
    @pytest.mark.asyncio
    async def test_middleware_retries_on_rate_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        attempts = {"count": 0}

        async def flaky_next() -> None:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ClientError(429, {"error": {"message": "Resource exhausted", "code": 429}})

        agent = MagicMock()
        agent.name = "Visionaer"
        context = AgentContext(
            agent=agent,
            messages=[Message(role="user", contents=[Content.from_text("Hallo")])],
        )

        await gemini_resilience_middleware(context, flaky_next)

        assert attempts["count"] == 3

    @pytest.mark.asyncio
    async def test_middleware_handles_content_filter(self) -> None:
        async def blocked_next() -> None:
            raise ClientError(
                400,
                {
                    "error": {
                        "message": "The response was blocked due to SAFETY",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )

        agent = MagicMock()
        agent.name = "Skeptiker"
        context = AgentContext(
            agent=agent,
            messages=[
                Message(role="user", contents=[Content.from_text("Riskanter Inhalt")])
            ],
        )

        await gemini_resilience_middleware(context, blocked_next)

        assert context.result is not None
        assert isinstance(context.result, AgentResponse)
        assert CONTENT_FILTER_FALLBACK in (context.result.text or "")


class TestModelUnavailableHandling:
    def test_detects_model_unavailable_404(self) -> None:
        exc = ClientError(
            404,
            {
                "error": {
                    "code": 404,
                    "message": (
                        "This model models/gemini-2.5-pro is no longer available "
                        "to new users. Please update your code to use a newer model."
                    ),
                    "status": "NOT_FOUND",
                }
            },
        )
        assert is_model_unavailable_error(exc) is True

    def test_format_message_mentions_model_and_env_hint(self) -> None:
        message = format_model_unavailable_message("gemini-2.5-pro")
        assert "gemini-2.5-pro" in message
        assert "GEMINI_MODEL" in message
        assert "gemini-3.6-flash" in message or "gemini-3.1-pro-preview" in message

    @pytest.mark.asyncio
    async def test_middleware_raises_friendly_model_unavailable_error(self) -> None:
        async def unavailable_next() -> None:
            raise ClientError(
                404,
                {
                    "error": {
                        "code": 404,
                        "message": (
                            "This model models/gemini-2.5-flash is no longer available "
                            "to new users."
                        ),
                        "status": "NOT_FOUND",
                    }
                },
            )

        agent = MagicMock()
        agent.name = "Manager"
        context = AgentContext(
            agent=agent,
            messages=[Message(role="user", contents=[Content.from_text("Thema")])],
        )

        with pytest.raises(ModelUnavailableError) as raised:
            await gemini_resilience_middleware(context, unavailable_next)

        assert "gemini-2.5-flash" in str(raised.value)
        assert "GEMINI_MODEL" in str(raised.value)


def _agent_context(name: str = "Manager") -> AgentContext:
    agent = MagicMock()
    agent.name = name
    return AgentContext(
        agent=agent,
        messages=[Message(role="user", contents=[Content.from_text("Thema")])],
    )


class TestAuthAndTransientErrors:
    def test_detects_auth_errors(self) -> None:
        assert is_auth_error(ClientError(401, {"error": {"message": "API key invalid"}}))
        assert is_auth_error(ClientError(403, {"error": {"message": "PERMISSION_DENIED"}}))
        assert not is_auth_error(ClientError(429, {"error": {"message": "Resource exhausted"}}))

    def test_detects_transient_server_errors(self) -> None:
        assert is_transient_error(ServerError(503, {"error": {"message": "UNAVAILABLE"}}))
        assert is_transient_error(ServerError(500, {"error": {"message": "INTERNAL"}}))
        assert is_transient_error(ClientError(429, {"error": {"message": "Resource exhausted"}}))
        assert not is_transient_error(ClientError(401, {"error": {"message": "bad key"}}))

    @pytest.mark.asyncio
    async def test_middleware_raises_auth_error(self) -> None:
        async def unauthorized() -> None:
            raise ClientError(401, {"error": {"message": "API key not valid", "code": 401}})

        with pytest.raises(GeminiAuthError) as raised:
            await gemini_resilience_middleware(_agent_context(), unauthorized)

        assert "API-Key" in str(raised.value) or "GEMINI_API_KEY" in str(raised.value)

    @pytest.mark.asyncio
    async def test_middleware_retries_server_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        attempts = {"count": 0}

        async def flaky() -> None:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ServerError(503, {"error": {"message": "UNAVAILABLE", "code": 503}})

        await gemini_resilience_middleware(_agent_context(), flaky)
        assert attempts["count"] == 3

    @pytest.mark.asyncio
    async def test_middleware_raises_transient_after_retries_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        async def always_busy() -> None:
            raise ClientError(429, {"error": {"message": "Resource exhausted", "code": 429}})

        with pytest.raises(GeminiTransientError) as raised:
            await gemini_resilience_middleware(_agent_context(), always_busy)

        assert "Rate" in str(raised.value) or "spaeter" in str(raised.value).lower() or "später" in str(raised.value).lower() or "Quota" in str(raised.value) or "Limit" in str(raised.value)


class TestWorkflowFailureEvents:
    def test_detects_failed_event_types(self) -> None:
        failed = MagicMock()
        failed.type = "failed"
        failed.data = None
        executor_failed = MagicMock()
        executor_failed.type = "executor_failed"
        executor_failed.data = MagicMock(error_type="ClientError", message="boom", executor_id="Skeptiker")
        ok = MagicMock()
        ok.type = "output"
        ok.data = None

        assert is_workflow_failure_event(failed) is True
        assert is_workflow_failure_event(executor_failed) is True
        assert is_workflow_failure_event(ok) is False

    def test_formats_workflow_failure_message(self) -> None:
        event = MagicMock()
        event.type = "executor_failed"
        event.data = MagicMock(
            error_type="AttributeError",
            message="something broke",
            executor_id="Visionaer",
        )
        text = format_workflow_failure_message(event)
        assert "Visionaer" in text
        assert "something broke" in text


class TestFollowUpTask:
    def test_build_follow_up_task_includes_topic_history_and_prompt(self) -> None:
        from src.research_team import build_follow_up_task

        task = build_follow_up_task(
            original_topic="RISC-V fuer Physical AI",
            prior_summary="Visionaer und Skeptiker einigten sich auf X.",
            user_prompt="Welche Foerderprojekte eignen sich?",
        )
        assert "RISC-V fuer Physical AI" in task
        assert "Visionaer und Skeptiker einigten sich auf X." in task
        assert "Welche Foerderprojekte eignen sich?" in task
        assert "Fortsetzung" in task or "fortsetzen" in task.lower() or "Neue Anweisung" in task
