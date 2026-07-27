"""Unit-Tests fuer das Forschungs-Debattier-Team (TDD)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_framework import Agent, AgentContext, AgentResponse, Content, Message, Workflow
from google.genai.errors import ClientError

from src.research_team import (
    CONTENT_FILTER_FALLBACK,
    MAX_RESET_COUNT,
    MAX_ROUND_COUNT,
    MAX_STALL_COUNT,
    ResearchTeam,
    build_research_team,
    create_manager,
    create_skeptic,
    create_visionary,
    gemini_resilience_middleware,
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
