"""Forschungs-Debattier-Team mit Magentic-Orchestrierung und Gemini."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agent_framework import (
    Agent,
    AgentContext,
    AgentResponse,
    Content,
    Message,
    Workflow,
    agent_middleware,
)
from agent_framework.orchestrations import MagenticBuilder
from google.genai.errors import ClientError

logger = logging.getLogger(__name__)

MAX_ROUND_COUNT = 10
MAX_STALL_COUNT = 3
MAX_RESET_COUNT = 2

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 0.5

CONTENT_FILTER_FALLBACK = (
    "Die Gemini-Antwort wurde durch den Content-Filter blockiert. "
    "Bitte formuliere die Anfrage wissenschaftlich neutraler."
)

SUGGESTED_MODELS = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-pro-preview")

VISIONARY_INSTRUCTIONS = """\
Du bist der Visionaer in einem wissenschaftlichen Forschungs-Debattier-Team.
Deine Aufgabe: Formuliere gewagte, originelle Thesen und Hypothesen zum Forschungsthema.
Denke spekulativ, aber bleibe anschlussfaehig an etablierte Theorien und Methoden.
Begruende jede These kurz und klar. Antworte auf Deutsch, praezise und diskussionsbereit.
"""

SKEPTIC_INSTRUCTIONS = """\
Du bist der Skeptiker in einem wissenschaftlichen Forschungs-Debattier-Team.
Deine Aufgabe: Validiere Behauptungen wissenschaftlich – pruefe Evidenz, Methodik,
Annahmen und Gegenargumente. Fordere Praezisierung und weise auf Schwaechen hin,
ohne die Debatte zu blockieren. Antworte auf Deutsch, klar und konstruktiv-kritisch.
"""

MANAGER_INSTRUCTIONS = """\
Du bist der Manager eines Forschungs-Debattier-Teams (Visionaer und Skeptiker).
Du koordinierst die Diskussion autonom: plane Runden, weise Sprecher zu,
halte den Fokus auf dem Forschungsziel und beende die Debatte, sobald eine
ausgewogene wissenschaftliche Bilanz vorliegt. Antworte auf Deutsch.
"""


class ModelUnavailableError(RuntimeError):
    """Das konfigurierte Gemini-Modell ist fuer diesen API-Key nicht nutzbar."""


def extract_model_name_from_error(exc: BaseException) -> str | None:
    """Extrahiert den Modellnamen aus einer Gemini-Fehlermeldung, falls vorhanden."""
    match = re.search(r"models/([\w.-]+)", str(exc))
    return match.group(1) if match else None


def is_model_unavailable_error(exc: BaseException) -> bool:
    """Erkennt 404/Deprecation-Fehler fuer nicht verfuegbare Gemini-Modelle."""
    message = str(exc).lower()
    if "no longer available" in message:
        return True
    if isinstance(exc, ClientError) and exc.code == 404:
        return any(
            marker in message
            for marker in ("model", "not_found", "not found", "not available")
        )
    return False


def format_model_unavailable_message(
    model: str | None = None,
    *,
    exc: BaseException | None = None,
) -> str:
    """Erzeugt eine nutzerfreundliche Fehlermeldung inkl. .env-Hinweis."""
    resolved = model or (extract_model_name_from_error(exc) if exc else None) or "unbekannt"
    suggestions = ", ".join(SUGGESTED_MODELS)
    return (
        f"Das konfigurierte Gemini-Modell '{resolved}' ist nicht verfuegbar "
        f"(fuer diesen API-Key gesperrt, veraltet oder unbekannt).\n"
        f"Bitte setze in der .env z.B. GEMINI_MODEL={SUGGESTED_MODELS[0]} "
        f"(Alternativen: {suggestions}) und starte erneut."
    )


def _is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, ClientError) and exc.code == 429:
        return True
    message = str(exc).lower()
    return "429" in message or "resource exhausted" in message or "rate limit" in message


def _is_content_filter_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = ("safety", "blocked", "content filter", "content_filter", "prohibited")
    if any(marker in message for marker in markers):
        return True
    if isinstance(exc, ClientError) and exc.code == 400:
        return any(marker in message for marker in markers)
    return False


@agent_middleware
async def gemini_resilience_middleware(
    context: AgentContext,
    call_next: Callable[[], Awaitable[None]],
) -> None:
    """Absichert Gemini-Aufrufe mit Retry, Content-Filter- und Modell-Fehlerbehandlung."""
    last_error: BaseException | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            await call_next()
            return
        except ModelUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - Middleware faengt API-Fehler zentral
            last_error = exc
            if is_model_unavailable_error(exc):
                message = format_model_unavailable_message(exc=exc)
                logger.error("Gemini-Modell nicht verfuegbar: %s", message)
                raise ModelUnavailableError(message) from exc

            if _is_content_filter_error(exc):
                logger.warning(
                    "Content-Filter fuer Agent %s: %s",
                    getattr(context.agent, "name", "?"),
                    exc,
                )
                context.result = AgentResponse(
                    messages=[
                        Message(
                            role="assistant",
                            contents=[Content.from_text(CONTENT_FILTER_FALLBACK)],
                        )
                    ]
                )
                return

            if _is_rate_limit_error(exc) and attempt < MAX_RETRIES:
                delay = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Rate-Limit (Versuch %s/%s), Backoff %.1fs: %s",
                    attempt,
                    MAX_RETRIES,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                continue

            raise

    if last_error is not None:
        raise last_error


def create_visionary(client: object) -> Agent:
    """Erzeugt den Visionaer-Agenten mit Resilienz-Middleware."""
    return Agent(
        client=client,  # type: ignore[arg-type]
        name="Visionaer",
        description="Formuliert gewagte wissenschaftliche Thesen und Hypothesen.",
        instructions=VISIONARY_INSTRUCTIONS,
        middleware=[gemini_resilience_middleware],
    )


def create_skeptic(client: object) -> Agent:
    """Erzeugt den Skeptiker-Agenten mit Resilienz-Middleware."""
    return Agent(
        client=client,  # type: ignore[arg-type]
        name="Skeptiker",
        description="Validiert Behauptungen wissenschaftlich und kritisch.",
        instructions=SKEPTIC_INSTRUCTIONS,
        middleware=[gemini_resilience_middleware],
    )


def create_manager(client: object) -> Agent:
    """Erzeugt den Manager-Agenten, der die Magentic-Diskussion leitet."""
    return Agent(
        client=client,  # type: ignore[arg-type]
        name="Manager",
        description="Koordiniert das Forschungs-Debattier-Team autonom.",
        instructions=MANAGER_INSTRUCTIONS,
        middleware=[gemini_resilience_middleware],
    )


@dataclass(frozen=True)
class ResearchTeam:
    """Gebautes Forschungs-Team inkl. Magentic-Workflow und Guardrails."""

    workflow: Workflow
    visionary: Agent
    skeptic: Agent
    manager: Agent
    max_round_count: int = MAX_ROUND_COUNT
    max_stall_count: int = MAX_STALL_COUNT
    max_reset_count: int = MAX_RESET_COUNT


def build_research_team(client: object) -> ResearchTeam:
    """Baut Visionaer, Skeptiker und Manager als Magentic-Workflow."""
    visionary = create_visionary(client)
    skeptic = create_skeptic(client)
    manager = create_manager(client)

    workflow = MagenticBuilder(
        participants=[visionary, skeptic],
        intermediate_output_from=[visionary, skeptic],
        manager_agent=manager,
        max_round_count=MAX_ROUND_COUNT,
        max_stall_count=MAX_STALL_COUNT,
        max_reset_count=MAX_RESET_COUNT,
    ).build()

    return ResearchTeam(
        workflow=workflow,
        visionary=visionary,
        skeptic=skeptic,
        manager=manager,
        max_round_count=MAX_ROUND_COUNT,
        max_stall_count=MAX_STALL_COUNT,
        max_reset_count=MAX_RESET_COUNT,
    )


def build_research_workflow(client: object) -> Workflow:
    """Kompatibilitaets-Wrapper: liefert nur den Magentic-Workflow."""
    return build_research_team(client).workflow
