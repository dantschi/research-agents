"""Interaktives CLI fuer das Forschungs-Debattier-Team (Human-in-the-Loop)."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from agent_framework import AgentResponseUpdate
from agent_framework_gemini import GeminiChatClient
from dotenv import load_dotenv
from google.genai.errors import APIError

from src.research_team import (
    ModelUnavailableError,
    ResearchTeam,
    ResearchTeamError,
    build_follow_up_task,
    build_research_team,
    format_model_unavailable_message,
    format_workflow_failure_message,
    is_model_unavailable_error,
    is_workflow_failure_event,
)
from src.thinking_indicator import ThinkingIndicator


def create_gemini_client() -> GeminiChatClient:
    """Erzeugt einen GeminiChatClient aus Umgebungsvariablen."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL") or "gemini-3.6-flash"
    if not api_key:
        raise RuntimeError(
            "Kein Gemini-API-Key gefunden. Setze GEMINI_API_KEY oder GOOGLE_API_KEY "
            "(siehe .env.example)."
        )
    return GeminiChatClient(api_key=api_key, model=model)


def _configured_model_name() -> str:
    return os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL") or "gemini-3.6-flash"


def _print_error(message: str) -> None:
    print(f"\nFehler: {message}", file=sys.stderr)


def _print_research_error(exc: BaseException) -> None:
    if isinstance(exc, ModelUnavailableError):
        _print_error(str(exc))
        return
    if isinstance(exc, ResearchTeamError):
        _print_error(str(exc))
        return
    if is_model_unavailable_error(exc):
        _print_error(format_model_unavailable_message(_configured_model_name(), exc=exc))
        return
    _print_error(str(exc))


def _is_completed_workflow_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "already been completed" in message or "no further messages can be processed" in message


async def stream_workflow(workflow: Any, prompt: str) -> tuple[bool, str]:
    """Streamt Magentic-Events live auf die Konsole.

    Returns:
        (True, gesammelter Output-Text) bei Erfolg, sonst (False, bisheriger Text).
    """
    last_message_id: str | None = None
    last_label: str | None = None
    collected: list[str] = []
    thinking = ThinkingIndicator(label="Nachdenken")

    try:
        await thinking.start()
        async for event in workflow.run(prompt, stream=True):
            if is_workflow_failure_event(event):
                await thinking.stop()
                _print_error(format_workflow_failure_message(event))
                return False, "".join(collected)

            if event.type not in ("intermediate", "output"):
                await thinking.start()
                continue
            if not isinstance(event.data, AgentResponseUpdate):
                await thinking.start()
                continue

            update: AgentResponseUpdate = event.data
            text = update.text or ""
            if not text:
                await thinking.start()
                continue

            await thinking.stop()
            message_id = update.message_id
            label = event.executor_id or (
                "Manager" if event.type == "output" else "Agent"
            )

            if message_id != last_message_id:
                if last_message_id is not None:
                    print()
                print(f"\n[{label}]: ", end="", flush=True)
                last_message_id = message_id
                last_label = label
            elif label != last_label:
                print(f"\n[{label}]: ", end="", flush=True)
                last_label = label

            print(text, end="", flush=True)
            collected.append(text)

        print("\n")
        return True, "".join(collected)
    except ResearchTeamError as exc:
        _print_research_error(exc)
        return False, "".join(collected)
    except APIError as exc:
        _print_research_error(exc)
        return False, "".join(collected)
    except RuntimeError as exc:
        if _is_completed_workflow_error(exc):
            _print_error(
                "Der vorherige Magentic-Lauf ist abgeschlossen. "
                "Fuer Folgefragen wird automatisch ein neuer Workflow gestartet – "
                "bitte den Prompt erneut eingeben."
            )
            return False, "".join(collected)
        raise
    except (TimeoutError, OSError, ConnectionError) as exc:
        _print_error(
            "Netzwerk- oder Timeout-Fehler bei der Gemini-Anfrage. "
            f"Details: {exc}"
        )
        return False, "".join(collected)
    finally:
        await thinking.stop()


async def read_input(prompt: str) -> str:
    """Liest Terminal-Eingabe ohne den Event-Loop zu blockieren."""
    return await asyncio.to_thread(input, prompt)


async def run_cli() -> None:
    """Phase 1: autonome Diskussion; Phase 2: neue Magentic-Instanz pro Prompt."""
    print("=" * 60)
    print(" Forschungs-Debattier-Team (Magentic + Gemini)")
    print(" Visionaer | Skeptiker | Manager")
    print("=" * 60)

    topic = (await read_input("\nForschungsthema: ")).strip()
    if not topic:
        print("Kein Thema angegeben – Abbruch.")
        return

    client = create_gemini_client()
    team: ResearchTeam = build_research_team(client)

    print("\n--- Phase 1: Autonome Diskussion ---\n")
    ok, phase1_text = await stream_workflow(team.workflow, topic)
    if not ok:
        print(
            "Diskussion abgebrochen. Bitte Konfiguration (.env) pruefen oder spaeter erneut starten."
        )
        return

    prior_summary = phase1_text.strip()

    print("--- Phase 2: Interaktiver Loop ---")
    print("Gib weitere Anweisungen ein (z.B. 'Skeptiker, gehe tiefer auf Punkt B ein').")
    print("Jeder Prompt startet einen neuen Magentic-Lauf mit bisherigem Kontext.")
    print("Beenden mit 'exit' oder 'quit'.\n")

    while True:
        user_input = (await read_input("Du> ")).strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Auf Wiedersehen.")
            break

        print("\n--- Weitere Magentic-Runde ---\n")
        # Magentic-Instanzen sind nach Abschluss terminiert – daher neu bauen.
        follow_up_team = build_research_team(client)
        follow_up_task = build_follow_up_task(
            original_topic=topic,
            prior_summary=prior_summary,
            user_prompt=user_input,
        )
        ok, round_text = await stream_workflow(follow_up_team.workflow, follow_up_task)
        if not ok:
            print("Runde abgebrochen. Du kannst es mit einem anderen Prompt erneut versuchen.")
            continue

        if round_text.strip():
            prior_summary = (
                f"{prior_summary}\n\nNutzer: {user_input}\n\nTeam:\n{round_text.strip()}"
            ).strip()


def main() -> None:
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
        sys.exit(130)
    except ResearchTeamError as exc:
        _print_research_error(exc)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
