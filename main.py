"""Interaktives CLI fuer das Forschungs-Debattier-Team (Human-in-the-Loop)."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from agent_framework import AgentResponseUpdate
from agent_framework_gemini import GeminiChatClient
from dotenv import load_dotenv

from src.research_team import ResearchTeam, build_research_team


def create_gemini_client() -> GeminiChatClient:
    """Erzeugt einen GeminiChatClient aus Umgebungsvariablen."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL") or "gemini-2.5-flash"
    if not api_key:
        raise RuntimeError(
            "Kein Gemini-API-Key gefunden. Setze GEMINI_API_KEY oder GOOGLE_API_KEY "
            "(siehe .env.example)."
        )
    return GeminiChatClient(api_key=api_key, model=model)


async def stream_workflow(workflow: Any, prompt: str) -> None:
    """Streamt Magentic-Events live auf die Konsole."""
    last_message_id: str | None = None
    last_label: str | None = None

    async for event in workflow.run(prompt, stream=True):
        if event.type not in ("intermediate", "output"):
            continue
        if not isinstance(event.data, AgentResponseUpdate):
            continue

        update: AgentResponseUpdate = event.data
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

        text = update.text or ""
        if text:
            print(text, end="", flush=True)

    print("\n")


async def read_input(prompt: str) -> str:
    """Liest Terminal-Eingabe ohne den Event-Loop zu blockieren."""
    return await asyncio.to_thread(input, prompt)


async def run_cli() -> None:
    """Phase 1: autonome Diskussion; Phase 2: interaktiver Human-in-the-Loop."""
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
    await stream_workflow(team.workflow, topic)

    print("--- Phase 2: Interaktiver Loop ---")
    print("Gib weitere Anweisungen ein (z.B. 'Skeptiker, gehe tiefer auf Punkt B ein').")
    print("Beenden mit 'exit' oder 'quit'.\n")

    while True:
        user_input = (await read_input("Du> ")).strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Auf Wiedersehen.")
            break

        print("\n--- Weitere Magentic-Runde ---\n")
        await stream_workflow(team.workflow, user_input)


def main() -> None:
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
        sys.exit(130)
    except RuntimeError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
