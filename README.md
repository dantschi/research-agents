# research-agents

Autonomes Forschungs-Debattier-Team mit Microsoft Agent Framework 1.0 und Google Gemini.

## Team

- **Visionaer** – formuliert gewagte wissenschaftliche Thesen
- **Skeptiker** – validiert Behauptungen wissenschaftlich
- **Manager** – orchestriert die Debatte via MagenticBuilder

## Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env   # GEMINI_API_KEY setzen
```

## Tests

```bash
python -m pytest
```

## CLI

```bash
python main.py
```

Phase 1 startet die autonome Magentic-Diskussion zum eingegebenen Forschungsthema.
Phase 2 nimmt weitere Terminal-Prompts entgegen und fuehrt sie im bestehenden Workflow-Kontext aus (`exit`/`quit` beendet).
