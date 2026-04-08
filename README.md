# Multi-Agent Skill-Enhanced RAG Risk Analysis System

**Code availability:** This repository accompanies a paper submitted to the Journal of Information and Sofrware Technology. A permanent DOI will be added upon acceptance.

A 9-agent Python system for FPSO (Floating Production, Storage and Offloading) vessel risk analysis. Combines a ChromaDB vector knowledge base, Mamdani fuzzy inference, and Dempster-Shafer uncertainty fusion to produce ranked risk assessments across five offshore engineering domains.

## Architecture

```
Query
  → L2: RAG Agent          — ChromaDB retrieval → structured evidence dict
  → L3: Skill Router       — keyword pre-filter + LLM → activates domain Skills
  → L4: Skill Agents (×N)  — parallel async; each runs fuzzy inference → SkillResult
  → L5: Synthesis Agent    — Dempster-Shafer BPA combination → combined risk profile
  → Validator              — LLM validation report + HITL pause for expert decision
  → L6: Output Agent       — formats final output (risk_rankings, risk_profile, traceability)
  → [Feedback Loop]        — optional; expert-gated weight updates to skill JSON files
```

### Risk Domains (Skills)

| ID | Domain | Key Variables |
|----|--------|---------------|
| SI | Structural Integrity | corrosion_rate, fatigue_level, inspection_frequency |
| MM | Maintenance Management | maintenance_frequency, equipment_condition, overdue_tasks |
| EH | Environmental Hazard | wave_height, wind_speed, current_speed |
| HF | Human Factors | training_adequacy, procedural_compliance, operator_fatigue |
| SysI | System Integration | redundancy_level, sensor_reliability, integration_complexity |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...
```

Requires Python 3.10+ and an Anthropic API key. ChromaDB persists to `./chroma_db/` by default (override with `CHROMA_PERSIST_DIR` env var).

## Running

### Web Dashboard (recommended)

```bash
py server.py
```

> **Windows note:** If `py` / `python` are not on PATH, use the full executable path:
> ```powershell
> C:\Users\ENRJREN\AppData\Local\Programs\Python\Python313\python.exe server.py
> ```

Open `http://localhost:5000` in a browser. The dashboard provides:
- Query input with optional domain chip filtering (SI / MM / EH / HF / SYSI)
- Live log stream showing each pipeline stage as it executes
- Expert Validation Checkpoint panel with Approve / Review / Reject controls
- Risk ranking bar chart, radar chart, and traceability panel

### Command Line

```bash
# Interactive mode (prompts for queries)
py main.py

# One-shot query
py main.py "assess CPP blade corrosion risk during storm conditions"
```

## Tests

```bash
# Run all unit tests (no API calls, recommended)
pytest tests/ -m "not integration" -v

# Run a single test file
pytest tests/test_fuzzy_engine.py -v

# Run integration tests (hit Claude API — costs tokens)
pytest tests/ -m integration -v
```

## Project Structure

```
multiagent/
├── agents/
│   ├── orchestrator.py          # Pipeline entry point; L2→L6 sequencing
│   ├── rag_agent.py             # L2: ChromaDB retrieval + evidence extraction
│   ├── skill_router.py          # L3: domain selection (keyword + LLM)
│   ├── skill_agents/
│   │   ├── base_skill_agent.py  # L4: fuzzy inference + LLM explanation
│   │   ├── si_agent.py          # Structural Integrity wrapper
│   │   ├── mm_agent.py          # Maintenance Management wrapper
│   │   ├── eh_agent.py          # Environmental Hazard wrapper
│   │   ├── hf_agent.py          # Human Factors wrapper
│   │   └── sysi_agent.py        # System Integration wrapper
│   ├── synthesis_agent.py       # L5: Dempster-Shafer BPA combination
│   ├── validator_agent.py       # Expert validation + HITL bridge
│   └── output_agent.py          # L6: final output formatting
├── core/
│   ├── fuzzy_engine.py          # TFN dataclass, Mamdani inference, defuzzification
│   ├── dempster_shafer.py       # BPA combination, belief/plausibility intervals
│   ├── skill_repository.py      # Loads/saves skill JSON files
│   ├── vector_store.py          # ChromaDB wrapper
│   ├── hitl_bridge.py           # threading.Event for web HITL synchronisation
│   └── stream_queue.py          # sys.stdout capture → queue.Queue for SSE
├── skills/                      # Versioned domain skill definitions (JSON)
│   ├── structural_integrity.json
│   ├── maintenance_management.json
│   ├── environmental_hazard.json
│   ├── human_factors.json
│   └── system_integration.json
├── feedback/
│   ├── skill_updater.py         # Expert-gated weight update loop
│   └── benchmarking.py          # Skill performance benchmarking
├── tests/                       # pytest unit and integration tests
├── server.py                    # Flask web server (/, /run, /stream, /hitl)
├── dashboard.html               # Single-page web UI
├── main.py                      # CLI entry point + knowledge base seeding
└── config.py                    # MODEL, paths, timeouts
```

## Key Data Contracts

**Evidence dict** (RAG Agent output → Skill Agent input):
```python
{
    "linguistic_assignments": {"corrosion_rate": "high", "wave_height": "medium", ...},
    "belief_masses":          {"Low": 0.1, "Medium": 0.3, "High": 0.6},
    "risk_factors":           [{"name": "blade corrosion", "severity": 0.82}],
    "evidence_summary":       "...",
    "source_documents":       [...]
}
```

**SkillResult** (L4 → L5):
```python
SkillResult(
    domain_id="SI",
    risk_score=0.74,          # 0–1 defuzzified crisp value
    belief_masses={"Low": ..., "Medium": ..., "High": ...},
    explanation="...",
    activated_templates=[...]
)
```

**Output dict** (L6):
```python
{
    "risk_rankings":  [{"rank": 1, "domain_id": "SI", "risk_score": 0.74, ...}],
    "risk_profile":   {"dominant_risk": "SI", "overall_risk_level": "High", ...},
    "traceability":   {...},
    "skill_trace":    {...}
}
```

## Core Math

**Fuzzy inference** (`core/fuzzy_engine.py`): Triangular fuzzy numbers (TFN) with Mamdani AND-aggregation and centroid defuzzification. Unassigned variables receive uniform uncertainty (`1/n` per term) so rules can partially fire on sparse evidence.

**Dempster-Shafer** (`core/dempster_shafer.py`): BPA combination over atomic hypotheses `{Low, Medium, High}`. Belief intervals are zero-width (`Bel == Pl`) by design — a consequence of using atomic singleton hypotheses, not a system error.

**Skill weights**: Stored in `skills/*.json` as `inference_templates[*].weight` and updated at runtime by the feedback loop when an expert approves. `benchmark.version` is incremented on each save.

## Human-in-the-Loop

The validator pauses the pipeline for expert review at two points:

1. **Validation Checkpoint** — after synthesis, before output. Expert sees the AI recommendation (approve / review / reject) and the risk ranking. In web mode the browser dashboard presents this; in CLI mode it reads from `stdin`.

2. **Feedback Checkpoint** *(optional, `--feedback` flag)* — after output, per domain. Expert approves or rejects proposed weight updates before they are written back to `skills/*.json`.

## Configuration

All constants are in `config.py`:

| Key | Default | Description |
|-----|---------|-------------|
| `MODEL` | `claude-sonnet-4-6` | Anthropic model used by all agents |
| `MAX_TOKENS` | `4096` | Max completion tokens per agent call |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence path |
| `SKILLS_DIR` | `./skills` | Skill JSON file directory |
| `HITL_TIMEOUT` | `None` | Seconds to wait for expert input (None = indefinite) |
