# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 9-agent Python system for FPSO (offshore vessel) risk analysis. Uses the Anthropic SDK, ChromaDB vector store, and a hybrid fuzzy logic + Dempster-Shafer uncertainty fusion approach. The domain is offshore oil & gas risk assessment across 5 domains: Structural Integrity (SI), Maintenance Management (MM), Environmental Hazard (EH), Human Factors (HF), System Integration (SysI).

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
cp .env.example .env   # then fill in ANTHROPIC_API_KEY

# First run seeds the ChromaDB knowledge base automatically
python main.py
```

Requires `ANTHROPIC_API_KEY` in a `.env` file at project root. ChromaDB persists to `C:/Multiagent/chroma_db/`.

## Running

```bash
# Interactive mode (prompts for queries)
python main.py

# One-shot query
python main.py "assess CPP blade corrosion risk during storm conditions"
```

## Tests

```bash
# Run all unit tests (no API calls)
pytest tests/ -v

# Run a single test file
pytest tests/test_fuzzy_engine.py -v

# Run integration tests (hit Claude API, cost money)
pytest tests/ -m integration -v

# Skip integration tests (default recommended)
pytest tests/ -m "not integration" -v
```

Most tests mock the Anthropic API. `test_skill_agents.py` and `test_integration.py` make real API calls unless mocked — mark them explicitly with `@pytest.mark.integration`.

## Pipeline Architecture

The orchestrator (`agents/orchestrator.py`) runs agents sequentially with L4 parallelized:

```
Query
  → L2: RAG Agent          — ChromaDB retrieval → structured evidence dict
  → L3: Skill Router       — keyword pre-filter + Claude LLM → activates domain Skills
  → L4: Skill Agents (×N)  — run in parallel via asyncio; each: fuzzy inference → Claude → SkillResult
  → L5: Synthesis Agent    — Dempster-Shafer BPA combination → combined risk profile
  → Validator              — Claude generates report + HITL pause for human expert
  → L6: Output Agent       — formats final output dict (risk_rankings, risk_profile, traceability, skill_trace)
  → [Feedback Loop]        — optional; expert-gated weight updates to skill JSON files
```

`run_pipeline()` is the synchronous entry point (wraps `asyncio.run`).

## Key Data Contracts

**Evidence dict** (output of RAG Agent, input to Skill Agents):
- `linguistic_assignments`: `{variable_name: "low"|"medium"|"high"}`
- `belief_masses`: `{"Low": float, "Medium": float, "High": float}` (sum to 1.0)
- `risk_factors`: list of `{name, severity (0-1)}`

**`SkillResult`** (`agents/skill_agents/base_skill_agent.py`): the L4→L5 data contract — carries `domain_id`, `risk_score` (0–1 defuzzified crisp), `belief_masses` (BPA), `explanation`, `activated_templates`.

**Output dict** (from L6): `risk_rankings`, `risk_profile`, `traceability`, `skill_trace`.

## Skills System

Skills are versioned JSON files in `skills/` — one per domain (e.g. `skills/structural_integrity.json`). Each Skill encodes:
- `linguistic_variables`: TFN definitions `{variable: {term: [l, m, u]}}`
- `inference_templates`: Mamdani rules `[{id, conditions: {var: term}, output: term, weight}]`
- `routing_confidence`: float used to weight BPAs during Dempster-Shafer synthesis
- `benchmark.version`: integer incremented by the feedback loop

`SkillRepository` (`core/skill_repository.py`) loads/saves these. `repo.save(skill)` overwrites the JSON file and is called by the feedback loop — skill weights are mutable at runtime.

## Core Math Modules

**`core/fuzzy_engine.py`**: `TFN` dataclass for triangular fuzzy numbers; `mamdani_inference()` for AND-aggregated rule firing with centroid defuzzification; `STANDARD_TERMS` defines the 5-point linguistic scale.

**`core/dempster_shafer.py`**: `combine_bpa()` implements Dempster's rule over atomic hypotheses `{Low, Medium, High}`; handles full conflict with uniform fallback. `BeliefInterval` computes `[Bel, Pl]` intervals.

## Configuration

All constants live in `config.py`. Key values:
- `MODEL = "claude-sonnet-4-6"` — used by all agents
- `CHROMA_PERSIST_DIR = "C:/Multiagent/chroma_db"` — absolute Windows path
- `SKILLS_DIR = "C:/Multiagent/skills"` — absolute Windows path
- `HITL_TIMEOUT = None` — validator waits indefinitely for expert input

## Human-in-the-Loop

The validator agent (`agents/validator_agent.py`) pauses for terminal input at the Expert Validation Checkpoint. Options: `[A]pprove`, `[R]eject`, `[M]odify`, `[S]kip`. The feedback loop (`feedback/skill_updater.py`) has a second HITL pause per domain that asks expert approval before writing updated weights back to the skill JSON files.
