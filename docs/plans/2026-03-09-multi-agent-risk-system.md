# Multi-Agent Skill-Enhanced RAG Risk Analysis System — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the monolithic Skill-Enhanced RAG Expert System with a 9-agent Python system using the Anthropic SDK, where 5 domain Skill Agents run in parallel and are orchestrated through a shared pipeline.

**Architecture:** An Orchestrator Agent coordinates sequential layers (RAG → Skill Router → parallel Skill Agents → Synthesis → Validator → Output → Feedback). Each layer is an independent agent implemented as an async function calling `client.messages.create()` with domain-specific system prompts and tools. Skills (S = ⟨τ, P, F, T, B⟩) are stored as versioned JSON files loaded at runtime by each Skill Agent. Human-in-the-loop pauses are implemented at the Expert Validation Checkpoint (L4) and Expert Gate (L6 feedback loop).

**Tech Stack:** Python 3.13, `anthropic` SDK, `chromadb` (vector store), `sentence-transformers` (embeddings), `numpy` (fuzzy math), `asyncio` (parallel Skill Agent execution), `pytest` (tests)

---

## Project Structure

```
C:/MultiAgent/
├── main.py
├── config.py
├── requirements.txt
├── skills/                        # Skill Repository (JSON, version-controlled)
│   ├── structural_integrity.json
│   ├── maintenance_management.json
│   ├── environmental_hazard.json
│   ├── human_factors.json
│   └── system_integration.json
├── core/
│   ├── __init__.py
│   ├── fuzzy_engine.py            # TFN math, Mamdani inference
│   ├── dempster_shafer.py         # DS combination, belief/plausibility
│   ├── vector_store.py            # ChromaDB wrapper
│   └── skill_repository.py        # Load/save/version Skills
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── rag_agent.py
│   ├── skill_router.py
│   ├── skill_agents/
│   │   ├── __init__.py
│   │   ├── base_skill_agent.py
│   │   ├── si_agent.py
│   │   ├── mm_agent.py
│   │   ├── eh_agent.py
│   │   ├── hf_agent.py
│   │   └── sysi_agent.py
│   ├── synthesis_agent.py
│   ├── validator_agent.py
│   └── output_agent.py
├── feedback/
│   ├── __init__.py
│   ├── benchmarking.py
│   └── skill_updater.py
└── tests/
    ├── test_fuzzy_engine.py
    ├── test_dempster_shafer.py
    ├── test_skill_repository.py
    ├── test_skill_router.py
    ├── test_skill_agents.py
    ├── test_synthesis_agent.py
    └── test_integration.py
```

---

## Task 1: Project Bootstrap + Requirements

**Files:**
- Create: `C:/MultiAgent/requirements.txt`
- Create: `C:/MultiAgent/config.py`

**Step 1: Create requirements.txt**

```
anthropic>=0.40.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
numpy>=1.26.0
python-docx>=1.1.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
python-dotenv>=1.0.0
```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

**Step 3: Create config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

# Vector store
CHROMA_PERSIST_DIR = "C:/MultiAgent/chroma_db"
COLLECTION_NAME = "risk_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Skill repository
SKILLS_DIR = "C:/MultiAgent/skills"

# Human-in-the-loop timeouts (seconds, None = wait indefinitely)
HITL_TIMEOUT = None
```

**Step 4: Create .env file**

```
ANTHROPIC_API_KEY=your_key_here
```

**Step 5: Create all __init__.py files**

```bash
touch C:/MultiAgent/core/__init__.py
touch C:/MultiAgent/agents/__init__.py
touch C:/MultiAgent/agents/skill_agents/__init__.py
touch C:/MultiAgent/feedback/__init__.py
touch C:/MultiAgent/tests/__init__.py
```

**Step 6: Commit**

```bash
git add .
git commit -m "feat: bootstrap project structure and config"
```

---

## Task 2: Fuzzy Engine (Core Math)

**Files:**
- Create: `C:/MultiAgent/core/fuzzy_engine.py`
- Create: `C:/MultiAgent/tests/test_fuzzy_engine.py`

**Background:** The fuzzy engine implements Triangular Fuzzy Numbers (TFN = (l, m, u)), fuzzification of crisp inputs into linguistic membership degrees, and Mamdani inference (rule activation → aggregation → centroid defuzzification). Linguistic terms map to TFNs: Very Low=(0,0,0.25), Low=(0,0.25,0.5), Medium=(0.25,0.5,0.75), High=(0.5,0.75,1.0), Very High=(0.75,1.0,1.0).

**Step 1: Write failing tests**

```python
# tests/test_fuzzy_engine.py
import pytest
import numpy as np
from core.fuzzy_engine import TFN, triangular_membership, fuzzify, mamdani_inference, defuzzify

def test_tfn_membership_at_peak():
    tfn = TFN(0.0, 0.5, 1.0)
    assert tfn.membership(0.5) == pytest.approx(1.0)

def test_tfn_membership_at_edges():
    tfn = TFN(0.0, 0.5, 1.0)
    assert tfn.membership(0.0) == pytest.approx(0.0)
    assert tfn.membership(1.0) == pytest.approx(0.0)

def test_tfn_membership_outside():
    tfn = TFN(0.2, 0.5, 0.8)
    assert tfn.membership(0.1) == pytest.approx(0.0)
    assert tfn.membership(0.9) == pytest.approx(0.0)

def test_fuzzify_returns_membership_degrees():
    terms = {
        "low":    TFN(0.0, 0.0, 0.5),
        "medium": TFN(0.0, 0.5, 1.0),
        "high":   TFN(0.5, 1.0, 1.0),
    }
    result = fuzzify(0.5, terms)
    assert "low" in result
    assert "medium" in result
    assert result["medium"] == pytest.approx(1.0)

def test_defuzzify_symmetric():
    # Symmetric activation should return middle value
    output_terms = {
        "low":  TFN(0.0, 0.0, 0.5),
        "high": TFN(0.5, 1.0, 1.0),
    }
    activations = {"low": 0.5, "high": 0.5}
    result = defuzzify(activations, output_terms)
    assert 0.3 < result < 0.7  # should be roughly centred

def test_mamdani_inference_high_risk():
    rules = [
        {"conditions": {"corrosion": "high"}, "output": "high", "weight": 1.0},
    ]
    input_memberships = {"corrosion": {"low": 0.0, "medium": 0.1, "high": 0.9}}
    output_terms = {
        "low":  TFN(0.0, 0.0, 0.5),
        "medium": TFN(0.0, 0.5, 1.0),
        "high": TFN(0.5, 1.0, 1.0),
    }
    score = mamdani_inference(rules, input_memberships, output_terms)
    assert score > 0.6
```

**Step 2: Run tests — verify they fail**

```bash
cd C:/MultiAgent && pytest tests/test_fuzzy_engine.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`

**Step 3: Implement fuzzy_engine.py**

```python
# core/fuzzy_engine.py
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class TFN:
    """Triangular Fuzzy Number (l, m, u)."""
    l: float
    m: float
    u: float

    def membership(self, x: float) -> float:
        if x <= self.l or x >= self.u:
            return 0.0
        if x <= self.m:
            return (x - self.l) / (self.m - self.l) if self.m != self.l else 1.0
        return (self.u - x) / (self.u - self.m) if self.u != self.m else 1.0


# Standard linguistic term TFNs
STANDARD_TERMS: Dict[str, TFN] = {
    "very_low":  TFN(0.00, 0.00, 0.25),
    "low":       TFN(0.00, 0.25, 0.50),
    "medium":    TFN(0.25, 0.50, 0.75),
    "high":      TFN(0.50, 0.75, 1.00),
    "very_high": TFN(0.75, 1.00, 1.00),
}


def fuzzify(crisp_value: float, terms: Dict[str, TFN]) -> Dict[str, float]:
    """Convert a crisp value to membership degrees for each linguistic term."""
    return {term: tfn.membership(crisp_value) for term, tfn in terms.items()}


def mamdani_inference(
    rules: List[Dict],
    input_memberships: Dict[str, Dict[str, float]],
    output_terms: Dict[str, TFN],
) -> float:
    """
    Mamdani fuzzy inference.
    rules: list of {"conditions": {var: term}, "output": term, "weight": float}
    input_memberships: {var: {term: degree}}
    Returns defuzzified crisp risk score.
    """
    output_activations: Dict[str, float] = {t: 0.0 for t in output_terms}

    for rule in rules:
        # AND aggregation (min) across conditions
        firing_strength = min(
            input_memberships.get(var, {}).get(term, 0.0)
            for var, term in rule["conditions"].items()
        )
        firing_strength *= rule.get("weight", 1.0)
        out_term = rule["output"]
        if out_term in output_activations:
            output_activations[out_term] = max(
                output_activations[out_term], firing_strength
            )

    return defuzzify(output_activations, output_terms)


def defuzzify(
    activations: Dict[str, float], output_terms: Dict[str, TFN]
) -> float:
    """Centroid defuzzification over discretised universe [0, 1]."""
    x_points = np.linspace(0, 1, 200)
    aggregated = np.zeros_like(x_points)

    for term, activation in activations.items():
        if term in output_terms and activation > 0:
            tfn = output_terms[term]
            memberships = np.array([tfn.membership(x) for x in x_points])
            aggregated = np.maximum(aggregated, activation * memberships)

    total = np.sum(aggregated)
    if total == 0:
        return 0.5  # default to medium uncertainty
    return float(np.sum(x_points * aggregated) / total)
```

**Step 4: Run tests — verify they pass**

```bash
cd C:/MultiAgent && pytest tests/test_fuzzy_engine.py -v
```
Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add core/fuzzy_engine.py tests/test_fuzzy_engine.py
git commit -m "feat: implement TFN fuzzy engine with Mamdani inference"
```

---

## Task 3: Dempster-Shafer Evidential Reasoning

**Files:**
- Create: `C:/MultiAgent/core/dempster_shafer.py`
- Create: `C:/MultiAgent/tests/test_dempster_shafer.py`

**Background:** DS theory combines evidence (basic probability assignments / belief mass functions) from multiple sources over a frame of discernment Ω = {Low, Medium, High}. Dempster's rule normalises combined mass by (1 − K) where K is the conflict mass. Belief(A) = sum of m(B) for all B ⊆ A. Plausibility(A) = 1 − Belief(Ā).

**Step 1: Write failing tests**

```python
# tests/test_dempster_shafer.py
import pytest
from core.dempster_shafer import (
    combine_bpa, belief, plausibility, BeliefInterval, normalise_bpa
)

def test_normalise_bpa_sums_to_one():
    raw = {"Low": 0.3, "Medium": 0.4, "High": 0.2}
    result = normalise_bpa(raw)
    assert sum(result.values()) == pytest.approx(1.0)

def test_combine_two_consistent_bpas():
    m1 = {"Low": 0.8, "High": 0.2}
    m2 = {"Low": 0.7, "High": 0.3}
    combined = combine_bpa(m1, m2)
    assert combined["Low"] > combined["High"]
    assert sum(combined.values()) == pytest.approx(1.0)

def test_belief_interval_low_risk():
    m = {"Low": 0.7, "Medium": 0.2, "High": 0.1}
    frame = ["Low", "Medium", "High"]
    bel = belief(m, "Low", frame)
    pla = plausibility(m, "Low", frame)
    assert bel <= pla
    assert 0.0 <= bel <= 1.0
    assert 0.0 <= pla <= 1.0

def test_belief_interval_dataclass():
    m = {"Low": 0.6, "Medium": 0.3, "High": 0.1}
    frame = ["Low", "Medium", "High"]
    interval = BeliefInterval.from_bpa(m, "Low", frame)
    assert interval.belief <= interval.plausibility

def test_high_conflict_combination():
    m1 = {"Low": 1.0}
    m2 = {"High": 1.0}
    # Should not raise, should return a valid distribution
    combined = combine_bpa(m1, m2)
    assert sum(combined.values()) == pytest.approx(1.0)
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_dempster_shafer.py -v
```

**Step 3: Implement dempster_shafer.py**

```python
# core/dempster_shafer.py
from dataclasses import dataclass
from typing import Dict, List
from itertools import product


def normalise_bpa(bpa: Dict[str, float]) -> Dict[str, float]:
    """Normalise a BPA so values sum to 1.0."""
    total = sum(bpa.values())
    if total == 0:
        n = len(bpa)
        return {k: 1.0 / n for k in bpa}
    return {k: v / total for k, v in bpa.items()}


def combine_bpa(m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
    """
    Dempster's rule of combination for two BPAs over atomic hypotheses.
    Handles full conflict by uniform distribution fallback.
    """
    combined: Dict[str, float] = {}
    conflict = 0.0

    for (h1, v1), (h2, v2) in product(m1.items(), m2.items()):
        if h1 == h2:
            combined[h1] = combined.get(h1, 0.0) + v1 * v2
        else:
            conflict += v1 * v2

    normaliser = 1.0 - conflict
    if normaliser < 1e-10:
        # Complete conflict — return uniform distribution over all seen hypotheses
        all_hyps = list(set(m1) | set(m2))
        return {h: 1.0 / len(all_hyps) for h in all_hyps}

    return {h: v / normaliser for h, v in combined.items()}


def combine_multiple_bpas(bpas: List[Dict[str, float]]) -> Dict[str, float]:
    """Sequentially combine a list of BPAs."""
    if not bpas:
        return {}
    result = normalise_bpa(bpas[0])
    for bpa in bpas[1:]:
        result = combine_bpa(result, normalise_bpa(bpa))
    return result


def belief(bpa: Dict[str, float], hypothesis: str, frame: List[str]) -> float:
    """Bel(A) = sum of m(B) for all B strictly subset of A (atomic case: B == hypothesis)."""
    return bpa.get(hypothesis, 0.0)


def plausibility(bpa: Dict[str, float], hypothesis: str, frame: List[str]) -> float:
    """Pl(A) = 1 - Bel(complement of A)."""
    complement_belief = sum(
        v for h, v in bpa.items() if h != hypothesis
    )
    return 1.0 - complement_belief


@dataclass
class BeliefInterval:
    hypothesis: str
    belief: float
    plausibility: float

    @classmethod
    def from_bpa(
        cls, bpa: Dict[str, float], hypothesis: str, frame: List[str]
    ) -> "BeliefInterval":
        bel = belief(bpa, hypothesis, frame)
        pla = plausibility(bpa, hypothesis, frame)
        return cls(hypothesis=hypothesis, belief=bel, plausibility=pla)

    def __str__(self) -> str:
        return f"{self.hypothesis}: [{self.belief:.3f}, {self.plausibility:.3f}]"
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_dempster_shafer.py -v
```

**Step 5: Commit**

```bash
git add core/dempster_shafer.py tests/test_dempster_shafer.py
git commit -m "feat: implement Dempster-Shafer evidential reasoning"
```

---

## Task 4: Skill Repository (JSON Skills)

**Files:**
- Create: `C:/MultiAgent/core/skill_repository.py`
- Create: `C:/MultiAgent/skills/structural_integrity.json`
- Create: `C:/MultiAgent/skills/maintenance_management.json`
- Create: `C:/MultiAgent/skills/environmental_hazard.json`
- Create: `C:/MultiAgent/skills/human_factors.json`
- Create: `C:/MultiAgent/skills/system_integration.json`
- Create: `C:/MultiAgent/tests/test_skill_repository.py`

**Background:** Each Skill encodes `S = ⟨τ, P, F, T, B⟩`. The JSON stores: domain id, linguistic variables (TFN triples), inference templates (conditions→output with weights), and benchmark metadata. The repository loads, validates, saves, and versions Skills.

**Step 1: Write failing tests**

```python
# tests/test_skill_repository.py
import pytest
from core.skill_repository import SkillRepository, Skill

def test_load_all_five_skills():
    repo = SkillRepository("C:/MultiAgent/skills")
    skills = repo.load_all()
    assert len(skills) == 5
    ids = {s.domain_id for s in skills}
    assert ids == {"SI", "MM", "EH", "HF", "SysI"}

def test_skill_has_required_fields():
    repo = SkillRepository("C:/MultiAgent/skills")
    skill = repo.load("SI")
    assert skill.domain_id == "SI"
    assert skill.domain_name
    assert skill.linguistic_variables
    assert skill.inference_templates
    assert skill.benchmark

def test_skill_version_increments_on_save(tmp_path):
    repo = SkillRepository("C:/MultiAgent/skills")
    skill = repo.load("SI")
    original_version = skill.benchmark["version"]
    skill.benchmark["version"] += 1
    repo.save(skill, str(tmp_path))
    reloaded = SkillRepository(str(tmp_path)).load("SI")
    assert reloaded.benchmark["version"] == original_version + 1

def test_get_inference_templates_for_skill():
    repo = SkillRepository("C:/MultiAgent/skills")
    skill = repo.load("MM")
    assert len(skill.inference_templates) >= 1
    for t in skill.inference_templates:
        assert "id" in t
        assert "conditions" in t
        assert "output" in t
        assert "weight" in t
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_skill_repository.py -v
```

**Step 3: Create skill JSON files**

`skills/structural_integrity.json`:
```json
{
  "domain_id": "SI",
  "domain_name": "Structural Integrity",
  "version": "1.0",
  "hazard_keywords": ["propeller", "blade", "fatigue", "corrosion", "mechanical wear", "thruster", "bearing", "seal", "vibration"],
  "linguistic_variables": {
    "corrosion_rate":        {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "fatigue_level":         {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "inspection_frequency":  {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "risk_score":            {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]}
  },
  "inference_templates": [
    {"id": "T_SI_1", "conditions": {"corrosion_rate": "high", "fatigue_level": "high"},  "output": "high",   "weight": 0.90},
    {"id": "T_SI_2", "conditions": {"corrosion_rate": "high", "fatigue_level": "medium"},"output": "high",   "weight": 0.80},
    {"id": "T_SI_3", "conditions": {"corrosion_rate": "medium","fatigue_level": "high"}, "output": "medium", "weight": 0.75},
    {"id": "T_SI_4", "conditions": {"corrosion_rate": "low",   "inspection_frequency": "low"}, "output": "medium", "weight": 0.65},
    {"id": "T_SI_5", "conditions": {"corrosion_rate": "low",   "fatigue_level": "low"},  "output": "low",    "weight": 0.85}
  ],
  "routing_confidence": 0.92,
  "benchmark": {
    "expert_agreement_rate": 0.92,
    "mean_deviation": 0.08,
    "version": 1,
    "approval_date": "2024-01-15"
  }
}
```

`skills/maintenance_management.json`:
```json
{
  "domain_id": "MM",
  "domain_name": "Maintenance Management",
  "version": "1.0",
  "hazard_keywords": ["maintenance", "overdue", "work order", "inspection interval", "degradation", "schedule"],
  "linguistic_variables": {
    "maintenance_frequency": {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "equipment_condition":   {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "overdue_tasks":         {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "risk_score":            {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]}
  },
  "inference_templates": [
    {"id": "T_MM_1", "conditions": {"overdue_tasks": "high",   "maintenance_frequency": "low"},    "output": "high",   "weight": 0.92},
    {"id": "T_MM_2", "conditions": {"equipment_condition": "low", "overdue_tasks": "medium"},       "output": "high",   "weight": 0.85},
    {"id": "T_MM_3", "conditions": {"maintenance_frequency": "medium", "equipment_condition": "medium"}, "output": "medium", "weight": 0.70},
    {"id": "T_MM_4", "conditions": {"maintenance_frequency": "high",  "equipment_condition": "high"}, "output": "low",    "weight": 0.88}
  ],
  "routing_confidence": 0.88,
  "benchmark": {
    "expert_agreement_rate": 0.89,
    "mean_deviation": 0.11,
    "version": 1,
    "approval_date": "2024-01-15"
  }
}
```

`skills/environmental_hazard.json`:
```json
{
  "domain_id": "EH",
  "domain_name": "Environmental Hazard",
  "version": "1.0",
  "hazard_keywords": ["wave", "wind", "sea state", "metocean", "weather", "current", "storm", "swell"],
  "linguistic_variables": {
    "wave_height":    {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "wind_speed":     {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "current_speed":  {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "risk_score":     {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]}
  },
  "inference_templates": [
    {"id": "T_EH_1", "conditions": {"wave_height": "high",   "wind_speed": "high"},    "output": "high",   "weight": 0.88},
    {"id": "T_EH_2", "conditions": {"wave_height": "high",   "current_speed": "medium"}, "output": "high", "weight": 0.78},
    {"id": "T_EH_3", "conditions": {"wave_height": "medium", "wind_speed": "medium"},  "output": "medium", "weight": 0.72},
    {"id": "T_EH_4", "conditions": {"wave_height": "low",    "wind_speed": "low"},     "output": "low",    "weight": 0.90}
  ],
  "routing_confidence": 0.85,
  "benchmark": {
    "expert_agreement_rate": 0.87,
    "mean_deviation": 0.13,
    "version": 1,
    "approval_date": "2024-01-15"
  }
}
```

`skills/human_factors.json`:
```json
{
  "domain_id": "HF",
  "domain_name": "Human Factors",
  "version": "1.0",
  "hazard_keywords": ["operator", "training", "procedure", "compliance", "fatigue", "error", "human reliability", "crew"],
  "linguistic_variables": {
    "training_adequacy":      {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "procedural_compliance":  {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "operator_fatigue":       {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "risk_score":             {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]}
  },
  "inference_templates": [
    {"id": "T_HF_1", "conditions": {"operator_fatigue": "high",  "procedural_compliance": "low"},  "output": "high",   "weight": 0.91},
    {"id": "T_HF_2", "conditions": {"training_adequacy": "low",  "procedural_compliance": "low"},  "output": "high",   "weight": 0.87},
    {"id": "T_HF_3", "conditions": {"training_adequacy": "medium","operator_fatigue": "medium"},   "output": "medium", "weight": 0.73},
    {"id": "T_HF_4", "conditions": {"training_adequacy": "high", "procedural_compliance": "high"}, "output": "low",    "weight": 0.89}
  ],
  "routing_confidence": 0.91,
  "benchmark": {
    "expert_agreement_rate": 0.91,
    "mean_deviation": 0.09,
    "version": 1,
    "approval_date": "2024-01-15"
  }
}
```

`skills/system_integration.json`:
```json
{
  "domain_id": "SysI",
  "domain_name": "System Integration",
  "version": "1.0",
  "hazard_keywords": ["dynamic positioning", "DP", "PRS", "DGPS", "control system", "sensor", "redundancy", "software fault", "position reference"],
  "linguistic_variables": {
    "redundancy_level":    {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "sensor_reliability":  {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "integration_complexity": {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]},
    "risk_score":          {"low": [0.0, 0.0, 0.4], "medium": [0.2, 0.5, 0.8], "high": [0.6, 1.0, 1.0]}
  },
  "inference_templates": [
    {"id": "T_SysI_1", "conditions": {"redundancy_level": "low",   "sensor_reliability": "low"},    "output": "high",   "weight": 0.95},
    {"id": "T_SysI_2", "conditions": {"integration_complexity": "high", "redundancy_level": "low"}, "output": "high",   "weight": 0.88},
    {"id": "T_SysI_3", "conditions": {"sensor_reliability": "medium", "redundancy_level": "medium"}, "output": "medium", "weight": 0.74},
    {"id": "T_SysI_4", "conditions": {"redundancy_level": "high",  "sensor_reliability": "high"},   "output": "low",    "weight": 0.91}
  ],
  "routing_confidence": 0.95,
  "benchmark": {
    "expert_agreement_rate": 0.95,
    "mean_deviation": 0.05,
    "version": 1,
    "approval_date": "2024-01-15"
  }
}
```

**Step 4: Implement skill_repository.py**

```python
# core/skill_repository.py
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class Skill:
    domain_id: str
    domain_name: str
    version: str
    hazard_keywords: List[str]
    linguistic_variables: Dict[str, Dict[str, List[float]]]
    inference_templates: List[Dict[str, Any]]
    routing_confidence: float
    benchmark: Dict[str, Any]

    def get_tfn_terms(self, variable: str):
        """Return TFN objects for a linguistic variable."""
        from core.fuzzy_engine import TFN
        raw = self.linguistic_variables.get(variable, {})
        return {term: TFN(*vals) for term, vals in raw.items()}


class SkillRepository:
    SKILL_FILES = {
        "SI":   "structural_integrity.json",
        "MM":   "maintenance_management.json",
        "EH":   "environmental_hazard.json",
        "HF":   "human_factors.json",
        "SysI": "system_integration.json",
    }

    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir

    def load(self, domain_id: str) -> Skill:
        filename = self.SKILL_FILES[domain_id]
        path = os.path.join(self.skills_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Skill(**data)

    def load_all(self) -> List[Skill]:
        return [self.load(did) for did in self.SKILL_FILES]

    def save(self, skill: Skill, skills_dir: str = None) -> None:
        target_dir = skills_dir or self.skills_dir
        filename = self.SKILL_FILES[skill.domain_id]
        path = os.path.join(target_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(skill.__dict__, f, indent=2)
```

**Step 5: Run tests — verify they pass**

```bash
pytest tests/test_skill_repository.py -v
```

**Step 6: Commit**

```bash
git add skills/ core/skill_repository.py tests/test_skill_repository.py
git commit -m "feat: add Skill Repository with 5 domain skill JSON files"
```

---

## Task 5: Vector Store (RAG Backend)

**Files:**
- Create: `C:/MultiAgent/core/vector_store.py`

**Background:** ChromaDB stores document chunks with embeddings. The `sentence-transformers` model `all-MiniLM-L6-v2` generates embeddings locally (no API call). The store supports `add_documents`, `query`, and `reset` operations.

**Step 1: Implement vector_store.py**

```python
# core/vector_store.py
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count() or 1),
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
        )
```

**Step 2: Smoke-test the vector store manually**

```python
# Run in Python REPL
from core.vector_store import VectorStore
vs = VectorStore()
vs.add_documents(
    documents=["High corrosion rates detected on CPP blades during inspection"],
    metadatas=[{"source": "test", "domain": "SI"}],
    ids=["test_001"]
)
print(vs.query("blade corrosion fatigue"))
```
Expected: returns the test document with metadata.

**Step 3: Commit**

```bash
git add core/vector_store.py
git commit -m "feat: add ChromaDB vector store wrapper"
```

---

## Task 6: RAG Agent (L2)

**Files:**
- Create: `C:/MultiAgent/agents/rag_agent.py`

**Background:** The RAG Agent (L2) takes a user query + active Skill hint, formulates a Skill-aware retrieval query, fetches top-k documents from the vector store, and uses Claude to extract structured evidence (risk factors, linguistic variable values, belief masses).

**Step 1: Implement rag_agent.py**

```python
# agents/rag_agent.py
import json
import anthropic
from core.vector_store import VectorStore
from config import ANTHROPIC_API_KEY, MODEL, MAX_TOKENS

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


SYSTEM_PROMPT = """You are a Skill-Aware RAG Agent for offshore risk analysis.
Given a user query and retrieved documents, extract structured evidence for risk assessment.
You must return a JSON object with:
- "risk_factors": list of identified risk factors with severity (0-1 scale)
- "linguistic_assignments": dict mapping variable names to linguistic terms (low/medium/high)
- "belief_masses": dict mapping risk levels (Low/Medium/High) to probability mass (sum to 1.0)
- "evidence_summary": 2-3 sentence summary of key findings
- "source_documents": list of relevant source snippets used

Return ONLY valid JSON. No markdown. No explanation outside the JSON."""


def run_rag_agent(query: str, skill_hint: str = None, n_docs: int = 5) -> dict:
    """
    L2: Skill-Aware RAG Agent.
    query: user's risk question
    skill_hint: domain_id of primary Skill (e.g. "SI") to bias retrieval
    Returns structured evidence dict.
    """
    vs = VectorStore()

    # Skill-aware query augmentation
    augmented_query = query
    if skill_hint:
        augmented_query = f"[{skill_hint}] {query}"

    retrieved = vs.query(augmented_query, n_results=n_docs)

    if not retrieved:
        context = "No documents found in knowledge base."
    else:
        context = "\n\n".join(
            f"[Source {i+1}] {doc['text']}" for i, doc in enumerate(retrieved)
        )

    user_message = f"""Query: {query}

Retrieved Context:
{context}

Extract structured risk evidence from the above context for the query."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return minimal structure
        return {
            "risk_factors": [],
            "linguistic_assignments": {},
            "belief_masses": {"Low": 0.33, "Medium": 0.34, "High": 0.33},
            "evidence_summary": raw[:500],
            "source_documents": [d["text"][:200] for d in retrieved],
        }
```

**Step 2: Commit**

```bash
git add agents/rag_agent.py
git commit -m "feat: implement Skill-Aware RAG Agent (L2)"
```

---

## Task 7: Skill Router Agent (L3)

**Files:**
- Create: `C:/MultiAgent/agents/skill_router.py`
- Create: `C:/MultiAgent/tests/test_skill_router.py`

**Background:** The Skill Router reads the query + RAG evidence and selects which Skill Agent(s) to activate, with confidence scores. Primary Skill + up to 2 secondary Skills are returned. Uses keyword matching + Claude reasoning.

**Step 1: Write failing tests**

```python
# tests/test_skill_router.py
import pytest
from agents.skill_router import route_skills

def test_routes_cpp_to_structural_integrity():
    query = "CPP blade fatigue and corrosion failure"
    evidence = {"risk_factors": [{"name": "blade corrosion", "severity": 0.8}],
                "linguistic_assignments": {"corrosion_rate": "high"}}
    result = route_skills(query, evidence)
    assert result["primary_skill"] == "SI"
    assert result["confidence"] > 0.7

def test_routes_dp_to_system_integration():
    query = "Dynamic positioning system software fault"
    evidence = {"risk_factors": [{"name": "DP failure", "severity": 0.9}],
                "linguistic_assignments": {}}
    result = route_skills(query, evidence)
    assert result["primary_skill"] == "SysI"

def test_returns_secondary_skills():
    query = "Operator error during DP manual override"
    evidence = {"risk_factors": [], "linguistic_assignments": {}}
    result = route_skills(query, evidence)
    assert "secondary_skills" in result
    assert isinstance(result["secondary_skills"], list)

def test_routing_result_has_required_keys():
    result = route_skills("maintenance overdue on CPP", {})
    assert all(k in result for k in ["primary_skill", "secondary_skills", "confidence", "routing_rationale"])
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_skill_router.py -v
```

**Step 3: Implement skill_router.py**

```python
# agents/skill_router.py
import json
import re
import anthropic
from config import ANTHROPIC_API_KEY, MODEL, MAX_TOKENS

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Keyword-based routing table (from Table 4 of the paper)
ROUTING_TABLE = {
    "SI":   ["propeller", "blade", "pitch", "fatigue", "corrosion", "thruster", "bearing", "seal", "vibration", "structural"],
    "MM":   ["maintenance", "overdue", "inspection", "work order", "degradation", "schedule", "lubrication"],
    "EH":   ["wave", "wind", "sea state", "metocean", "weather", "current", "storm", "swell", "environmental"],
    "HF":   ["operator", "training", "procedure", "compliance", "fatigue", "error", "human", "crew", "manual override"],
    "SysI": ["dynamic positioning", "dp ", "prs", "dgps", "sensor", "redundancy", "software", "control system", "position reference"],
}

SKILL_NAMES = {
    "SI": "Structural Integrity",
    "MM": "Maintenance Management",
    "EH": "Environmental Hazard",
    "HF": "Human Factors",
    "SysI": "System Integration",
}

SYSTEM_PROMPT = """You are a Skill Router for an offshore risk analysis expert system.
Given a query and extracted evidence, select the most appropriate domain Skills to activate.
Available Skills: SI (Structural Integrity), MM (Maintenance Management), EH (Environmental Hazard), HF (Human Factors), SysI (System Integration).
Return ONLY valid JSON with keys: primary_skill, secondary_skills (list, max 2), confidence (0-1), routing_rationale (one sentence).
No markdown. No explanation outside JSON."""


def _keyword_score(text: str) -> dict:
    text_lower = text.lower()
    scores = {}
    for skill_id, keywords in ROUTING_TABLE.items():
        scores[skill_id] = sum(1 for kw in keywords if kw in text_lower)
    return scores


def route_skills(query: str, evidence: dict) -> dict:
    """
    L3: Skill Router Agent.
    Returns primary_skill, secondary_skills, confidence, routing_rationale.
    """
    # Fast keyword pre-filter
    combined_text = query + " " + json.dumps(evidence.get("linguistic_assignments", {}))
    kw_scores = _keyword_score(combined_text)
    kw_ranked = sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)

    # Build hint for LLM
    hint = ", ".join(f"{sid}={score}" for sid, score in kw_ranked if score > 0) or "no keyword matches"

    user_message = f"""Query: {query}

Extracted Evidence: {json.dumps(evidence, indent=2)[:1000]}

Keyword match scores: {hint}

Select the appropriate Skills to activate."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        result = json.loads(raw)
        # Ensure required keys
        result.setdefault("secondary_skills", [])
        result.setdefault("confidence", 0.5)
        result.setdefault("routing_rationale", "Keyword-based routing.")
        return result
    except json.JSONDecodeError:
        # Fallback to top keyword match
        top_skill = kw_ranked[0][0] if kw_ranked and kw_ranked[0][1] > 0 else "SI"
        return {
            "primary_skill": top_skill,
            "secondary_skills": [],
            "confidence": 0.5,
            "routing_rationale": "Fallback to keyword routing.",
        }
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_skill_router.py -v
```

**Step 5: Commit**

```bash
git add agents/skill_router.py tests/test_skill_router.py
git commit -m "feat: implement Skill Router Agent (L3) with keyword + LLM routing"
```

---

## Task 8: Base Skill Agent + 5 Domain Skill Agents (L4)

**Files:**
- Create: `C:/MultiAgent/agents/skill_agents/base_skill_agent.py`
- Create: `C:/MultiAgent/agents/skill_agents/si_agent.py`
- Create: `C:/MultiAgent/agents/skill_agents/mm_agent.py`
- Create: `C:/MultiAgent/agents/skill_agents/eh_agent.py`
- Create: `C:/MultiAgent/agents/skill_agents/hf_agent.py`
- Create: `C:/MultiAgent/agents/skill_agents/sysi_agent.py`
- Create: `C:/MultiAgent/tests/test_skill_agents.py`

**Background:** Each Skill Agent (a) loads its Skill JSON, (b) fuzzifies the RAG evidence using the Skill's linguistic variables, (c) runs Mamdani fuzzy inference using the Skill's weighted templates, (d) uses Claude to reason over the results and assign belief masses, (e) returns a `SkillResult` with risk_score, belief_masses, and explanation.

**Step 1: Write failing tests**

```python
# tests/test_skill_agents.py
import pytest
from agents.skill_agents.base_skill_agent import run_skill_agent, SkillResult

def test_skill_agent_returns_skill_result():
    evidence = {
        "linguistic_assignments": {"corrosion_rate": "high", "fatigue_level": "medium"},
        "belief_masses": {"Low": 0.1, "Medium": 0.3, "High": 0.6},
        "risk_factors": [{"name": "blade corrosion", "severity": 0.8}],
    }
    result = run_skill_agent("SI", evidence)
    assert isinstance(result, SkillResult)
    assert 0.0 <= result.risk_score <= 1.0
    assert abs(sum(result.belief_masses.values()) - 1.0) < 0.05
    assert result.domain_id == "SI"
    assert result.explanation

def test_all_five_skill_agents_run():
    evidence = {
        "linguistic_assignments": {},
        "belief_masses": {"Low": 0.33, "Medium": 0.34, "High": 0.33},
        "risk_factors": [],
    }
    for domain_id in ["SI", "MM", "EH", "HF", "SysI"]:
        result = run_skill_agent(domain_id, evidence)
        assert isinstance(result, SkillResult)
        assert result.domain_id == domain_id

def test_high_risk_evidence_produces_high_score():
    evidence = {
        "linguistic_assignments": {"corrosion_rate": "high", "fatigue_level": "high"},
        "belief_masses": {"Low": 0.05, "Medium": 0.15, "High": 0.80},
        "risk_factors": [{"name": "severe corrosion", "severity": 0.95}],
    }
    result = run_skill_agent("SI", evidence)
    assert result.risk_score > 0.55
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_skill_agents.py -v
```

**Step 3: Implement base_skill_agent.py**

```python
# agents/skill_agents/base_skill_agent.py
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any
import anthropic
from core.skill_repository import SkillRepository, Skill
from core.fuzzy_engine import fuzzify, mamdani_inference, STANDARD_TERMS
from config import ANTHROPIC_API_KEY, MODEL, MAX_TOKENS, SKILLS_DIR

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


@dataclass
class SkillResult:
    domain_id: str
    domain_name: str
    risk_score: float                          # defuzzified crisp score 0-1
    belief_masses: Dict[str, float]            # {Low, Medium, High} BPA
    explanation: str                           # Claude's chain of reasoning
    activated_templates: list = field(default_factory=list)
    skill_version: int = 1


SYSTEM_PROMPT_TEMPLATE = """You are the {domain_name} Skill Agent in an offshore risk analysis expert system.
Your role: given fuzzy inference results and extracted evidence, produce a calibrated belief mass assignment.
Return ONLY valid JSON with keys:
- "belief_masses": {{"Low": float, "Medium": float, "High": float}} (must sum to 1.0)
- "explanation": string (2-3 sentences explaining the risk reasoning)
- "activated_templates": list of template IDs that fired with strength > 0.1
No markdown. No text outside JSON."""


def run_skill_agent(domain_id: str, evidence: dict) -> SkillResult:
    """
    L4: Run a single domain Skill Agent.
    domain_id: one of SI, MM, EH, HF, SysI
    evidence: output from RAG Agent
    """
    repo = SkillRepository(SKILLS_DIR)
    skill = repo.load(domain_id)

    # Step 1: Fuzzify evidence using Skill's linguistic variables
    linguistic_assignments = evidence.get("linguistic_assignments", {})
    input_memberships: Dict[str, Dict[str, float]] = {}

    for var_name, terms_config in skill.linguistic_variables.items():
        if var_name == "risk_score":
            continue
        if var_name in linguistic_assignments:
            # Convert linguistic assignment to membership (assigned term gets full membership)
            assigned_term = linguistic_assignments[var_name]
            from core.fuzzy_engine import TFN
            terms = {t: TFN(*vals) for t, vals in terms_config.items()}
            memberships = {t: 0.0 for t in terms}
            if assigned_term in memberships:
                memberships[assigned_term] = 1.0
            else:
                # Partial: all medium
                memberships = {t: (1.0 if t == "medium" else 0.0) for t in terms}
            input_memberships[var_name] = memberships

    # Step 2: Mamdani fuzzy inference
    from core.fuzzy_engine import TFN
    output_terms = {
        t: TFN(*vals)
        for t, vals in skill.linguistic_variables.get("risk_score", {
            "low": [0.0, 0.0, 0.4],
            "medium": [0.2, 0.5, 0.8],
            "high": [0.6, 1.0, 1.0],
        }).items()
    }

    risk_score = mamdani_inference(skill.inference_templates, input_memberships, output_terms)

    # Step 3: Claude reasoning over fuzzy results → belief mass assignment
    rag_bpa = evidence.get("belief_masses", {"Low": 0.33, "Medium": 0.34, "High": 0.33})

    user_message = f"""Domain: {skill.domain_name}
Fuzzy inference risk score: {risk_score:.3f} (0=low, 1=high)
RAG-extracted belief masses: {json.dumps(rag_bpa)}
Input linguistic assignments: {json.dumps(linguistic_assignments)}
Risk factors: {json.dumps(evidence.get('risk_factors', [])[:3])}
Inference templates available: {[t['id'] for t in skill.inference_templates]}

Produce calibrated belief masses and explain the {skill.domain_name} risk assessment."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT_TEMPLATE.format(domain_name=skill.domain_name),
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = json.loads(raw)
        bpa = parsed.get("belief_masses", {"Low": 0.33, "Medium": 0.34, "High": 0.33})
        # Normalise
        total = sum(bpa.values())
        bpa = {k: v / total for k, v in bpa.items()}
        return SkillResult(
            domain_id=domain_id,
            domain_name=skill.domain_name,
            risk_score=risk_score,
            belief_masses=bpa,
            explanation=parsed.get("explanation", ""),
            activated_templates=parsed.get("activated_templates", []),
            skill_version=skill.benchmark.get("version", 1),
        )
    except json.JSONDecodeError:
        return SkillResult(
            domain_id=domain_id,
            domain_name=skill.domain_name,
            risk_score=risk_score,
            belief_masses={"Low": 1 - risk_score, "Medium": 0.0, "High": risk_score},
            explanation=raw[:300],
            skill_version=skill.benchmark.get("version", 1),
        )
```

**Step 4: Create thin wrappers for each domain (si_agent.py pattern)**

```python
# agents/skill_agents/si_agent.py
from agents.skill_agents.base_skill_agent import run_skill_agent, SkillResult

def run(evidence: dict) -> SkillResult:
    return run_skill_agent("SI", evidence)
```

Repeat for `mm_agent.py` ("MM"), `eh_agent.py` ("EH"), `hf_agent.py` ("HF"), `sysi_agent.py` ("SysI").

**Step 5: Run tests — verify they pass**

```bash
pytest tests/test_skill_agents.py -v
```

**Step 6: Commit**

```bash
git add agents/skill_agents/ tests/test_skill_agents.py
git commit -m "feat: implement 5 parallel Skill Agents (L4) with fuzzy inference + Claude reasoning"
```

---

## Task 9: Synthesis Agent — Dempster-Shafer (L5)

**Files:**
- Create: `C:/MultiAgent/agents/synthesis_agent.py`
- Create: `C:/MultiAgent/tests/test_synthesis_agent.py`

**Background:** Takes results from all activated Skill Agents, applies Skill-informed weights (from `routing_confidence`), combines BPAs via Dempster's rule, computes belief/plausibility intervals per risk level, and returns ranked risk assessment.

**Step 1: Write failing tests**

```python
# tests/test_synthesis_agent.py
import pytest
from agents.skill_agents.base_skill_agent import SkillResult
from agents.synthesis_agent import synthesise

def make_result(domain_id, risk_score, bpa):
    return SkillResult(domain_id=domain_id, domain_name=domain_id,
                       risk_score=risk_score, belief_masses=bpa, explanation="test")

def test_synthesis_returns_combined_bpa():
    results = [
        make_result("SI",   0.7, {"Low": 0.1, "Medium": 0.2, "High": 0.7}),
        make_result("MM",   0.8, {"Low": 0.05,"Medium": 0.15,"High": 0.8}),
    ]
    output = synthesise(results)
    assert "combined_bpa" in output
    assert abs(sum(output["combined_bpa"].values()) - 1.0) < 0.05

def test_synthesis_returns_belief_intervals():
    results = [make_result("SI", 0.5, {"Low": 0.3, "Medium": 0.4, "High": 0.3})]
    output = synthesise(results)
    assert "belief_intervals" in output
    for level in ["Low", "Medium", "High"]:
        assert level in output["belief_intervals"]

def test_synthesis_identifies_dominant_factor():
    results = [
        make_result("SI",   0.3, {"Low": 0.7, "Medium": 0.2, "High": 0.1}),
        make_result("MM",   0.9, {"Low": 0.05,"Medium": 0.1, "High": 0.85}),
    ]
    output = synthesise(results)
    assert output["dominant_factor"] == "MM"

def test_synthesis_produces_risk_ranking():
    results = [
        make_result("SI",  0.6, {"Low": 0.1, "Medium": 0.3, "High": 0.6}),
        make_result("EH",  0.4, {"Low": 0.4, "Medium": 0.4, "High": 0.2}),
        make_result("HF",  0.7, {"Low": 0.05,"Medium": 0.2, "High": 0.75}),
    ]
    output = synthesise(results)
    assert "risk_ranking" in output
    assert len(output["risk_ranking"]) == 3
```

**Step 2: Run tests — verify they fail**

```bash
pytest tests/test_synthesis_agent.py -v
```

**Step 3: Implement synthesis_agent.py**

```python
# agents/synthesis_agent.py
from typing import List, Dict, Any
from agents.skill_agents.base_skill_agent import SkillResult
from core.dempster_shafer import combine_multiple_bpas, BeliefInterval, normalise_bpa

FRAME = ["Low", "Medium", "High"]


def synthesise(skill_results: List[SkillResult]) -> Dict[str, Any]:
    """
    L5: Skill-Guided Synthesis using Dempster-Shafer combination.
    Applies Skill-informed weights before combining BPAs.
    """
    if not skill_results:
        return {
            "combined_bpa": {"Low": 0.33, "Medium": 0.34, "High": 0.33},
            "belief_intervals": {},
            "dominant_factor": None,
            "risk_ranking": [],
        }

    # Load routing confidences as Skill-informed weights
    from core.skill_repository import SkillRepository
    from config import SKILLS_DIR
    repo = SkillRepository(SKILLS_DIR)

    weighted_bpas = []
    for result in skill_results:
        try:
            skill = repo.load(result.domain_id)
            weight = skill.routing_confidence
        except Exception:
            weight = 0.5

        # Apply weight: blend BPA toward uniform by (1 - weight)
        uniform = {k: 1.0 / len(FRAME) for k in FRAME}
        bpa = result.belief_masses
        weighted = {
            k: weight * bpa.get(k, 0.0) + (1 - weight) * uniform[k]
            for k in FRAME
        }
        weighted_bpas.append(normalise_bpa(weighted))

    combined = combine_multiple_bpas(weighted_bpas)

    # Belief/plausibility intervals
    from core.dempster_shafer import belief, plausibility
    intervals = {
        level: BeliefInterval.from_bpa(combined, level, FRAME)
        for level in FRAME
    }

    # Risk ranking by risk_score (highest risk first)
    ranking = sorted(skill_results, key=lambda r: r.risk_score, reverse=True)

    # Dominant factor = highest High belief
    dominant = max(skill_results, key=lambda r: r.belief_masses.get("High", 0.0))

    return {
        "combined_bpa": combined,
        "belief_intervals": {k: str(v) for k, v in intervals.items()},
        "dominant_factor": dominant.domain_id,
        "dominant_factor_name": dominant.domain_name,
        "dominant_belief_high": dominant.belief_masses.get("High", 0.0),
        "risk_ranking": [
            {
                "rank": i + 1,
                "domain_id": r.domain_id,
                "domain_name": r.domain_name,
                "risk_score": round(r.risk_score, 3),
                "high_belief": round(r.belief_masses.get("High", 0.0), 3),
            }
            for i, r in enumerate(ranking)
        ],
        "skill_results": skill_results,
    }
```

**Step 4: Run tests — verify they pass**

```bash
pytest tests/test_synthesis_agent.py -v
```

**Step 5: Commit**

```bash
git add agents/synthesis_agent.py tests/test_synthesis_agent.py
git commit -m "feat: implement Synthesis Agent (L5) with Dempster-Shafer fusion"
```

---

## Task 10: Validator Agent — Human-in-the-Loop (L4 gate)

**Files:**
- Create: `C:/MultiAgent/agents/validator_agent.py`

**Background:** The Validator Agent presents the synthesis results to a human expert for approval. It uses Claude to generate a validation report (flagging high-uncertainty or high-conflict findings), then pauses for human input (`input()`). The expert can: approve, reject (with reason), or request re-analysis.

**Step 1: Implement validator_agent.py**

```python
# agents/validator_agent.py
import json
import anthropic
from typing import Dict, Any
from config import ANTHROPIC_API_KEY, MODEL, MAX_TOKENS

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are an Expert Validation Agent for an offshore risk analysis system.
Generate a concise validation report for the human expert to review.
Highlight: (1) high-uncertainty findings (wide belief-plausibility intervals), (2) dominant risk factors, (3) any reasoning inconsistencies.
Return JSON with keys: validation_summary (string), flags (list of strings), recommendation (approve/review/reject), confidence_note (string)."""


def run_validator(synthesis_output: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Expert Validation Checkpoint (L4 gate).
    Returns validated synthesis output with expert decision recorded.
    """
    user_message = f"""Query: {query}

Synthesis Results:
- Combined BPA: {json.dumps(synthesis_output['combined_bpa'], indent=2)}
- Dominant Factor: {synthesis_output.get('dominant_factor_name')} ({synthesis_output.get('dominant_belief_high', 0):.1%} High belief)
- Risk Ranking: {json.dumps(synthesis_output['risk_ranking'], indent=2)}
- Belief Intervals: {json.dumps(synthesis_output['belief_intervals'], indent=2)}

Generate validation report."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    import re
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        validation_report = json.loads(raw)
    except json.JSONDecodeError:
        validation_report = {"validation_summary": raw, "flags": [], "recommendation": "review", "confidence_note": ""}

    # ─── HUMAN-IN-THE-LOOP PAUSE ───
    print("\n" + "="*60)
    print("EXPERT VALIDATION CHECKPOINT")
    print("="*60)
    print(f"\nValidation Summary: {validation_report.get('validation_summary', '')}")
    print(f"\nFlags:")
    for flag in validation_report.get("flags", []):
        print(f"  ⚠  {flag}")
    print(f"\nAI Recommendation: {validation_report.get('recommendation', 'review').upper()}")
    print(f"\nRisk Ranking:")
    for item in synthesis_output.get("risk_ranking", []):
        print(f"  #{item['rank']} {item['domain_name']}: score={item['risk_score']:.3f}, High belief={item['high_belief']:.3f}")
    print("\n" + "-"*60)
    print("Options: [A]pprove  [R]eject  [M]odify weights  [S]kip")
    expert_input = input("Expert decision: ").strip().upper()
    # ──────────────────────────────

    expert_decision = {
        "A": "approved",
        "R": "rejected",
        "M": "modify_requested",
        "S": "skipped",
    }.get(expert_input, "approved")

    expert_note = ""
    if expert_decision in ("rejected", "modify_requested"):
        expert_note = input("Please provide reason/instructions: ").strip()

    synthesis_output["validation"] = {
        "report": validation_report,
        "expert_decision": expert_decision,
        "expert_note": expert_note,
    }

    return synthesis_output
```

**Step 2: Commit**

```bash
git add agents/validator_agent.py
git commit -m "feat: implement Expert Validator Agent with human-in-the-loop checkpoint"
```

---

## Task 11: Output Agent + Feedback Loop (L6)

**Files:**
- Create: `C:/MultiAgent/agents/output_agent.py`
- Create: `C:/MultiAgent/feedback/benchmarking.py`
- Create: `C:/MultiAgent/feedback/skill_updater.py`

**Step 1: Implement output_agent.py**

```python
# agents/output_agent.py
import json
from datetime import datetime
from typing import Dict, Any


def generate_output(query: str, synthesis: Dict[str, Any]) -> Dict[str, Any]:
    """
    L6: Output Agent — generates Risk Rankings, Risk Profile, Traceability, Skill Trace.
    """
    timestamp = datetime.now().isoformat()
    validation = synthesis.get("validation", {})

    output = {
        "timestamp": timestamp,
        "query": query,
        "risk_rankings": synthesis.get("risk_ranking", []),
        "risk_profile": {
            "overall_bpa": synthesis.get("combined_bpa", {}),
            "belief_intervals": synthesis.get("belief_intervals", {}),
            "dominant_factor": synthesis.get("dominant_factor_name", ""),
        },
        "traceability": {
            "skill_results": [
                {
                    "domain_id": r.domain_id,
                    "domain_name": r.domain_name,
                    "risk_score": r.risk_score,
                    "belief_masses": r.belief_masses,
                    "activated_templates": r.activated_templates,
                    "skill_version": r.skill_version,
                }
                for r in synthesis.get("skill_results", [])
            ]
        },
        "skill_trace": {
            "routing_decision": synthesis.get("routing_decision", {}),
            "expert_validation": validation.get("expert_decision", "not_validated"),
            "expert_note": validation.get("expert_note", ""),
        },
    }

    print("\n" + "="*60)
    print("RISK ANALYSIS OUTPUT")
    print("="*60)
    print(f"\nQuery: {query}")
    print(f"\nDominant Risk Factor: {output['risk_profile']['dominant_factor']}")
    print(f"\nRisk Rankings:")
    for item in output["risk_rankings"]:
        bar = "█" * int(item["risk_score"] * 20)
        print(f"  #{item['rank']} {item['domain_name']:30s} {bar} {item['risk_score']:.3f}")
    print(f"\nBelief Intervals:")
    for level, interval in output["risk_profile"]["belief_intervals"].items():
        print(f"  {level}: {interval}")
    print(f"\nExpert Validation: {output['skill_trace']['expert_validation'].upper()}")
    print("="*60)

    return output
```

**Step 2: Implement feedback/benchmarking.py**

```python
# feedback/benchmarking.py
from typing import Dict, Any


def benchmark_skill_performance(
    skill_result, expert_ground_truth: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compare Skill Agent output against expert ground truth.
    expert_ground_truth: {"Low": float, "Medium": float, "High": float}
    Returns agreement_rate, mean_deviation, per_template_metrics.
    """
    predicted = skill_result.belief_masses
    frame = ["Low", "Medium", "High"]

    deviations = [
        abs(predicted.get(level, 0.0) - expert_ground_truth.get(level, 0.0))
        for level in frame
    ]
    mean_deviation = sum(deviations) / len(deviations)
    agreement_rate = 1.0 - mean_deviation

    dominant_predicted = max(predicted, key=predicted.get)
    dominant_truth = max(expert_ground_truth, key=expert_ground_truth.get)
    dominant_match = dominant_predicted == dominant_truth

    return {
        "domain_id": skill_result.domain_id,
        "agreement_rate": round(agreement_rate, 4),
        "mean_deviation": round(mean_deviation, 4),
        "dominant_match": dominant_match,
        "per_level_deviation": {
            level: round(dev, 4) for level, dev in zip(frame, deviations)
        },
    }
```

**Step 3: Implement feedback/skill_updater.py**

```python
# feedback/skill_updater.py
from typing import Dict, Any
from core.skill_repository import SkillRepository
from config import SKILLS_DIR

LEARNING_RATE = 0.1  # Weight update step size


def propose_weight_updates(
    skill_result, benchmark_result: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute proposed template weight updates based on benchmark performance.
    Increases weights of templates that fired in high-agreement cases.
    Returns {template_id: proposed_new_weight}.
    """
    repo = SkillRepository(SKILLS_DIR)
    skill = repo.load(skill_result.domain_id)

    updates = {}
    for template in skill.inference_templates:
        tid = template["id"]
        current_weight = template["weight"]

        if tid in skill_result.activated_templates:
            if benchmark_result["agreement_rate"] > 0.85:
                # Reinforce: increase weight
                new_weight = min(1.0, current_weight + LEARNING_RATE * benchmark_result["agreement_rate"])
            else:
                # Penalise: decrease weight
                new_weight = max(0.1, current_weight - LEARNING_RATE * benchmark_result["mean_deviation"])
            updates[tid] = round(new_weight, 4)

    return updates


def apply_expert_gated_update(
    domain_id: str,
    proposed_updates: Dict[str, float],
    expert_approved: bool,
    expert_note: str = "",
) -> bool:
    """
    Expert-gated feedback loop: apply weight updates only if expert approves.
    Returns True if update was applied.
    """
    if not expert_approved:
        print(f"  [Feedback] Update REJECTED for {domain_id}: {expert_note}")
        return False

    repo = SkillRepository(SKILLS_DIR)
    skill = repo.load(domain_id)

    for template in skill.inference_templates:
        if template["id"] in proposed_updates:
            old_w = template["weight"]
            template["weight"] = proposed_updates[template["id"]]
            print(f"  [Feedback] {template['id']}: weight {old_w:.3f} → {template['weight']:.3f}")

    skill.benchmark["version"] += 1
    from datetime import date
    skill.benchmark["approval_date"] = str(date.today())

    repo.save(skill)
    print(f"  [Feedback] Skill {domain_id} updated to version {skill.benchmark['version']}")
    return True


def run_expert_gated_feedback(output: Dict[str, Any], expert_ground_truth: Dict[str, Dict[str, float]] = None):
    """
    Full feedback loop: benchmark → propose updates → expert gate → apply.
    expert_ground_truth: {domain_id: {Low: float, Medium: float, High: float}}
    """
    if not expert_ground_truth:
        print("\n[Feedback] No ground truth provided. Skipping feedback loop.")
        return

    from feedback.benchmarking import benchmark_skill_performance

    print("\n" + "="*60)
    print("EXPERT-GATED FEEDBACK LOOP")
    print("="*60)

    for sr in output.get("traceability", {}).get("skill_results", []):
        did = sr["domain_id"]
        if did not in expert_ground_truth:
            continue

        from agents.skill_agents.base_skill_agent import SkillResult
        sr_obj = SkillResult(
            domain_id=did,
            domain_name=sr["domain_name"],
            risk_score=sr["risk_score"],
            belief_masses=sr["belief_masses"],
            explanation="",
            activated_templates=sr.get("activated_templates", []),
            skill_version=sr.get("skill_version", 1),
        )

        benchmark = benchmark_skill_performance(sr_obj, expert_ground_truth[did])
        proposed = propose_weight_updates(sr_obj, benchmark)

        print(f"\n{did} — Agreement: {benchmark['agreement_rate']:.1%}, Mean dev: {benchmark['mean_deviation']:.3f}")
        print(f"  Proposed updates: {proposed}")

        if not proposed:
            continue

        approve = input(f"  Approve updates for {did}? [Y/N]: ").strip().upper() == "Y"
        note = "" if approve else input("  Reason for rejection: ").strip()
        apply_expert_gated_update(did, proposed, approve, note)
```

**Step 4: Commit**

```bash
git add agents/output_agent.py feedback/benchmarking.py feedback/skill_updater.py
git commit -m "feat: implement Output Agent (L6) and expert-gated feedback loop"
```

---

## Task 12: Orchestrator Agent + Async Parallel Execution

**Files:**
- Create: `C:/MultiAgent/agents/orchestrator.py`

**Background:** The Orchestrator coordinates the full pipeline: query → RAG → Skill Router → parallel Skill Agents (asyncio) → Synthesis → Validator → Output. It holds shared pipeline state and handles routing decisions.

**Step 1: Implement orchestrator.py**

```python
# agents/orchestrator.py
import asyncio
from typing import Dict, Any, List
from agents.rag_agent import run_rag_agent
from agents.skill_router import route_skills
from agents.skill_agents.base_skill_agent import run_skill_agent, SkillResult
from agents.synthesis_agent import synthesise
from agents.validator_agent import run_validator
from agents.output_agent import generate_output


async def _run_skill_agent_async(domain_id: str, evidence: dict) -> SkillResult:
    """Run a single Skill Agent in a thread pool (I/O bound due to API calls)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_skill_agent, domain_id, evidence)


async def run_pipeline_async(
    query: str,
    expert_ground_truth: Dict[str, Dict[str, float]] = None,
    run_feedback: bool = False,
) -> Dict[str, Any]:
    """
    Full async pipeline: L1 → L2 → L3 → L4 (parallel) → L5 → Validator → L6.
    """
    print(f"\n{'='*60}")
    print(f"MULTI-AGENT RISK ANALYSIS SYSTEM")
    print(f"{'='*60}")
    print(f"Query: {query}\n")

    # L2: RAG Agent
    print("[L2] Running Skill-Aware RAG Agent...")
    evidence = run_rag_agent(query)

    # L3: Skill Router
    print("[L3] Running Skill Router...")
    routing = route_skills(query, evidence)
    primary = routing["primary_skill"]
    secondary = routing.get("secondary_skills", [])
    active_skills = list({primary} | set(secondary))
    print(f"  → Activated Skills: {active_skills} (confidence: {routing['confidence']:.2f})")

    # L4: Parallel Skill Agents
    print(f"[L4] Running {len(active_skills)} Skill Agent(s) in parallel...")
    tasks = [_run_skill_agent_async(sid, evidence) for sid in active_skills]
    skill_results: List[SkillResult] = await asyncio.gather(*tasks)
    print(f"  → Skill Agents completed: {[r.domain_id for r in skill_results]}")

    # L5: Synthesis
    print("[L5] Running Synthesis Agent (Dempster-Shafer)...")
    synthesis = synthesise(skill_results)
    synthesis["routing_decision"] = routing

    # Expert Validation Checkpoint
    print("[Validator] Invoking Expert Validation Checkpoint...")
    validated = run_validator(synthesis, query)

    # L6: Output
    print("[L6] Generating output...")
    output = generate_output(query, validated)

    # Feedback Loop (optional)
    if run_feedback and expert_ground_truth:
        from feedback.skill_updater import run_expert_gated_feedback
        run_expert_gated_feedback(output, expert_ground_truth)

    return output


def run_pipeline(
    query: str,
    expert_ground_truth: Dict = None,
    run_feedback: bool = False,
) -> Dict[str, Any]:
    """Synchronous wrapper for the async pipeline."""
    return asyncio.run(run_pipeline_async(query, expert_ground_truth, run_feedback))
```

**Step 2: Commit**

```bash
git add agents/orchestrator.py
git commit -m "feat: implement Orchestrator with async parallel Skill Agent execution"
```

---

## Task 13: Main Entry Point + Knowledge Base Seeding

**Files:**
- Create: `C:/MultiAgent/main.py`

**Step 1: Implement main.py**

```python
# main.py
"""
Multi-Agent Skill-Enhanced RAG Risk Analysis System
Entry point: seed knowledge base (first run), then run interactive pipeline.
"""
import sys
from agents.orchestrator import run_pipeline
from core.vector_store import VectorStore

# Sample FPSO risk knowledge documents for seeding
SEED_DOCUMENTS = [
    ("CPP propeller blade fatigue and corrosion are primary structural integrity failure modes on FPSOs. High blade stress combined with corrosive marine environment leads to accelerated degradation.", {"source": "FPSO_safety_manual", "domain": "SI"}, "doc_SI_001"),
    ("Thruster bearing failures are associated with vibration levels exceeding 7mm/s RMS and inadequate lubrication maintenance intervals.", {"source": "thruster_maintenance_guide", "domain": "SI"}, "doc_SI_002"),
    ("Maintenance management failures account for 34% of FPSO incidents. Overdue preventive maintenance tasks are the leading indicator of imminent equipment failure.", {"source": "offshore_incident_database", "domain": "MM"}, "doc_MM_001"),
    ("Dynamic Positioning system failures: redundancy loss in DP class 2 systems increases collision probability during tandem offloading by a factor of 3.5.", {"source": "DP_safety_case", "domain": "SysI"}, "doc_SysI_001"),
    ("DGPS signal loss during tandem offloading operations triggers automatic DP alert. Position reference system (PRS) failures require immediate manual override procedures.", {"source": "DP_operations_manual", "domain": "SysI"}, "doc_SysI_002"),
    ("Significant wave height above 3.5m and wind speed above 25 knots define operability limits for FPSO tandem offloading operations in the North Sea.", {"source": "metocean_operations_guide", "domain": "EH"}, "doc_EH_001"),
    ("Operator fatigue following extended watch-keeping periods is a contributing factor in 28% of DP loss-of-position incidents. Training adequacy scores below 70% correlate with increased error rates.", {"source": "human_factors_report", "domain": "HF"}, "doc_HF_001"),
    ("Structural integrity assessment requires inspection of all load-bearing members at 5-year intervals. Corrosion rates exceeding 0.3mm/year require immediate remedial action.", {"source": "structural_integrity_standard", "domain": "SI"}, "doc_SI_003"),
]


def seed_knowledge_base():
    vs = VectorStore()
    if vs.count() > 0:
        print(f"[KB] Knowledge base already contains {vs.count()} documents. Skipping seed.")
        return
    docs = [d[0] for d in SEED_DOCUMENTS]
    metas = [d[1] for d in SEED_DOCUMENTS]
    ids = [d[2] for d in SEED_DOCUMENTS]
    vs.add_documents(docs, metas, ids)
    print(f"[KB] Seeded knowledge base with {len(SEED_DOCUMENTS)} documents.")


def interactive_mode():
    print("\n" + "="*60)
    print("Multi-Agent Skill-Enhanced RAG Risk Analysis System")
    print("Type 'quit' to exit, 'feedback' after analysis to run feedback loop")
    print("="*60)

    while True:
        query = input("\nEnter risk analysis query: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        run_fb = input("Run feedback loop after analysis? [Y/N]: ").strip().upper() == "Y"
        result = run_pipeline(query, run_feedback=run_fb)
        print("\n[System] Analysis complete. Output saved.")


if __name__ == "__main__":
    seed_knowledge_base()
    if len(sys.argv) > 1:
        # One-shot query mode
        query = " ".join(sys.argv[1:])
        run_pipeline(query)
    else:
        interactive_mode()
```

**Step 2: Run full integration test**

```bash
cd C:/MultiAgent
python main.py "CPP propeller blade showing signs of corrosion and fatigue cracks"
```

Expected: Full pipeline runs — RAG retrieves docs, Skill Router activates SI (+MM), parallel agents produce risk scores, Synthesis identifies dominant factor, Validator checkpoint appears, Output shows risk rankings.

**Step 3: Run all tests**

```bash
pytest tests/ -v
```
Expected: all tests pass.

**Step 4: Final commit**

```bash
git add main.py
git commit -m "feat: add main entry point with knowledge base seeding and interactive mode"
```

---

## Task 14: Integration Test

**Files:**
- Create: `C:/MultiAgent/tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
import pytest
from unittest.mock import patch, MagicMock
from agents.orchestrator import run_pipeline


@pytest.mark.integration
def test_full_pipeline_returns_output_structure():
    """End-to-end pipeline test with mocked API calls."""
    mock_rag_evidence = {
        "risk_factors": [{"name": "blade fatigue", "severity": 0.8}],
        "linguistic_assignments": {"corrosion_rate": "high", "fatigue_level": "high"},
        "belief_masses": {"Low": 0.1, "Medium": 0.2, "High": 0.7},
        "evidence_summary": "High corrosion detected on CPP blades.",
        "source_documents": ["CPP inspection record 2024"]
    }
    mock_routing = {
        "primary_skill": "SI",
        "secondary_skills": ["MM"],
        "confidence": 0.92,
        "routing_rationale": "Structural keywords detected."
    }

    with patch("agents.rag_agent.run_rag_agent", return_value=mock_rag_evidence), \
         patch("agents.skill_router.route_skills", return_value=mock_routing), \
         patch("agents.validator_agent.run_validator", side_effect=lambda s, q: {**s, "validation": {"expert_decision": "approved", "expert_note": ""}}):
        result = run_pipeline("CPP blade corrosion risk assessment")

    assert "risk_rankings" in result
    assert "risk_profile" in result
    assert "traceability" in result
    assert "skill_trace" in result
    assert result["skill_trace"]["expert_validation"] == "approved"
```

**Step 2: Run integration test**

```bash
pytest tests/test_integration.py -v -m integration
```

**Step 3: Final commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for full pipeline"
```

---

## Summary

| Task | Component | Status |
|------|-----------|--------|
| 1 | Project bootstrap + config | ⬜ |
| 2 | Fuzzy Engine (TFN + Mamdani) | ⬜ |
| 3 | Dempster-Shafer engine | ⬜ |
| 4 | Skill Repository (5 JSON Skills) | ⬜ |
| 5 | Vector Store (ChromaDB) | ⬜ |
| 6 | RAG Agent (L2) | ⬜ |
| 7 | Skill Router (L3) | ⬜ |
| 8 | 5 Parallel Skill Agents (L4) | ⬜ |
| 9 | Synthesis Agent — DS (L5) | ⬜ |
| 10 | Validator Agent — HITL | ⬜ |
| 11 | Output Agent + Feedback Loop (L6) | ⬜ |
| 12 | Orchestrator + async parallel | ⬜ |
| 13 | Main entry point + KB seeding | ⬜ |
| 14 | Integration tests | ⬜ |
