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
