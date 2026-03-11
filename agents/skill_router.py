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
