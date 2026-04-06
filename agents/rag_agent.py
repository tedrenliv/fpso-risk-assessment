import json
import anthropic
from core.vector_store import VectorStore
from config import ANTHROPIC_API_KEY, MODEL, MAX_TOKENS

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# Complete schema of all skill input variables — must match skills/*.json linguistic_variables keys.
# Each entry: "variable_name": "plain-English description for Claude"
SKILL_VARIABLE_SCHEMA = {
    # Structural Integrity (SI)
    "corrosion_rate":        "rate/severity of material corrosion (low/medium/high)",
    "fatigue_level":         "structural fatigue or cyclic-loading severity (low/medium/high)",
    "inspection_frequency":  "how often inspections are carried out (low=rare, high=frequent)",
    # Maintenance Management (MM)
    "maintenance_frequency": "how frequently preventive maintenance is performed (low=rare, high=frequent)",
    "equipment_condition":   "overall physical condition of equipment (low=poor, high=good)",
    "overdue_tasks":         "volume/severity of overdue maintenance tasks (low=few, high=many)",
    # Environmental Hazard (EH)
    "wave_height":           "significant wave height / sea state severity (low/medium/high)",
    "wind_speed":            "wind speed severity (low/medium/high)",
    "current_speed":         "ocean current speed severity (low/medium/high)",
    # Human Factors (HF)
    "training_adequacy":     "adequacy of operator/crew training (low=inadequate, high=excellent)",
    "procedural_compliance": "degree of compliance with operational procedures (low=poor, high=strict)",
    "operator_fatigue":      "operator or crew fatigue level (low=rested, high=exhausted)",
    # System Integration (SysI)
    "redundancy_level":      "level of system redundancy (low=single-point failure risk, high=fully redundant)",
    "sensor_reliability":    "reliability of sensors and position-reference systems (low/medium/high)",
    "integration_complexity":"complexity of control-system integration (low=simple, high=very complex)",
}

_VARIABLE_LIST = "\n".join(
    f'  "{k}": {v}' for k, v in SKILL_VARIABLE_SCHEMA.items()
)

SYSTEM_PROMPT = f"""You are a Skill-Aware RAG Agent for offshore risk analysis.
Given a user query and retrieved documents, extract structured evidence for risk assessment.
You must return a JSON object with:
- "risk_factors": list of identified risk factors with severity (0-1 scale)
- "linguistic_assignments": dict mapping variable names to linguistic terms (ONLY low/medium/high)
- "belief_masses": dict mapping risk levels (Low/Medium/High) to probability mass (sum to 1.0)
- "evidence_summary": 2-3 sentence summary of key findings
- "source_documents": list of relevant source snippets used

CRITICAL — for "linguistic_assignments" you MUST use ONLY these exact variable names:
{_VARIABLE_LIST}

Assign low/medium/high to every variable that is relevant to the query.
Omit variables with no evidence either way. Do not invent new variable names.

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

Retrieved Knowledge Base Context:
{context}

Instructions:
Use the QUERY TEXT as your primary evidence source for linguistic_assignments.
Parse the query for explicit risk indicators and map them directly:
  - Stated high conditions (e.g. "storm", "corrosion", "overdue", "failure", "exceedance") → assign "high"
  - Stated low/reduced conditions (e.g. "reduced crew", "low maintenance", "absent") → assign "low"
  - Ambiguous or intermediate conditions → assign "medium"
  - Variables with no query evidence → omit from linguistic_assignments

Use the retrieved context as additional supporting evidence where it adds specificity.
Assign belief_masses to reflect the overall risk level implied by the query."""

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
