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
