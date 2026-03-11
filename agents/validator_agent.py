import json
import re
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

    from core import stream_queue as _sq
    if _sq._orig_stdout is not None:
        # Web mode: signal the SSE stream and wait for browser HITL
        import json as _json
        from core import hitl_bridge as _hb
        _hitl_payload = {
            "summary": validation_report.get("validation_summary", ""),
            "flags": validation_report.get("flags", []),
            "recommendation": validation_report.get("recommendation", "review"),
            "ranking": synthesis_output.get("risk_ranking", []),
        }
        _sq.get_queue().put(f"__HITL__:{_json.dumps(_hitl_payload)}")
        expert_decision, expert_note = _hb.wait_for_decision()
    else:
        # Terminal mode: original input() flow
        print("Options: [A]pprove  [R]eject  [M]odify weights  [S]kip")
        expert_input = input("Expert decision: ").strip().upper()
        expert_decision = {
            "A": "approved",
            "R": "rejected",
            "M": "modify_requested",
            "S": "skipped",
        }.get(expert_input, "approved")
        expert_note = ""
        if expert_decision in ("rejected", "modify_requested"):
            expert_note = input("Please provide reason/instructions: ").strip()
    # ──────────────────────────────

    synthesis_output["validation"] = {
        "report": validation_report,
        "expert_decision": expert_decision,
        "expert_note": expert_note,
    }

    return synthesis_output
