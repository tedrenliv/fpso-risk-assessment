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
