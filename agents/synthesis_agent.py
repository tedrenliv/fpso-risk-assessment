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
