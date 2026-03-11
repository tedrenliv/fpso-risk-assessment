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
