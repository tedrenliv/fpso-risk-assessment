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


_CHIP_TO_DOMAIN = {"SI": "SI", "MM": "MM", "EH": "EH", "HF": "HF", "SYSI": "SysI"}


async def run_pipeline_async(
    query: str,
    expert_ground_truth: Dict[str, Dict[str, float]] = None,
    run_feedback: bool = False,
    forced_skills: list = None,
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

    # L3: Skill Router (or forced override from UI)
    print("[L3] Running Skill Router...")
    if forced_skills:
        active_skills = [_CHIP_TO_DOMAIN.get(s.upper(), s) for s in forced_skills]
        routing = {"primary_skill": active_skills[0], "secondary_skills": active_skills[1:],
                   "confidence": 1.0, "forced": True}
        print(f"  → Forced Skills: {active_skills} (user-selected)")
    else:
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
    forced_skills: list = None,
) -> Dict[str, Any]:
    """Synchronous wrapper for the async pipeline."""
    return asyncio.run(run_pipeline_async(query, expert_ground_truth, run_feedback, forced_skills))
