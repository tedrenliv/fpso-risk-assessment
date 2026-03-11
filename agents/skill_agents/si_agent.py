from agents.skill_agents.base_skill_agent import run_skill_agent, SkillResult

def run(evidence: dict) -> SkillResult:
    return run_skill_agent("SI", evidence)
