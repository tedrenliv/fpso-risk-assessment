import pytest
from agents.skill_agents.base_skill_agent import run_skill_agent, SkillResult

def test_skill_agent_returns_skill_result():
    evidence = {
        "linguistic_assignments": {"corrosion_rate": "high", "fatigue_level": "medium"},
        "belief_masses": {"Low": 0.1, "Medium": 0.3, "High": 0.6},
        "risk_factors": [{"name": "blade corrosion", "severity": 0.8}],
    }
    result = run_skill_agent("SI", evidence)
    assert isinstance(result, SkillResult)
    assert 0.0 <= result.risk_score <= 1.0
    assert abs(sum(result.belief_masses.values()) - 1.0) < 0.05
    assert result.domain_id == "SI"
    assert result.explanation

def test_all_five_skill_agents_run():
    evidence = {
        "linguistic_assignments": {},
        "belief_masses": {"Low": 0.33, "Medium": 0.34, "High": 0.33},
        "risk_factors": [],
    }
    for domain_id in ["SI", "MM", "EH", "HF", "SysI"]:
        result = run_skill_agent(domain_id, evidence)
        assert isinstance(result, SkillResult)
        assert result.domain_id == domain_id

def test_high_risk_evidence_produces_high_score():
    evidence = {
        "linguistic_assignments": {"corrosion_rate": "high", "fatigue_level": "high"},
        "belief_masses": {"Low": 0.05, "Medium": 0.15, "High": 0.80},
        "risk_factors": [{"name": "severe corrosion", "severity": 0.95}],
    }
    result = run_skill_agent("SI", evidence)
    assert result.risk_score > 0.55
