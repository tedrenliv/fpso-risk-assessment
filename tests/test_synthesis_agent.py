import pytest
from agents.skill_agents.base_skill_agent import SkillResult
from agents.synthesis_agent import synthesise

def make_result(domain_id, risk_score, bpa):
    return SkillResult(domain_id=domain_id, domain_name=domain_id,
                       risk_score=risk_score, belief_masses=bpa, explanation="test")

def test_synthesis_returns_combined_bpa():
    results = [
        make_result("SI",   0.7, {"Low": 0.1, "Medium": 0.2, "High": 0.7}),
        make_result("MM",   0.8, {"Low": 0.05,"Medium": 0.15,"High": 0.8}),
    ]
    output = synthesise(results)
    assert "combined_bpa" in output
    assert abs(sum(output["combined_bpa"].values()) - 1.0) < 0.05

def test_synthesis_returns_belief_intervals():
    results = [make_result("SI", 0.5, {"Low": 0.3, "Medium": 0.4, "High": 0.3})]
    output = synthesise(results)
    assert "belief_intervals" in output
    for level in ["Low", "Medium", "High"]:
        assert level in output["belief_intervals"]

def test_synthesis_identifies_dominant_factor():
    results = [
        make_result("SI",   0.3, {"Low": 0.7, "Medium": 0.2, "High": 0.1}),
        make_result("MM",   0.9, {"Low": 0.05,"Medium": 0.1, "High": 0.85}),
    ]
    output = synthesise(results)
    assert output["dominant_factor"] == "MM"

def test_synthesis_produces_risk_ranking():
    results = [
        make_result("SI",  0.6, {"Low": 0.1, "Medium": 0.3, "High": 0.6}),
        make_result("EH",  0.4, {"Low": 0.4, "Medium": 0.4, "High": 0.2}),
        make_result("HF",  0.7, {"Low": 0.05,"Medium": 0.2, "High": 0.75}),
    ]
    output = synthesise(results)
    assert "risk_ranking" in output
    assert len(output["risk_ranking"]) == 3
