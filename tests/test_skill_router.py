import pytest
from agents.skill_router import route_skills

def test_routes_cpp_to_structural_integrity():
    query = "CPP blade fatigue and corrosion failure"
    evidence = {"risk_factors": [{"name": "blade corrosion", "severity": 0.8}],
                "linguistic_assignments": {"corrosion_rate": "high"}}
    result = route_skills(query, evidence)
    assert result["primary_skill"] == "SI"
    assert result["confidence"] > 0.7

def test_routes_dp_to_system_integration():
    query = "Dynamic positioning system software fault"
    evidence = {"risk_factors": [{"name": "DP failure", "severity": 0.9}],
                "linguistic_assignments": {}}
    result = route_skills(query, evidence)
    assert result["primary_skill"] == "SysI"

def test_returns_secondary_skills():
    query = "Operator error during DP manual override"
    evidence = {"risk_factors": [], "linguistic_assignments": {}}
    result = route_skills(query, evidence)
    assert "secondary_skills" in result
    assert isinstance(result["secondary_skills"], list)

def test_routing_result_has_required_keys():
    result = route_skills("maintenance overdue on CPP", {})
    assert all(k in result for k in ["primary_skill", "secondary_skills", "confidence", "routing_rationale"])
