import pytest
from unittest.mock import patch, MagicMock
from agents.orchestrator import run_pipeline


@pytest.mark.integration
def test_full_pipeline_returns_output_structure():
    """End-to-end pipeline test with mocked API calls."""
    mock_rag_evidence = {
        "risk_factors": [{"name": "blade fatigue", "severity": 0.8}],
        "linguistic_assignments": {"corrosion_rate": "high", "fatigue_level": "high"},
        "belief_masses": {"Low": 0.1, "Medium": 0.2, "High": 0.7},
        "evidence_summary": "High corrosion detected on CPP blades.",
        "source_documents": ["CPP inspection record 2024"]
    }
    mock_routing = {
        "primary_skill": "SI",
        "secondary_skills": ["MM"],
        "confidence": 0.92,
        "routing_rationale": "Structural keywords detected."
    }

    with patch("agents.orchestrator.run_rag_agent", return_value=mock_rag_evidence), \
         patch("agents.orchestrator.route_skills", return_value=mock_routing), \
         patch("agents.orchestrator.run_validator", side_effect=lambda s, q: {**s, "validation": {"expert_decision": "approved", "expert_note": ""}}):
        result = run_pipeline("CPP blade corrosion risk assessment")

    assert "risk_rankings" in result
    assert "risk_profile" in result
    assert "traceability" in result
    assert "skill_trace" in result
    assert result["skill_trace"]["expert_validation"] == "approved"
