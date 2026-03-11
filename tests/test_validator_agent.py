from unittest.mock import patch, MagicMock
from core import hitl_bridge, stream_queue


def _make_synthesis():
    return {
        "combined_bpa": {"Low": 0.1, "Medium": 0.2, "High": 0.7},
        "dominant_factor_name": "EH",
        "dominant_belief_high": 0.7,
        "risk_ranking": [
            {"rank": 1, "domain_name": "Environmental Hazard",
             "risk_score": 0.83, "high_belief": 0.70}
        ],
        "belief_intervals": {},
    }


def _make_mock_response():
    mock = MagicMock()
    mock.content = [MagicMock(text=(
        '{"validation_summary": "Risk is high", "flags": [], '
        '"recommendation": "approve", "confidence_note": ""}'
    ))]
    return mock


def test_validator_uses_hitl_bridge_in_web_mode():
    """When stream_queue is installed (web mode), validator uses hitl_bridge."""
    stream_queue.reset()
    stream_queue.install()
    hitl_bridge.reset()
    hitl_bridge.set_decision("approved", "")  # pre-set so it doesn't block

    with patch("agents.validator_agent.client.messages.create",
               return_value=_make_mock_response()):
        from agents import validator_agent
        result = validator_agent.run_validator(_make_synthesis(), "test query")

    stream_queue.uninstall()
    assert result["validation"]["expert_decision"] == "approved"


def test_validator_falls_back_to_input_in_terminal_mode(monkeypatch):
    """When stream_queue is NOT installed (terminal mode), validator uses input()."""
    stream_queue.uninstall()  # ensure terminal mode

    monkeypatch.setattr("builtins.input", lambda _: "A")

    with patch("agents.validator_agent.client.messages.create",
               return_value=_make_mock_response()):
        from agents import validator_agent
        result = validator_agent.run_validator(_make_synthesis(), "test query")

    assert result["validation"]["expert_decision"] == "approved"
