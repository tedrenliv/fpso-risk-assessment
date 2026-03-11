import pytest
import numpy as np
from core.fuzzy_engine import TFN, triangular_membership, fuzzify, mamdani_inference, defuzzify

def test_tfn_membership_at_peak():
    tfn = TFN(0.0, 0.5, 1.0)
    assert tfn.membership(0.5) == pytest.approx(1.0)

def test_tfn_membership_at_edges():
    tfn = TFN(0.0, 0.5, 1.0)
    assert tfn.membership(0.0) == pytest.approx(0.0)
    assert tfn.membership(1.0) == pytest.approx(0.0)

def test_tfn_membership_outside():
    tfn = TFN(0.2, 0.5, 0.8)
    assert tfn.membership(0.1) == pytest.approx(0.0)
    assert tfn.membership(0.9) == pytest.approx(0.0)

def test_fuzzify_returns_membership_degrees():
    terms = {
        "low":    TFN(0.0, 0.0, 0.5),
        "medium": TFN(0.0, 0.5, 1.0),
        "high":   TFN(0.5, 1.0, 1.0),
    }
    result = fuzzify(0.5, terms)
    assert "low" in result
    assert "medium" in result
    assert result["medium"] == pytest.approx(1.0)

def test_defuzzify_symmetric():
    # Symmetric activation should return middle value
    output_terms = {
        "low":  TFN(0.0, 0.0, 0.5),
        "high": TFN(0.5, 1.0, 1.0),
    }
    activations = {"low": 0.5, "high": 0.5}
    result = defuzzify(activations, output_terms)
    assert 0.3 < result < 0.7  # should be roughly centred

def test_mamdani_inference_high_risk():
    rules = [
        {"conditions": {"corrosion": "high"}, "output": "high", "weight": 1.0},
    ]
    input_memberships = {"corrosion": {"low": 0.0, "medium": 0.1, "high": 0.9}}
    output_terms = {
        "low":  TFN(0.0, 0.0, 0.5),
        "medium": TFN(0.0, 0.5, 1.0),
        "high": TFN(0.5, 1.0, 1.0),
    }
    score = mamdani_inference(rules, input_memberships, output_terms)
    assert score > 0.6
