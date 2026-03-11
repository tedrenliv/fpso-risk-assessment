import pytest
from core.dempster_shafer import (
    combine_bpa, belief, plausibility, BeliefInterval, normalise_bpa
)

def test_normalise_bpa_sums_to_one():
    raw = {"Low": 0.3, "Medium": 0.4, "High": 0.2}
    result = normalise_bpa(raw)
    assert sum(result.values()) == pytest.approx(1.0)

def test_combine_two_consistent_bpas():
    m1 = {"Low": 0.8, "High": 0.2}
    m2 = {"Low": 0.7, "High": 0.3}
    combined = combine_bpa(m1, m2)
    assert combined["Low"] > combined["High"]
    assert sum(combined.values()) == pytest.approx(1.0)

def test_belief_interval_low_risk():
    m = {"Low": 0.7, "Medium": 0.2, "High": 0.1}
    frame = ["Low", "Medium", "High"]
    bel = belief(m, "Low", frame)
    pla = plausibility(m, "Low", frame)
    assert bel <= pla
    assert 0.0 <= bel <= 1.0
    assert 0.0 <= pla <= 1.0

def test_belief_interval_dataclass():
    m = {"Low": 0.6, "Medium": 0.3, "High": 0.1}
    frame = ["Low", "Medium", "High"]
    interval = BeliefInterval.from_bpa(m, "Low", frame)
    assert interval.belief <= interval.plausibility

def test_high_conflict_combination():
    m1 = {"Low": 1.0}
    m2 = {"High": 1.0}
    # Should not raise, should return a valid distribution
    combined = combine_bpa(m1, m2)
    assert sum(combined.values()) == pytest.approx(1.0)
