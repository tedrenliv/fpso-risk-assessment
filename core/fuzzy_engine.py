from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class TFN:
    """Triangular Fuzzy Number (l, m, u)."""
    l: float
    m: float
    u: float

    def membership(self, x: float) -> float:
        if x <= self.l or x >= self.u:
            return 0.0
        if x <= self.m:
            return (x - self.l) / (self.m - self.l) if self.m != self.l else 1.0
        return (self.u - x) / (self.u - self.m) if self.u != self.m else 1.0


def triangular_membership(x: float, l: float, m: float, u: float) -> float:
    """Convenience function for triangular membership."""
    return TFN(l, m, u).membership(x)


# Standard linguistic term TFNs
STANDARD_TERMS: Dict[str, TFN] = {
    "very_low":  TFN(0.00, 0.00, 0.25),
    "low":       TFN(0.00, 0.25, 0.50),
    "medium":    TFN(0.25, 0.50, 0.75),
    "high":      TFN(0.50, 0.75, 1.00),
    "very_high": TFN(0.75, 1.00, 1.00),
}


def fuzzify(crisp_value: float, terms: Dict[str, TFN]) -> Dict[str, float]:
    """Convert a crisp value to membership degrees for each linguistic term."""
    return {term: tfn.membership(crisp_value) for term, tfn in terms.items()}


def mamdani_inference(
    rules: List[Dict],
    input_memberships: Dict[str, Dict[str, float]],
    output_terms: Dict[str, TFN],
) -> float:
    """
    Mamdani fuzzy inference.
    rules: list of {"conditions": {var: term}, "output": term, "weight": float}
    input_memberships: {var: {term: degree}}
    Returns defuzzified crisp risk score.
    """
    output_activations: Dict[str, float] = {t: 0.0 for t in output_terms}

    for rule in rules:
        # AND aggregation (min) across conditions
        firing_strength = min(
            input_memberships.get(var, {}).get(term, 0.0)
            for var, term in rule["conditions"].items()
        )
        firing_strength *= rule.get("weight", 1.0)
        out_term = rule["output"]
        if out_term in output_activations:
            output_activations[out_term] = max(
                output_activations[out_term], firing_strength
            )

    return defuzzify(output_activations, output_terms)


def defuzzify(
    activations: Dict[str, float], output_terms: Dict[str, TFN]
) -> float:
    """Centroid defuzzification over discretised universe [0, 1]."""
    x_points = np.linspace(0, 1, 200)
    aggregated = np.zeros_like(x_points)

    for term, activation in activations.items():
        if term in output_terms and activation > 0:
            tfn = output_terms[term]
            memberships = np.array([tfn.membership(x) for x in x_points])
            aggregated = np.maximum(aggregated, activation * memberships)

    total = np.sum(aggregated)
    if total == 0:
        return 0.5  # default to medium uncertainty
    return float(np.sum(x_points * aggregated) / total)
