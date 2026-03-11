from dataclasses import dataclass
from typing import Dict, List
from itertools import product


def normalise_bpa(bpa: Dict[str, float]) -> Dict[str, float]:
    """Normalise a BPA so values sum to 1.0."""
    total = sum(bpa.values())
    if total == 0:
        n = len(bpa)
        return {k: 1.0 / n for k in bpa}
    return {k: v / total for k, v in bpa.items()}


def combine_bpa(m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
    """
    Dempster's rule of combination for two BPAs over atomic hypotheses.
    Handles full conflict by uniform distribution fallback.
    """
    combined: Dict[str, float] = {}
    conflict = 0.0

    for (h1, v1), (h2, v2) in product(m1.items(), m2.items()):
        if h1 == h2:
            combined[h1] = combined.get(h1, 0.0) + v1 * v2
        else:
            conflict += v1 * v2

    normaliser = 1.0 - conflict
    if normaliser < 1e-10:
        # Complete conflict — return uniform distribution over all seen hypotheses
        all_hyps = list(set(m1) | set(m2))
        return {h: 1.0 / len(all_hyps) for h in all_hyps}

    return {h: v / normaliser for h, v in combined.items()}


def combine_multiple_bpas(bpas: List[Dict[str, float]]) -> Dict[str, float]:
    """Sequentially combine a list of BPAs."""
    if not bpas:
        return {}
    result = normalise_bpa(bpas[0])
    for bpa in bpas[1:]:
        result = combine_bpa(result, normalise_bpa(bpa))
    return result


def belief(bpa: Dict[str, float], hypothesis: str, frame: List[str]) -> float:
    """Bel(A) = sum of m(B) for all B strictly subset of A (atomic case: B == hypothesis)."""
    return bpa.get(hypothesis, 0.0)


def plausibility(bpa: Dict[str, float], hypothesis: str, frame: List[str]) -> float:
    """Pl(A) = 1 - Bel(complement of A)."""
    complement_belief = sum(
        v for h, v in bpa.items() if h != hypothesis
    )
    return 1.0 - complement_belief


@dataclass
class BeliefInterval:
    hypothesis: str
    belief: float
    plausibility: float

    @classmethod
    def from_bpa(
        cls, bpa: Dict[str, float], hypothesis: str, frame: List[str]
    ) -> "BeliefInterval":
        bel = belief(bpa, hypothesis, frame)
        pla = plausibility(bpa, hypothesis, frame)
        return cls(hypothesis=hypothesis, belief=bel, plausibility=pla)

    def __str__(self) -> str:
        return f"{self.hypothesis}: [{self.belief:.3f}, {self.plausibility:.3f}]"
