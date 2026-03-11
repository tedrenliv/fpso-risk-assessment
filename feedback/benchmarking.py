from typing import Dict, Any


def benchmark_skill_performance(
    skill_result, expert_ground_truth: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compare Skill Agent output against expert ground truth.
    expert_ground_truth: {"Low": float, "Medium": float, "High": float}
    Returns agreement_rate, mean_deviation, per_template_metrics.
    """
    predicted = skill_result.belief_masses
    frame = ["Low", "Medium", "High"]

    deviations = [
        abs(predicted.get(level, 0.0) - expert_ground_truth.get(level, 0.0))
        for level in frame
    ]
    mean_deviation = sum(deviations) / len(deviations)
    agreement_rate = 1.0 - mean_deviation

    dominant_predicted = max(predicted, key=predicted.get)
    dominant_truth = max(expert_ground_truth, key=expert_ground_truth.get)
    dominant_match = dominant_predicted == dominant_truth

    return {
        "domain_id": skill_result.domain_id,
        "agreement_rate": round(agreement_rate, 4),
        "mean_deviation": round(mean_deviation, 4),
        "dominant_match": dominant_match,
        "per_level_deviation": {
            level: round(dev, 4) for level, dev in zip(frame, deviations)
        },
    }
