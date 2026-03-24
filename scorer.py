"""
scorer.py: RecommendationScorer for recommendation systems
"""
from typing import Callable, Dict, List, Tuple

class RecommendationScorer:
    def __init__(self):
        self.scorers: Dict[str, Tuple[Callable, float]] = {}

    def add_scorer(self, name: str, function: Callable, weight: float):
        self.scorers[name] = (function, weight)

    def calculate_score(self, user_id, item_id, context: Dict) -> Tuple[float, str]:
        total, total_weight = 0.0, 0.0
        explanations = []
        for name, (func, weight) in self.scorers.items():
            score = func(user_id, item_id, context)
            explanations.append(f"{name}: {score:.2f} (w={weight})")
            total += score * weight
            total_weight += weight
        final_score = total / total_weight if total_weight else 0.0
        explanation = "; ".join(explanations)
        return min(max(final_score, 0.0), 1.0), explanation

    def rank_candidates(self, user_id, candidates: List[str], context: Dict, limit: int = 10) -> List[Tuple[str, float, str]]:
        scored = [(item, *self.calculate_score(user_id, item, context)) for item in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
