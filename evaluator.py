"""
evaluator.py: RecommendationEvaluator for recommendation systems
"""
import math
from typing import List, Set, Dict

class RecommendationEvaluator:
    @staticmethod
    def precision_at_k(recommendations: List[str], relevant_items: Set[str], k: int) -> float:
        if not recommendations or not relevant_items or k == 0:
            return 0.0
        top_k = recommendations[:k]
        hits = sum(1 for item in top_k if item in relevant_items)
        return hits / k

    @staticmethod
    def recall_at_k(recommendations: List[str], relevant_items: Set[str], k: int) -> float:
        if not recommendations or not relevant_items or k == 0:
            return 0.0
        top_k = recommendations[:k]
        hits = sum(1 for item in top_k if item in relevant_items)
        return hits / len(relevant_items) if relevant_items else 0.0

    @staticmethod
    def ndcg_at_k(recommendations: List[str], relevant_items: Set[str], k: int) -> float:
        def dcg(recs):
            return sum((1 if rec in relevant_items else 0) / math.log2(idx + 2) for idx, rec in enumerate(recs[:k]))
        ideal = dcg(list(relevant_items)[:k])
        actual = dcg(recommendations)
        return actual / ideal if ideal > 0 else 0.0

    @staticmethod
    def evaluate_all(recommendations_dict: Dict[str, List[str]], ground_truth_dict: Dict[str, Set[str]], k: int) -> Dict[str, float]:
        precisions, recalls, ndcgs = [], [], []
        for user, recs in recommendations_dict.items():
            relevant = ground_truth_dict.get(user, set())
            precisions.append(RecommendationEvaluator.precision_at_k(recs, relevant, k))
            recalls.append(RecommendationEvaluator.recall_at_k(recs, relevant, k))
            ndcgs.append(RecommendationEvaluator.ndcg_at_k(recs, relevant, k))
        return {
            'precision_at_k': sum(precisions) / len(precisions) if precisions else 0.0,
            'recall_at_k': sum(recalls) / len(recalls) if recalls else 0.0,
            'ndcg_at_k': sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        }
