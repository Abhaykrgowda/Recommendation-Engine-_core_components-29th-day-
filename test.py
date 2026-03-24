"""
test.py: Simple tests for recommendation engine components
"""
from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator

# --- SimilarityCalculator tests ---
print("Testing SimilarityCalculator...")
assert SimilarityCalculator.cosine_similarity([1, 0], [0, 1]) == 0.0
assert round(SimilarityCalculator.cosine_similarity([1, 1], [1, 1]), 2) == 1.0
assert SimilarityCalculator.jaccard_similarity({"a", "b"}, {"b", "c"}) == 0.3333333333333333
assert SimilarityCalculator.jaccard_similarity(set(), set()) == 1.0
import math
print("Pearson output:", SimilarityCalculator.pearson_correlation([1, 2, 3], [1, 2, 3]))
assert math.isclose(SimilarityCalculator.pearson_correlation([1, 2, 3], [1, 2, 3]), 1.0, abs_tol=1e-8)

# --- CandidateGenerator tests ---
print("Testing CandidateGenerator...")
user_histories = {"u1": {"i1", "i2"}, "u2": {"i2", "i3"}, "u3": {"i4"}}
item_similarities = {"i1": {"i3"}, "i2": {"i4"}, "i3": {"i1"}, "i4": {"i2"}}
item_popularity = {"i1": 10, "i2": 20, "i3": 5, "i4": 15}
gen = CandidateGenerator(user_histories, item_similarities, item_popularity)
assert "i3" in gen.collaborative_candidates("u1")
assert "i4" in gen.content_based_candidates("u1")
assert gen.popularity_candidates()[0] == "i2"
assert len(gen.hybrid_candidates("u1")) > 0

# --- RecommendationScorer tests ---
print("Testing RecommendationScorer...")
scorer = RecommendationScorer()
scorer.add_scorer("popularity", lambda u, i, ctx: ctx["popularity"].get(i, 0)/20, 0.5)
scorer.add_scorer("recency", lambda u, i, ctx: ctx["recency"].get(i, 0), 0.5)
context = {"popularity": item_popularity, "recency": {"i1": 0.8, "i2": 0.2, "i3": 0.5, "i4": 0.1}}
score, explanation = scorer.calculate_score("u1", "i1", context)
assert 0.0 <= score <= 1.0
ranked = scorer.rank_candidates("u1", ["i1", "i2", "i3", "i4"], context, limit=2)
assert len(ranked) == 2

# --- RecommendationEvaluator tests ---
print("Testing RecommendationEvaluator...")
recs = ["i2", "i1", "i3"]
relevant = {"i1", "i2"}
assert RecommendationEvaluator.precision_at_k(recs, relevant, 2) == 1.0
assert RecommendationEvaluator.recall_at_k(recs, relevant, 2) == 1.0
assert RecommendationEvaluator.ndcg_at_k(recs, relevant, 3) > 0.0
recommendations_dict = {"u1": ["i2", "i1", "i3"]}
ground_truth_dict = {"u1": {"i1", "i2"}}
metrics = RecommendationEvaluator.evaluate_all(recommendations_dict, ground_truth_dict, 2)
assert all(0.0 <= v <= 1.0 for v in metrics.values())
print("All tests passed!")
