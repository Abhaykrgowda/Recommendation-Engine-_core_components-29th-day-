# Day 29: Recommendation Engine Components

## Overview
This project implements the core algorithmic components of a recommendation engine. Each module is designed to be modular and testable, forming the foundation for a complete recommendation system.

## Project Structure
```
day29_project/
├── similarity.py      # SimilarityCalculator: cosine, jaccard, pearson
├── candidate_gen.py   # CandidateGenerator: collaborative, content, popularity, hybrid
├── scorer.py          # RecommendationScorer: scoring and ranking
├── evaluator.py       # RecommendationEvaluator: precision, recall, ndcg
├── test.py            # Simple tests for all modules
```

## Components
### 1. SimilarityCalculator (`similarity.py`)
- `cosine_similarity(vec1, vec2)`: Compares user/item vectors
- `jaccard_similarity(set1, set2)`: Compares sets (skills/tags)
- `pearson_correlation(ratings1, ratings2)`: Compares rating patterns
- Handles edge cases (empty sets, zero vectors)

### 2. CandidateGenerator (`candidate_gen.py`)
- `collaborative_candidates(user_id)`: Items liked by similar users
- `content_based_candidates(user_id)`: Items similar to user's history
- `popularity_candidates()`: Most popular items overall
- `hybrid_candidates(user_id)`: Combines multiple strategies
- Handles cold start users and limits results

### 3. RecommendationScorer (`scorer.py`)
- `add_scorer(name, function, weight)`: Register scoring functions
- `calculate_score(user_id, item_id, context)`: Score a single item
- `rank_candidates(user_id, candidates, limit)`: Return top N items
- Supports multiple weighted scoring factors and explanations

### 4. RecommendationEvaluator (`evaluator.py`)
- `precision_at_k`, `recall_at_k`, `ndcg_at_k`: Standard metrics
- `evaluate_all`: Averages metrics across users
- Handles missing ground truth data

## Testing
Run all tests with:
```sh
cd day29_project
& "..\.venv\Scripts\python.exe" test.py
```
All modules include at least one test case. You should see `All tests passed!` if everything is working.

## Notes
- Uses only Python standard library
- Data is stored in dictionaries for simplicity
- Focus is on correctness and modularity

## Next Steps
These components are ready to be integrated into a full recommendation system.
