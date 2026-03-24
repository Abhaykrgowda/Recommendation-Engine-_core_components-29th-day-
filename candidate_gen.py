"""
candidate_gen.py: CandidateGenerator for recommendation systems
"""
from typing import List, Dict, Set
from collections import Counter

class CandidateGenerator:
    def __init__(self, user_histories: Dict, item_similarities: Dict, item_popularity: Dict):
        self.user_histories = user_histories  # user_id -> set of item_ids
        self.item_similarities = item_similarities  # item_id -> set of similar item_ids
        self.item_popularity = item_popularity  # item_id -> count

    def collaborative_candidates(self, user_id: str, limit: int = 20) -> List[str]:
        user_items = self.user_histories.get(user_id, set())
        if not user_items:
            return self.popularity_candidates(limit)
        similar_users = [uid for uid, items in self.user_histories.items() if uid != user_id and items & user_items]
        candidate_items = Counter()
        for uid in similar_users:
            candidate_items.update(self.user_histories[uid])
        # Remove items already seen by user
        for item in user_items:
            candidate_items.pop(item, None)
        return [item for item, _ in candidate_items.most_common(limit)]

    def content_based_candidates(self, user_id: str, limit: int = 20) -> List[str]:
        user_items = self.user_histories.get(user_id, set())
        if not user_items:
            return self.popularity_candidates(limit)
        candidates = set()
        for item in user_items:
            candidates.update(self.item_similarities.get(item, set()))
        candidates -= user_items
        return list(candidates)[:limit]

    def popularity_candidates(self, limit: int = 20) -> List[str]:
        return [item for item, _ in Counter(self.item_popularity).most_common(limit)]

    def hybrid_candidates(self, user_id: str, limit: int = 20) -> List[str]:
        c1 = set(self.collaborative_candidates(user_id, limit*2))
        c2 = set(self.content_based_candidates(user_id, limit*2))
        c3 = set(self.popularity_candidates(limit*2))
        combined = list(c1 | c2 | c3)
        return combined[:limit]
