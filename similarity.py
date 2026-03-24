"""
similarity.py: SimilarityCalculator for recommendation systems
"""
import math
from typing import List, Set

class SimilarityCalculator:
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def jaccard_similarity(set1: Set, set2: Set) -> float:
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0.0

    @staticmethod
    def pearson_correlation(ratings1: List[float], ratings2: List[float]) -> float:
        if not ratings1 or not ratings2 or len(ratings1) != len(ratings2):
            return 0.0
        n = len(ratings1)
        mean1 = sum(ratings1) / n
        mean2 = sum(ratings2) / n
        num = sum((a - mean1) * (b - mean2) for a, b in zip(ratings1, ratings2))
        den1 = math.sqrt(sum((a - mean1) ** 2 for a in ratings1))
        den2 = math.sqrt(sum((b - mean2) ** 2 for b in ratings2))
        if den1 == 0 or den2 == 0:
            return 0.0
        return num / (den1 * den2)
