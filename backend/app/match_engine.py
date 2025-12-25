
import math
import numpy as np
from sqlalchemy.orm import Session
from . import models

class MatchEngine:
    def __init__(self, inference_service):
        self.inference_service = inference_service

    def calculate_match(self, user_a: models.User, user_b: models.User) -> dict:
        """
        Layer 2: Match Intelligence.
        Returns: {
            "score": float (0-100),
            "type": str (Mirror, Complement, Contrast),
            "explanation_facts": dict
        }
        """
        # 1. Vector Similarity (Composite)
        vec_a = self._get_vector(user_a)
        vec_b = self._get_vector(user_b)
        
        if vec_a is None or vec_b is None:
            return {"score": 0, "type": "Unknown", "explanation_facts": {}}

        cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        # Map -1..1 to 0..100
        score = (cosine_sim + 1) * 50
        
        # 2. Determine Match Type
        # Heuristics:
        # High Sim (> 85) -> Mirror
        # Mid Sim (60-85) + Different Top Genres -> Complement
        # Low Sim (< 60) -> Contrast (if potential exists)
        
        match_type = "Contrast"
        explanation = {}
        
        if score > 85:
            match_type = "Mirror"
            explanation["reason"] = "You two are practically the same person!"
        elif score > 60:
            match_type = "Complement"
            explanation["reason"] = "Different tastes, but compatible vibes."
        else:
            match_type = "Contrast"
            explanation["reason"] = "Opposites attract!"
            
        # 3. Deep Dive (Facts)
        # TODO: Extract shared genres, top artists overlap locally
        
        return {
            "score": round(score, 1),
            "type": match_type,
            "explanation_facts": explanation
        }

    def _get_vector(self, user):
        if user.taste_vectors:
             return self.inference_service.mix_tastes(user.taste_vectors)
        # Fallback if only legacy
        # In real app, we'd fetch from inference service cache
        # For MVP, assume vectors are in user object or return None
        return None
