
import datetime
from sqlalchemy.orm import Session
from . import models
import random

class DateEngine:
    def __init__(self, inference_service):
        self.inference_service = inference_service

    def suggest_date(self, user_a: models.User, user_b: models.User, db: Session, location_override=None) -> dict:
        """
        Layer 4: The Closer.
        Suggests a specific venue (Food/Activity) based on combined taste.
        """
        # 1. Fuse Tastes
        vec_a = self.inference_service.mix_tastes(user_a.taste_vectors) if user_a.taste_vectors else None
        vec_b = self.inference_service.mix_tastes(user_b.taste_vectors) if user_b.taste_vectors else None
        
        if vec_a is None or vec_b is None:
            return None
            
        # Average Vector (Center of Friendship)
        center_vec = (vec_a + vec_b) / 2.0
        
        # 2. Retrieve Candidates (Food Domain)
        # Using MultiDomainInferenceService logic locally exposed or direct
        # Since we are inside App structure, we can access inference_service.collections
        
        candidates = []
        if self.inference_service.collections and "food" in self.inference_service.collections:
            collection = self.inference_service.collections["food"]
            results = collection.query(
                 query_embeddings=[center_vec.tolist()],
                 n_results=50
            )
            # Filter by location (Constraint)
            # Need to fetch objects to check metadata/geo
            ids = results['ids'][0] if results['ids'] else []
            candidates = ids
            
        else:
            # Fallback: Random from DB if no collection
            return {
                "title": "Date Night at Mario's", 
                "reason": "Classic choice.",
                "pitch": "Trust us, the pasta is worth it."
            }

        if not candidates:
             return None
             
        # 3. Apply Constraints (Geo, Budget) - Simplified for MVP
        # Fetch items from DB
        items = db.query(models.Item).filter(models.Item.ml_id.in_(candidates)).all()
        
        # TODO: Filter by distance if user preferences set
        
        if not items:
            return None
            
        chosen_item = random.choice(items) # Pick one from top relevant
        
        # 4. Generate Authority Pitch
        pitch_style = random.choice(["evidence", "mystery"])
        
        if pitch_style == "evidence":
            pitch = f"Based on your shared love for {chosen_item.genres}, this is the perfect spot."
        else:
            pitch = "Don't ask questions. Just go. Thank us later."
            
        return {
            "item_id": chosen_item.id,
            "title": chosen_item.title,
            "image": chosen_item.poster_path, # URL
            "pitch": pitch,
            "style": pitch_style
        }
