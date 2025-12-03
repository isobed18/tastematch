from sqlalchemy.orm import Session
from . import models
from .vector_db import vector_db
from typing import List, Dict, Tuple
import numpy as np
import json

class Recommender:
    def __init__(self):
        self.initialized = False

    def initialize(self, db: Session):
        print("Initializing Recommender System (ChromaDB)...")
        vector_db.initialize()
        self.initialized = True
        print("Recommender Initialized.")

    def update_user_profile(self, user_id: int, item_id: int, action: str, db: Session):
        """
        Update user's taste vector based on a new swipe.
        """
        if not self.initialized:
            self.initialize(db)

        user = db.query(models.User).filter(models.User.id == user_id).first()
        item = db.query(models.Item).filter(models.Item.id == item_id).first()
        
        if not user or not item:
            return

        # Determine which vector to update
        is_movie = item.type == models.ItemType.movie or item.type == "movie"
        current_taste = user.movie_taste if is_movie else user.game_taste
        
        # Ensure current_taste is a list (it might be None or empty)
        if not current_taste:
            current_taste = []
        
        # Get item vector from ChromaDB (or generate it)
        # Ideally we fetch from Chroma, but for speed we can regenerate since we have the text
        # Construct text same as migration script
        meta = item.metadata_content or {}
        genres = meta.get('genres', '')
        if isinstance(genres, list):
            genres = ", ".join(genres)
        description = meta.get('description') or meta.get('overview') or ""
        text_to_embed = f"Title: {item.title}. Genres: {genres}. Description: {description}"
        
        item_vector = vector_db.generate_vector(text_to_embed)
        
        # Update logic: Weighted moving average
        # Action weights
        weight = 1.0
        if action == models.SwipeAction.superlike:
            weight = 2.0
        elif action == models.SwipeAction.dislike:
            weight = -0.5 # Push vector away slightly? Or just ignore?
            # For now, let's only positively reinforce likes/superlikes to build a "taste" profile
            # Dislikes could be handled by subtracting, but that might drift to zero.
            # Let's ignore dislikes for the *positive* taste vector for now, 
            # or maybe implement a separate "dislike" vector later.
            return 

        # Convert to numpy for math
        user_vec = np.array(current_taste) if len(current_taste) > 0 else np.zeros_like(item_vector)
        item_vec = np.array(item_vector)
        
        # Simple average for now: (old * N + new) / (N + 1)
        # But since we don't track N easily here without extra columns, let's use a learning rate
        # New = Old + alpha * (Target - Old)
        alpha = 0.1 * weight # Learning rate
        
        if len(current_taste) == 0:
            new_vec = item_vec
        else:
            new_vec = user_vec + alpha * (item_vec - user_vec)
            
        # Normalize
        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec = new_vec / norm
            
        # Save back to DB
        if is_movie:
            user.movie_taste = new_vec.tolist()
        else:
            user.game_taste = new_vec.tolist()
            
        # Force update flag for SQLAlchemy if it doesn't detect JSON change automatically
        # (It usually does if we reassign)
        db.add(user)
        db.commit()
        print(f"Updated {'Movie' if is_movie else 'Game'} taste for User {user.username}")

    def recommend(self, user_id: int, db: Session, item_type: str, limit: int = 10) -> List[models.Item]:
        if not self.initialized:
            self.initialize(db)

        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            return []

        # Get User Vector
        is_movie = item_type == 'movie'
        user_vector = user.movie_taste if is_movie else user.game_taste
        
        if not user_vector or len(user_vector) == 0:
            return [] # No taste yet

        # Query ChromaDB
        # We need to filter by type
        results = vector_db.search_items(
            query_vector=user_vector,
            limit=limit * 2, # Fetch more to filter out swiped
            where={"type": item_type}
        )
        
        if not results:
            return []
            
        # Filter out swiped items
        swiped_ids = [s.item_id for s in db.query(models.Swipe.item_id).filter(models.Swipe.user_id == user_id).all()]
        
        recommendations = []
        for res in results:
            item_id = int(res['id'])
            if item_id not in swiped_ids:
                recommendations.append(item_id)
                if len(recommendations) >= limit:
                    break
        
        if not recommendations:
            return []

        # Fetch Item objects
        # Preserve order from ChromaDB results
        items = db.query(models.Item).filter(models.Item.id.in_(recommendations)).all()
        items_map = {i.id: i for i in items}
        ordered_items = [items_map[i] for i in recommendations if i in items_map]
        
        return ordered_items

# Global instance
recommender = Recommender()
