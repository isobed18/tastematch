import os
import sys
import torch
import numpy as np
from sqlalchemy.orm import sessionmaker

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from app.database import engine
from app import models
from app.inference_service import InferenceService

# Setup DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def check_item_sim():
    print("Loading Items...")
    # Get an Action movie
    action_movie = db.query(models.Item).filter(models.Item.genres.like("%Action%")).first()
    # Get a Romance movie
    romance_movie = db.query(models.Item).filter(models.Item.genres.like("%Romance%")).first()
    
    if not action_movie or not romance_movie:
        print("Could not find sample movies.")
        return

    print(f"Movie A: {action_movie.title} ({action_movie.genres})")
    print(f"Movie B: {romance_movie.title} ({romance_movie.genres})")

    svc = InferenceService()
    
    # We need access to item embeddings.
    # InferenceService -> retrieval_engine -> item_embeddings (Tensor)
    # We need indices.
    
    svc.retrieval_engine.index_items()
    item_map = svc.retrieval_engine.item_map
    embeddings = svc.retrieval_engine.item_embeddings
    
    idx_a = item_map.get(action_movie.ml_id)
    idx_b = item_map.get(romance_movie.ml_id)
    
    if idx_a is None or idx_b is None:
        print("One of the items is not in the model map.")
        return
        
    vec_a = embeddings[idx_a].cpu().numpy()
    vec_b = embeddings[idx_b].cpu().numpy()
    
    sim = cosine_sim(vec_a, vec_b)
    print(f"Item Similarity: {sim:.4f}")
    
    # Check Average Similarity of random items
    print("Checking Histogram of Random Pairs...")
    scores = []
    num_items = len(embeddings)
    indices = list(range(num_items))
    import random
    for _ in range(100):
        i1, i2 = random.sample(indices, 2)
        v1 = embeddings[i1].cpu().numpy()
        v2 = embeddings[i2].cpu().numpy()
        scores.append(cosine_sim(v1, v2))
        
    avg_score = np.mean(scores)
    print(f"Average Random Similarity: {avg_score:.4f}")
    print(f"Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")

if __name__ == "__main__":
    check_item_sim()
