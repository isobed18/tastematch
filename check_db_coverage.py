
import os
import sys
import torch
# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from app.inference_service import InferenceService
from app import models, database
from sqlalchemy.orm import Session

def check_coverage():
    print("Initializing DB...")
    db = next(database.get_db())
    
    print("Initializing Inference Service...")
    service = InferenceService() 
    
    user_id = 2
    print(f"Checking candidates for User {user_id}...")
    
    # Mock history (or fetch real if needed, but lets assume empty for cold start test or just retrieval check)
    # Actually fetch real history to be accurate
    history_query = db.query(models.Item.ml_id).join(
        models.Swipe, models.Swipe.item_id == models.Item.id
    ).filter(models.Swipe.user_id == user_id, models.Swipe.action.in_(['like', 'superlike'])).all()
    history = [r[0] for r in history_query]
    print(f"User History Length: {len(history)}")
    
    candidates = service.get_recommendations(user_id, history, k_final=100)
    print(f"Inference returned {len(candidates)} candidates.")
    
    cand_ids = [c[0] for c in candidates]
    
    # Check DB existence
    existing = db.query(models.Item.ml_id).filter(models.Item.ml_id.in_(cand_ids)).all()
    existing_ids = set([r[0] for r in existing])
    
    print(f"Found {len(existing_ids)}/{len(cand_ids)} candidates in DB.")
    
    missing = [c for c in cand_ids if c not in existing_ids]
    print(f"Missing Examples: {missing[:10]}")
    
    # Check if DB is empty?
    total_items = db.query(models.Item).count()
    print(f"Total Items in DB: {total_items}")

if __name__ == "__main__":
    check_coverage()
