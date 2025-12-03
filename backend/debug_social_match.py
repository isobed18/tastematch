import sys
import os
import argparse
import numpy as np
import torch
from sqlalchemy.orm import Session

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models import User, Swipe, Item
from app.inference_service import inference_engine

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def debug_social_match(nickname):
    db = SessionLocal()
    try:
        # 1. Get Target User
        target_user = db.query(User).filter(User.username == nickname).first()
        if not target_user:
            print(f"User '{nickname}' not found.")
            return

        print(f"\n--- Social Match Test for: {target_user.username} (ID: {target_user.id}) ---")
        
        # 2. Get History
        swipes = db.query(Swipe).join(Item).filter(
            Swipe.user_id == target_user.id,
            Swipe.rating.isnot(None),
            Item.ml_id.isnot(None)
        ).all()
        
        if not swipes:
            print("User has no history to invert.")
            return

        print(f"Target User History: {len(swipes)} items.")
        
        # Ensure model is loaded
        if not inference_engine.initialized:
            inference_engine.load_model()
        
        # 3. Generate Target Vector (Real-time to be sure)
        target_history = [{'ml_id': s.item.ml_id, 'rating': s.rating} for s in swipes]
        print("Generating Target Vector...")
        target_v_tensor = inference_engine.fold_in_user(target_history, verbose=False)
        target_vec = target_v_tensor.view(-1).cpu().numpy()

        # 4. Generate Anti-User History
        print("\nGenerating Anti-User (The Nemesis)...")
        anti_history = []
        for s in swipes:
            # Invert Rating: 5 -> 1, 1 -> 5, 3 -> 3
            # Simple inversion formula: 6 - rating
            inverted_rating = 6.0 - s.rating
            # Clamp to valid range just in case
            inverted_rating = max(0.5, min(5.0, inverted_rating))
            
            anti_history.append({'ml_id': s.item.ml_id, 'rating': inverted_rating})
            
        # 5. Generate Anti-User Vector
        print("Optimizing Anti-User Vector...")
        anti_v_tensor = inference_engine.fold_in_user(anti_history, verbose=False)
        anti_vec = anti_v_tensor.view(-1).cpu().numpy()
        
        # 6. Compare
        similarity = cosine_similarity(target_vec, anti_vec)
        
        print(f"\n--- Results ---")
        print(f"Target Vector Norm: {np.linalg.norm(target_vec):.4f}")
        print(f"Anti Vector Norm:   {np.linalg.norm(anti_vec):.4f}")
        print(f"Cosine Similarity:  {similarity:.4f}")
        
        if similarity < 0:
            print("SUCCESS: Vectors are negatively correlated (Opposites).")
        elif similarity < 0.2:
            print("SUCCESS: Vectors are Orthogonal/Unrelated (Low similarity).")
        else:
            print("WARNING: Vectors are still somewhat similar. Model might be dominated by genre bias or popularity.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nickname", type=str)
    args = parser.parse_args()
    debug_social_match(args.nickname)
