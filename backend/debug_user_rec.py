import sys
import os
import argparse

# Add backend directory to sys.path so we can import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models import User, Swipe, Item
from app.inference_service import inference_engine

def debug_user(nickname):
    db = SessionLocal()
    try:
        # 1. Find User
        user = db.query(User).filter(User.username == nickname).first()
        if not user:
            print(f"Error: User '{nickname}' not found.")
            return

        print(f"\n--- Debugging Recommendations for: {user.username} (ID: {user.id}) ---")

        # 2. Check History
        swipes = db.query(Swipe).join(Item).filter(
            Swipe.user_id == user.id,
            Swipe.rating.isnot(None), 
            Item.ml_id.isnot(None)
        ).all()
        
        print(f"History: {len(swipes)} rated items.")
        if len(swipes) > 0:
            print("Last 5 Swipes:")
            for s in swipes[-5:]:
                print(f"  - {s.item.title} (Rating: {s.rating})")

        # 3. Generate Recommendations
        print("\nCalling InferenceService...")
        # Access internal flow for debugging
        # Fetch directly to see where it breaks
        ratings = [{'ml_id': s.item.ml_id, 'rating': s.rating} for s in swipes]
        
        # Manually run parts of inference service logic here to see
        # But easier to just call the function if we trust the logs added in inference_service.py
        # The logs in inference_service.py ALREADY print top_ml_ids.
        # Wait, the user output in Step 1345 showed:
        # [NCF] Top Recommendations for User 1:
        # (Nothing below it)
        
        # This means result list was empty.
        # Updated signature: top, bottom, vec
        top_items, bottom_items, vec = inference_engine.get_recommendations(user.id, db, limit=10, verbose=True)
        
        if not top_items:
            print("No recommendations returned.")
        else:
            for i, item in enumerate(recommendations):
                print(f"{i+1}. {item.title} (Score: {item.score:.2f})")
                
        # Check DB count
        item_count = db.query(Item).filter(Item.ml_id.isnot(None)).count()
        print(f"\nDebug Info: Total Items in DB with ml_id: {item_count}")
        
        # Check one valid ML_ID from history
        if swipes:
            mid = swipes[0].item.ml_id
            print(f"Sample ML_ID from history: {mid} - Exists in DB? {db.query(Item).filter(Item.ml_id == mid).first() is not None}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Recommendations for a User")
    parser.add_argument("nickname", type=str, help="User's nickname (username)")
    args = parser.parse_args()
    
    debug_user(args.nickname)
