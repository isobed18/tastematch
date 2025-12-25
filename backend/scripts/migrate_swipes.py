
import sys
import os
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import Swipe, Interaction, User

def migrate_swipes():
    db = SessionLocal()
    print("Starting migration from 'swipes' to 'interactions'...")
    
    try:
        # 1. Get all swipes
        swipes = db.query(Swipe).all()
        print(f"Found {len(swipes)} swipes to migrate.")
        
        count = 0
        for swipe in swipes:
            # Check if this interaction already exists (idempotency)
            exists = db.query(Interaction).filter(
                Interaction.user_id == swipe.user_id,
                Interaction.item_id == swipe.item_id,
                Interaction.action == swipe.action
            ).first()
            
            if exists:
                continue

            # Determine Weight based on Action
            weight = 1.0
            if swipe.action == "like":
                weight = 1.0
            elif swipe.action == "dislike":
                weight = -1.0
            elif swipe.action == "superlike":
                weight = 2.0
            elif swipe.action == "watchlist":
                weight = 1.5
            
            new_interaction = Interaction(
                user_id=swipe.user_id,
                item_id=swipe.item_id,
                item_type="movie", # Assuming old swipes are movies
                action=swipe.action,
                weight=weight,
                timestamp=swipe.timestamp
            )
            
            db.add(new_interaction)
            count += 1
            
            if count % 100 == 0:
                print(f"Migrated {count} records...")
        
        db.commit()
        print(f"Migration complete. {count} new interactions created.")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    migrate_swipes()
