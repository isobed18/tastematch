import os
import sys
import random
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from app.database import Base, engine
from app import models
from app.inference_service import InferenceService
from app.auth import get_password_hash

# Setup DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

# Setup Inference
print("Initializing Inference Service...")
inference_service = InferenceService()

# BOT CONFIGURATION
# Hetero only as requested for simplicity
BOTS = [
    {
        "username": "ActionAli",
        "gender": "male",
        "interested_in": "female",
        "city": "Istanbul",
        "bio": "I live for adrenaline. Explosions and car chases!",
        "liked_genres": ["Action", "Adventure", "Crime"]
    },
    {
        "username": "RomanceRuyam",
        "gender": "female",
        "interested_in": "male",
        "city": "Istanbul",
        "bio": "Looking for my fairytale ending. 🌹",
        "liked_genres": ["Romance", "Drama"]
    },
    {
        "username": "HorrorHasan",
        "gender": "male",
        "interested_in": "female",
        "city": "Istanbul",
        "bio": "Scare me if you can. Huge fan of slasher movies.",
        "liked_genres": ["Horror", "Thriller", "Mystery"]
    },
    {
        "username": "SciFiSelin",
        "gender": "female",
        "interested_in": "male",
        "city": "Istanbul",
        "bio": "Space is the final frontier. 🚀",
        "liked_genres": ["Science Fiction", "Fantasy", "Action"]
    },
    # --- COUNTERPARTS ---
    {
        "username": "RomanceRifat",
        "gender": "male",
        "interested_in": "female",
        "city": "Istanbul",
        "bio": "I believe in true love. 💍",
        "liked_genres": ["Romance", "Drama"]
    },
    {
        "username": "ActionAyse",
        "gender": "female",
        "interested_in": "male",
        "city": "Istanbul",
        "bio": "Fast cars and fighting spirit.",
        "liked_genres": ["Action", "Adventure"]
    }
]

def create_bots():
    print(f"Creating {len(BOTS)} bots...")
    
    # Fetch a pool of movies to swipe on
    # We need a mix so they can find things they like and dislike
    all_items = db.query(models.Item).filter(models.Item.ml_id.isnot(None)).limit(500).all()
    print(f"Loaded {len(all_items)} movies for simulation.")
    
    for bot_conf in BOTS:
        # 1. Check if exists
        user = db.query(models.User).filter(models.User.username == bot_conf['username']).first()
        if not user:
            user = models.User(
                username=bot_conf['username'],
                hashed_password=get_password_hash("123456"),
                gender=bot_conf['gender'],
                interested_in=bot_conf['interested_in'],
                location_city=bot_conf['city'],
                bio=bot_conf['bio']
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Created user: {user.username}")
        else:
            print(f"Updating existing user: {user.username}")
            user.gender = bot_conf['gender']
            user.interested_in = bot_conf['interested_in']
            user.location_city = bot_conf['city']
            user.bio = bot_conf['bio']
            db.commit()

        # 2. Simulate Swipes
        # We want ~50 swipes per user to get a good vector
        # Logic: If item genre overlaps with liked_genres -> LIKE, else DISLIKE
        
        print(f"Simulating swipes for {user.username} ({bot_conf['liked_genres']})...")
        
        # Clear existing swipes for clean test
        db.query(models.Swipe).filter(models.Swipe.user_id == user.id).delete()
        db.commit()
        
        liked_history_ml_ids = []
        
        # Shuffle items to get random selection
        random.shuffle(all_items)
        simulation_pool = all_items[:100] 
        
        for item in simulation_pool:
            if not item.genres:
                continue
                
            item_genres = [g.strip() for g in item.genres.replace('|', ',').split(',')]
            
            # Check overlap
            is_liked = any(g in bot_conf['liked_genres'] for g in item_genres)
            
            action = models.SwipeAction.like if is_liked else models.SwipeAction.dislike
            
            # Add some noise (10% chance to do random action)
            if random.random() < 0.1:
                action = models.SwipeAction.dislike if is_liked else models.SwipeAction.like
                
            swipe = models.Swipe(
                user_id=user.id,
                item_id=item.id,
                action=action,
                timestamp=datetime.datetime.utcnow()
            )
            db.add(swipe)
            
            if action in [models.SwipeAction.like, models.SwipeAction.superlike]:
                liked_history_ml_ids.append(item.ml_id)
                
        db.commit()
        print(f"  - Swiped {len(simulation_pool)} items. Liked {len(liked_history_ml_ids)}.")
        
        # 3. Generate Vector
        if liked_history_ml_ids:
            print(f"  - Generating User Vector from {len(liked_history_ml_ids)} likes...")
            vector = inference_service.get_user_embedding(user.id, liked_history_ml_ids)
            
            user.embedding = vector
            user.embedding_version = 1
            db.commit()
            print(f"  - Vector generated and saved.")
        else:
            print(f"  - WARNING: No likes, cannot generate vector.")

    db.close()
    print("Bot creation complete.")

if __name__ == "__main__":
    create_bots()
