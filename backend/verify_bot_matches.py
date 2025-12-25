import os
import sys
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from app.database import engine
from app import models

# Setup DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0.0

def verify_matches():
    print("Fetching Bots...")
    bots = db.query(models.User).filter(models.User.username.in_([
        "ActionAli", "ActionAyse", 
        "RomanceRuyam", "RomanceRifat", 
        "HorrorHasan", "SciFiSelin"
    ])).all()
    
    bot_map = {b.username: np.array(b.embedding) for b in bots if b.embedding}
    
    if len(bot_map) < 2:
        print("Not enough bots with embeddings found.")
        return

    print("\n--- SIMILARITY MATRIX ---")
    
    pairs = [
        ("ActionAli", "ActionAyse", "High"),
        ("RomanceRuyam", "RomanceRifat", "High"),
        ("ActionAli", "RomanceRuyam", "Low"),
        ("HorrorHasan", "RomanceRuyam", "Low"),
        ("SciFiSelin", "ActionAli", "Medium/High"), # SciFi & Action overlap
    ]
    
    for u1_name, u2_name, expected in pairs:
        if u1_name in bot_map and u2_name in bot_map:
            sim = cosine_similarity(bot_map[u1_name], bot_map[u2_name])
            print(f"{u1_name} vs {u2_name}: {sim:.4f} (Expected: {expected})")
        else:
            print(f"Skipping {u1_name} vs {u2_name} (Missing Data)")

    db.close()

if __name__ == "__main__":
    verify_matches()
