import os
import sys
import collections
from sqlalchemy.orm import sessionmaker

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from app.database import engine
from app import models

# Setup DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

def analyze_user_vs_bot(user_id, bot_username):
    print(f"--- ANALYZING User {user_id} vs {bot_username} ---")
    
    # Get Users
    user = db.query(models.User).filter(models.User.id == user_id).first()
    bot = db.query(models.User).filter(models.User.username == bot_username).first()
    
    if not user or not bot:
        print("User or Bot not found.")
        return

    # Helper to get genre stats
    def get_genre_stats(u_id):
        # Join Swipe -> Item
        swipes = db.query(models.Swipe, models.Item).join(models.Item).filter(
            models.Swipe.user_id == u_id,
            models.Swipe.action.in_([models.SwipeAction.like, models.SwipeAction.superlike])
        ).all()
        
        genre_counts = collections.Counter()
        total_swipes = len(swipes)
        
        for s, item in swipes:
            if not item.genres: continue
            # Normalize
            genres = [g.strip() for g in item.genres.replace('|', ',').split(',')]
            for g in genres:
                genre_counts[g] += 1
                
        return genre_counts, total_swipes

    u_stats, u_total = get_genre_stats(user.id)
    b_stats, b_total = get_genre_stats(bot.id)
    
    print("--- DIVERGENCE REPORT ---")
    unique_user = sorted(list(set(u_stats.keys()) - set(b_stats.keys())))
    print(f"User Unique: {unique_user}")
    
    unique_bot = sorted(list(set(b_stats.keys()) - set(u_stats.keys())))
    print(f"Bot Unique: {unique_bot}")
    print("-------------------------")
        
    db.close()

if __name__ == "__main__":
    analyze_user_vs_bot(2, "RomanceRuyam")
