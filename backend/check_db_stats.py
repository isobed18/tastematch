from app.database import SessionLocal
from app import models

def check_stats():
    db = SessionLocal()
    try:
        user = db.query(models.User).first()
        if not user:
            print("No users found.")
            return

        print(f"Checking stats for user: {user.username} (ID: {user.id})")
        
        # Count Items
        movie_count = db.query(models.Item).filter(models.Item.type == 'movie').count()
        game_count = db.query(models.Item).filter(models.Item.type == 'game').count()
        print(f"Total Movies: {movie_count}")
        print(f"Total Games: {game_count}")
        
        # Count Swipes
        swipe_count = db.query(models.Swipe).filter(models.Swipe.user_id == user.id).count()
        print(f"Total Swipes by User: {swipe_count}")
        
        # Check remaining items
        swiped_ids = db.query(models.Swipe.item_id).filter(models.Swipe.user_id == user.id).subquery()
        remaining_movies = db.query(models.Item).filter(models.Item.type == 'movie', models.Item.id.notin_(swiped_ids)).count()
        remaining_games = db.query(models.Item).filter(models.Item.type == 'game', models.Item.id.notin_(swiped_ids)).count()
        
        print(f"Remaining Movies: {remaining_movies}")
        print(f"Remaining Games: {remaining_games}")
        
    finally:
        db.close()

if __name__ == "__main__":
    check_stats()
