from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas, database
from ..inference_service import InferenceService
from .auth import get_current_user
from sqlalchemy.sql.expression import func
import random
import datetime

router = APIRouter(
    prefix="/feed",
    tags=["feed"]
)

@router.get("/", response_model=List[schemas.ItemOut])
def get_feed(
    limit: int = 25, 
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Returns a MIXED feed for user training/swiping.
    Composition: 10 High Affinity + 5 Low Affinity + 10 Random.
    Outcome: Updates User Vector (Fold-In) but does NOT count as the "Daily Recommendation".
    """
    
    # 1. Get Candidates (Top Only)
    # We ask for 10 high affinity items
    # PASS DB SESSION HERE
    top_items, _, user_vec = InferenceService().get_recommendations(current_user.id, db, limit=10)
    
    # Update User Vector (Continuous Training)
    current_user.embedding = user_vec 
    db.commit()

    # 2. Composition: 10 High Affinity Only
    final_feed = []
    
    for item in top_items:
        if item.poster_path:
            item.image_url = f"https://image.tmdb.org/t/p/w500{item.poster_path}"
        else:
            item.image_url = "https://via.placeholder.com/500x750?text=No+Image"
            
        item.is_recommendation = True
        item.match_type = "none" # Hide match badge (Blind Test)
        item.match_score = getattr(item, 'score', 0.0)
        final_feed.append(item)
        
    return final_feed

@router.get("/daily", response_model=schemas.ItemOut)
def get_daily_match(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    The 'Big Reward'. One single movie recommendation per day.
    Logic: Top NCF Candidates -> Resorted by TMDB Quality (Vote Average).
    """
    
    # 1. DAILY LIMIT CHECK
    if current_user.last_daily_feed:
        last_date = current_user.last_daily_feed.date()
        today = datetime.datetime.utcnow().date()
        # Uncomment to enforce strict daily limit
        # if last_date == today:
        #      from fastapi import HTTPException
        #      raise HTTPException(status_code=429, detail="Daily match already received.")
    
    # 2. Get Top Candidates (Pool of 50)
    # We ignore bottom/vector for this call
    # PASS DB SESSION
    top_items, _, _ = InferenceService().get_recommendations(current_user.id, db, limit=50)
    
    if not top_items:
        # Fallback to popularity
        chosen_item = db.query(models.Item).order_by(models.Item.popularity.desc()).first()
    else:
        # 3. Filter/Rank by Quality (TMDB Vote Average)
        print("\n--- Daily Recommendation Decision Logic ---")
        
        candidates = []
        for item in top_items:
            ncf_score = getattr(item, 'score', 0.0) # 0.5 - 5.0
            tmdb_rating = item.vote_average or 0.0 # 0.0 - 10.0
            
            candidates.append({
                'item': item,
                'ncf_score': ncf_score,
                'tmdb_rating': tmdb_rating
            })
            
        # Strategy: Pick from Top 20 NCF scores, then sort by TMDB
        relevant_candidates = sorted(candidates, key=lambda x: x['ncf_score'], reverse=True)[:20]
        
        # Now sort these 20 by TMDB Rating (Quality)
        final_pick = sorted(relevant_candidates, key=lambda x: x['tmdb_rating'], reverse=True)[0]
        
        chosen_item = final_pick['item']
        print(f"WINNER: {chosen_item.title}")
        print(f"Reason: High NCF ({final_pick['ncf_score']:.2f}) AND Highest TMDB Quality ({final_pick['tmdb_rating']}) among peers.\n")
    
    # 4. Update State
    current_user.last_daily_feed = datetime.datetime.utcnow()
    db.commit()
    
    # 5. Return
    if chosen_item.poster_path:
        chosen_item.image_url = f"https://image.tmdb.org/t/p/w500{chosen_item.poster_path}"
    else:
        chosen_item.image_url = "https://via.placeholder.com/500x750?text=No+Image"
    chosen_item.is_recommendation = True
    chosen_item.match_type = "perfect"
    chosen_item.match_score = getattr(chosen_item, 'score', 5.0)
    
    return chosen_item

@router.get("/match", response_model=schemas.ItemOut)
def get_match_legacy():
    from fastapi import HTTPException
    raise HTTPException(status_code=410, detail="Feature removed. Use /feed/daily for matches.")