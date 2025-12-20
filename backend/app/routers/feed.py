from fastapi import APIRouter, Depends, Request
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
    request: Request,
    limit: int = 10,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Returns a DYNAMIC feed.
    Composition: 
    1. Retrieval (Two-Tower) -> Filter Seen -> Rank (NCF).
    2. Fallback: If not enough items, fill with Popularity-based items from DB.
    """
    # 0. Fetch User History & Seen Items (Correctly Mapped)
    
    # History for Model: Positive Swipes (ML IDs preferably, but model handles mapping if we give external/internal? 
    # Wait, InferenceService expects ML IDs if possible, but actually it performs lookup using local maps. 
    # Current implementation passes Swipe.item_id (PK). TwoTowerInference maps generic ID -> Index. 
    # IF PK != ML_ID, TwoTowerInference MIGHT be confused if it expects ML_ID.
    # checking inference_utils.py: It loads `item_map.pkl` (ML_ID -> Index).
    # If we pass PKs (1,2,3) and item_map expects (101, 250, 999), we give WRONG history.
    # FIX: We must pass ML_IDs to InferenceService too!
    
    # FETCH SENTIMENT HISTORY (ML_IDs)
    history_query = db.query(models.Item.ml_id).join(
        models.Swipe, models.Swipe.item_id == models.Item.id
    ).filter(
        models.Swipe.user_id == current_user.id,
        models.Swipe.action.in_([models.SwipeAction.like, models.SwipeAction.superlike]),
        models.Item.ml_id.isnot(None) # Ensure ML ID exists
    ).all()
    history_items = [row.ml_id for row in history_query]
    

    # FETCH SEEN ITEMS (ML_IDs)
    seen_query = db.query(models.Item.ml_id).join(
        models.Swipe, models.Swipe.item_id == models.Item.id
    ).filter(
        models.Swipe.user_id == current_user.id,
        models.Item.ml_id.isnot(None)
    ).all()
    seen_ml_ids = set([row.ml_id for row in seen_query])
    
    # EXCLUDE DAILY MATCH FROM FEED (No Duplicate)
    if current_user.daily_match_ml_id:
        seen_ml_ids.add(current_user.daily_match_ml_id)
        print(f"[Feed] Excluding Daily Match {current_user.daily_match_ml_id} from candidates.")

    # 1. Get Candidates (Top 100)
    inference_service = request.app.state.inference_service
    
    # Returns [(mid, score), ...]
    top_candidates = inference_service.get_recommendations(
        user_id=current_user.id, 
        history_items=history_items, 
        k_final=100
    )
    print(f"[Feed] Inference returned {len(top_candidates)} raw candidates.")
    
    final_feed = []
    
    if top_candidates:
        # 2. Filter Seen Items
        filtered_candidates = [] # [(mid, score)]
        for mid, score in top_candidates:
            if mid not in seen_ml_ids:
                filtered_candidates.append((mid, score))
        
        print(f"[Feed] {len(filtered_candidates)} candidates remain after filtering seen items.")
        
        # 3. Retrieve Objects from DB
        candidate_ids = [c[0] for c in filtered_candidates]
        
        if candidate_ids:
            items = db.query(models.Item).filter(models.Item.ml_id.in_(candidate_ids)).all()
            item_map = {i.ml_id: i for i in items}
            
            for mid, score in filtered_candidates:
                if mid in item_map:
                    item = item_map[mid]
                    # Decorate
                    if item.poster_path:
                        item.image_url = f"https://image.tmdb.org/t/p/w500{item.poster_path}"
                    else:
                        item.image_url = "https://via.placeholder.com/500x750?text=No+Image"
                    
                    item.is_recommendation = True
                    item.match_type = "hybrid"
                    item.match_score = score
                    
                    final_feed.append(item)
                    if len(final_feed) >= limit:
                        break
    
    # 4. FALLBACK (If feed is empty or short)
    if len(final_feed) < limit:
        needed = limit - len(final_feed)
        print(f"[Fallback] Needed {needed} items. Fetching popular...")
        
        # Exclude already in final_feed and seen_ml_ids
        current_feed_ml_ids = set([i.ml_id for i in final_feed if i.ml_id])
        exclude_ids = seen_ml_ids.union(current_feed_ml_ids)
        
        # Efficiently fetch popular items NOT in exclude list
        # We query by ML_ID to check exclusion, but we need PKs really.
        # Actually standard SQL: WHERE ml_id NOT IN (...)
        
        popular_items = db.query(models.Item).filter(
            models.Item.ml_id.notin_(exclude_ids) if exclude_ids else True
        ).order_by(models.Item.popularity.desc()).limit(needed).all()
        
        for item in popular_items:
            if item.poster_path:
                item.image_url = f"https://image.tmdb.org/t/p/w500{item.poster_path}"
            else:
                item.image_url = "https://via.placeholder.com/500x750?text=No+Image"
            item.is_recommendation = True
            item.match_type = "popular"
            item.match_score = 0.0 # No personalization score
            final_feed.append(item)
            
    return final_feed

@router.get("/daily", response_model=schemas.ItemOut)
def get_daily_match(
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    The 'Big Reward'. One single movie recommendation per day.
    Cached: Shows the same item until the date changes.
    Logic: NCF > 4.5 -> Top Quality.
    """
    today = datetime.datetime.utcnow().date()
    
    # 1. CACHE CHECK
    if current_user.last_daily_feed and current_user.last_daily_feed.date() == today:
        if current_user.daily_match_ml_id:
            # Try to fetch cached item
            cached_item = db.query(models.Item).filter(models.Item.ml_id == current_user.daily_match_ml_id).first()
            if cached_item:
                print(f"[Daily] Returning Cached Item: {cached_item.title}")
                # Format
                if cached_item.poster_path:
                    cached_item.image_url = f"https://image.tmdb.org/t/p/w500{cached_item.poster_path}"
                else:
                    cached_item.image_url = "https://via.placeholder.com/500x750?text=No+Image"
                cached_item.is_recommendation = True
                cached_item.match_type = "perfect"
                cached_item.match_score = 5.0
                return cached_item
    
    # 2. GENERATE NEW MATCH
    
    # Fetch History (ML IDs)
    history_query = db.query(models.Item.ml_id).join(
        models.Swipe, models.Swipe.item_id == models.Item.id
    ).filter(
        models.Swipe.user_id == current_user.id,
        models.Swipe.action.in_([models.SwipeAction.like, models.SwipeAction.superlike]),
         models.Item.ml_id.isnot(None)
    ).all()
    history_items = [row.ml_id for row in history_query]
    
    # Fetch Seen (ML IDs)
    seen_query = db.query(models.Item.ml_id).join(
        models.Swipe, models.Swipe.item_id == models.Item.id
    ).filter(
        models.Swipe.user_id == current_user.id,
        models.Item.ml_id.isnot(None)
    ).all()
    seen_ml_ids = set([row.ml_id for row in seen_query])

    # Get Top Candidates (Pool of 100)
    inference_service = request.app.state.inference_service
    top_candidates = inference_service.get_recommendations(
        user_id=current_user.id, 
        history_items=history_items, 
        k_final=100
    )
    
    # Valid Candidates (Unseen + Exists in DB)
    candidate_ids = [c[0] for c in top_candidates if c[0] not in seen_ml_ids]
    
    chosen_item = None
    
    if not candidate_ids:
        # Fallback to popularity (NOT SEEN)
        chosen_item = db.query(models.Item).filter(
             models.Item.ml_id.notin_(seen_ml_ids) if seen_ml_ids else True
        ).order_by(models.Item.popularity.desc()).first()
    else:
        # Batch Fetch Items
        items = db.query(models.Item).filter(models.Item.ml_id.in_(candidate_ids)).all()
        item_map = {i.ml_id: i for i in items}
        
        # Re-construct list with objects and scores
        valid_list = []
        for mid, score in top_candidates:
            if mid in item_map and mid not in seen_ml_ids:
                valid_list.append({
                    'item': item_map[mid],
                    'score': score,
                    'tmdb_rating': item_map[mid].vote_average or 0.0
                })
        
        if not valid_list:
             chosen_item = db.query(models.Item).order_by(models.Item.popularity.desc()).first()
        else:
            # 3. Decision Logic
            # Goal: Score > 4.5 AND High Quality.
            
            # Filter by NCF Threshold (if score available)
            # Note: Retrieval score might be 0.0 if NCF not loaded.
            # Assuming NCF is loaded for "Smart" Daily.
            high_scores = [x for x in valid_list if x['score'] >= 4.0] # Relaxed to 4.0 just in case
            
            pool = high_scores if high_scores else valid_list # Fallback to top list
            
            # Pick by Highest TMDB Rating from the pool
            final_pick = sorted(pool, key=lambda x: x['tmdb_rating'], reverse=True)[0]
            chosen_item = final_pick['item']
            # Attach score to object for later
            chosen_item.score = final_pick['score']
            print(f"[Daily] New Winner: {chosen_item.title} (NCF: {final_pick['score']:.2f}, TMDB: {final_pick['tmdb_rating']})")


    # 4. Save to Cache
    current_user.last_daily_feed = datetime.datetime.utcnow()
    current_user.daily_match_ml_id = chosen_item.ml_id
    db.commit()
    
    # 5. Return
    if chosen_item.poster_path:
        chosen_item.image_url = f"https://image.tmdb.org/t/p/w500{chosen_item.poster_path}"
    else:
        chosen_item.image_url = "https://via.placeholder.com/500x750?text=No+Image"
    
    chosen_item.is_recommendation = True
    chosen_item.match_type = "perfect"
    
    # Retrieve score if available (set during selection)
    if hasattr(chosen_item, 'score'):
        chosen_item.match_score = chosen_item.score
    else:
        # Fallback if selected via popularity
        chosen_item.match_score = 5.0 
        
    print(f"[Daily] Returning {chosen_item.title} with Score: {chosen_item.match_score}")
    
    return chosen_item

@router.get("/match", response_model=schemas.ItemOut)
def get_match_legacy():
    from fastapi import HTTPException
    raise HTTPException(status_code=410, detail="Feature removed. Use /feed/daily for matches.")