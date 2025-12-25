from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas, database
from .auth import get_current_user
import numpy as np
from sqlalchemy.sql import text
import datetime

router = APIRouter(
    prefix="/social",
    tags=["social"]
)

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    raw_sim = dot_product / (norm_v1 * norm_v2)
    
    # Calibration: Space is anisotropic, baseline similarity is high (~0.7-0.8).
    # We rescale [0.75, 1.0] -> [0.0, 1.0] to give meaningful percentages.
    baseline = 0.75
    if raw_sim < baseline:
        return 0.0 # Or small epsilon
        
    adjusted_sim = (raw_sim - baseline) / (1 - baseline)
    return min(max(adjusted_sim, 0.0), 1.0)

@router.post("/refresh_vector")
def trigger_refresh_vector(
    request: Request,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    vec = refresh_user_vector(current_user, db, request)
    return {"message": "Vector Refreshed", "vector_preview": vec[:5]}

def refresh_user_vector(user: models.User, db: Session, request: Request):
    """
    Calculates User Vector from Swipe History using Two-Tower Model.
    Saves to DB.
    """
    # 1. Get History
    swipes = db.query(models.Swipe).filter(
        models.Swipe.user_id == user.id,
        models.Swipe.action.in_([models.SwipeAction.like, models.SwipeAction.superlike])
    ).order_by(models.Swipe.timestamp.desc()).limit(50).all()
    
    history_items = []
    # We need ML_IDs for the model. 
    # Check if Item model has ml_id (it should, we added it).
    # If not, we might fail. But let's assume items have ml_id populated.
    
    for s in swipes:
        if s.item.ml_id:
            history_items.append(s.item.ml_id)
            
    print(f"[Profile Refresh] User {user.id}: Found {len(history_items)} valid history items for vector calc.")
            
    # 1.5 Calculate Persona Tags (Auto-Tagging)
    from collections import Counter
    genre_counter = Counter()
    
    for s in swipes:
        # Tagging Logic
        if s.item.genres:
             # genres str: "Action|Sci-Fi"
             g_list = [g.strip() for g in s.item.genres.replace('|', ',').split(',')]
             genre_counter.update(g_list)
             
    # Select Top 3 Genres
    top_genres = [g for g, count in genre_counter.most_common(3)]
    user.tags = [f"{g} Fan" for g in top_genres] # e.g. ["Sci-Fi Fan", "Action Fan"]
    print(f"[Profile Refresh] Generated Tags: {user.tags}")
            
    # 2. Call Model
    inference_service = request.app.state.inference_service
    user_vec = inference_service.get_user_embedding(user.id, history_items)
    
    # 3. Save
    user.embedding = user_vec
    user.embedding_version = 1
    db.commit()
    
    return user_vec

@router.post("/swipe")
def swipe_user(
    swipe_data: schemas.UserSwipe,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    # 1. Record the Interaction
    # Check if already swiped
    existing = db.query(models.UserInteraction).filter(
        models.UserInteraction.liker_id == current_user.id,
        models.UserInteraction.liked_id == swipe_data.liked_user_id
    ).first()
    
    if existing:
        return {"message": "Already swiped", "is_match": False}
        
    interaction = models.UserInteraction(
        liker_id=current_user.id,
        liked_id=swipe_data.liked_user_id,
        action=swipe_data.action
    )
    db.add(interaction)
    db.commit()
    
    # 2. Check for Mutual Match
    is_match = False
    if swipe_data.action == "like":
        # Did they like me?
        reverse_like = db.query(models.UserInteraction).filter(
            models.UserInteraction.liker_id == swipe_data.liked_user_id,
            models.UserInteraction.liked_id == current_user.id,
            models.UserInteraction.action == "like"
        ).first()
        
        if reverse_like:
            is_match = True
            
    return {"message": "Swipe recorded", "is_match": is_match}

@router.get("/candidates", response_model=List[schemas.UserMatchOut])
def get_candidates(
    limit: int = 10,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
    request: Request = None 
):
    """
    Returns swiping candidates (People you haven't swiped on yet).
    Sorted by similarity.
    """
    # 1. Ensure Vector is Up-to-Date
    my_vector = refresh_user_vector(current_user, db, request)
    my_vector = np.array(my_vector)
    
    # 2. Get IDs I've already swiped on
    swiped_ids_query = db.query(models.UserInteraction.liked_id).filter(
        models.UserInteraction.liker_id == current_user.id
    ).all()
    swiped_ids = [x[0] for x in swiped_ids_query]
    swiped_ids.append(current_user.id) # Don't show myself
    
    # Exclude Blocked Users (My blocks AND their blocks)
    blocked_ids_query = db.query(models.BlockedUser.blocked_id).filter(
        models.BlockedUser.blocker_id == current_user.id
    ).all()
    
    blocker_ids_query = db.query(models.BlockedUser.blocker_id).filter(
        models.BlockedUser.blocked_id == current_user.id
    ).all()
    
    blocked_filter = [x[0] for x in blocked_ids_query] + [x[0] for x in blocker_ids_query]
    
    exclusion_ids = swiped_ids + blocked_filter
    
    # 3. Candidate Query
    query = db.query(models.User).filter(
        models.User.id.notin_(exclusion_ids),
        models.User.embedding.isnot(None)
    )
    
    # Filters
    if current_user.location_city:
        query = query.filter(models.User.location_city == current_user.location_city)
        
    if current_user.interested_in and current_user.interested_in != 'both':
        query = query.filter(models.User.gender == current_user.interested_in)
        
    candidates = query.all()
    
    matches = []
    
    for user in candidates:
        other_vector = np.array(user.embedding)
        
        if my_vector.shape != other_vector.shape:
            continue
            
        similarity = cosine_similarity(my_vector, other_vector)
        
        # Add everyone regardless of score (User requested "Show all")
        # But we still calculate score for sorting
        matches.append({
            "user_id": user.id,
            "username": user.username,
            "similarity": float(similarity),
            "user_obj": user # Temp storage
        })
            
    # 4. Sort by similarity desc
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    # 5. Enrich Top Candidates
    final_results = []
    
    for m in matches[:limit]:
         user = m['user_obj']
         # Reuse logic
         common_data = calculate_common_interests(current_user.id, user.id, db)
         
         m['common_movies'] = common_data['movies']
         m['shared_genres'] = common_data['genres']
         
         if common_data['movies']:
             reason = f"You both like {common_data['movies'][0]}"
         elif common_data['genres']:
             reason = f"You are both fans of {common_data['genres'][0]} movies."
         else:
             reason = "Your taste profiles have a strong mathematical correlation."
             
         m['match_reason'] = reason
         m['tags'] = user.tags if user.tags else []
         m['bio'] = user.bio
         
         # Cleanup temp key
         del m['user_obj']
         final_results.append(m)
         
    return final_results

@router.get("/confirmed_matches", response_model=List[schemas.UserMatchOut])
def get_confirmed_matches(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Returns mutual matches (I liked them AND they liked me).
    """
    # 1. My Likes
    my_likes = db.query(models.UserInteraction.liked_id).filter(
        models.UserInteraction.liker_id == current_user.id,
        models.UserInteraction.action == 'like'
    ).subquery()
    
    # 2. Who Liked Me (from the pool of people I liked)
    mutual_matches = db.query(models.UserInteraction.liker_id).filter(
        models.UserInteraction.liked_id == current_user.id,
        models.UserInteraction.action == 'like',
        models.UserInteraction.liker_id.in_(my_likes)
    ).all()
    
    match_ids = [x[0] for x in mutual_matches]
    
    if not match_ids:
        return []
        
    # 3. Fetch User Details
    matched_users = db.query(models.User).filter(models.User.id.in_(match_ids)).all()
    
    final_results = []
    
    for user in matched_users:
         # Calculate Score (Optional but good for display)
         similarity = 0.0
         if current_user.embedding and user.embedding:
             v1 = np.array(current_user.embedding)
             v2 = np.array(user.embedding)
             if v1.shape == v2.shape:
                 similarity = float(cosine_similarity(v1, v2))
                 
         # Reuse enrichment
         common_data = calculate_common_interests(current_user.id, user.id, db)
         
         if common_data['movies']:
             reason = f"You both like {common_data['movies'][0]}"
         elif common_data['genres']:
             reason = f"You are both fans of {common_data['genres'][0]} movies."
         else:
             reason = "You matched!"
             
         final_results.append({
             "user_id": user.id,
             "username": user.username,
             "similarity": similarity,
             "common_movies": common_data['movies'],
             "shared_genres": common_data['genres'],
             "match_reason": reason,
             "tags": user.tags if user.tags else [],
             "bio": user.bio
         })
         
    return final_results

def calculate_common_interests(user_a_id: int, user_b_id: int, db: Session):
    """
    Finds intersection of Liked movies and top shared genres.
    """
    # 1. Get Liked Item IDs for both (Set Intersection)
    # Optimized: Use SQL IN clause or simple fetch
    likes_a = db.query(models.Swipe.item_id).filter(
        models.Swipe.user_id == user_a_id, 
        models.Swipe.action.in_(['like', 'superlike'])
    ).all()
    set_a = {x[0] for x in likes_a}
    
    likes_b = db.query(models.Swipe.item_id).filter(
        models.Swipe.user_id == user_b_id, 
        models.Swipe.action.in_(['like', 'superlike'])
    ).all()
    set_b = {x[0] for x in likes_b}
    
    common_ids = list(set_a.intersection(set_b))
    
    if not common_ids:
        return {'movies': [], 'genres': []}
        
    # 2. Fetch Item Details
    # Randomly pick 3 common items or top popular ones? taking first 3 for now
    example_ids = common_ids[:10] # fetch a few to analyze genres
    
    items = db.query(models.Item).filter(models.Item.id.in_(example_ids)).all()
    
    # Extract Titles
    titles = [item.title for item in items][:3]
    
    # Extract Genres
    # Genre format in DB string: "Action|Sci-Fi" or "Action, Sci-Fi"
    genre_counts = {}
    for item in items:
        if not item.genres: continue
        # Normalize separators
        g_list = item.genres.replace('|', ',').split(',')
        for g in g_list:
            g = g.strip()
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1
                
    # Sort genres by frequency
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    top_genres = [g[0] for g in sorted_genres[:3]]
    
    return {
        'movies': titles,
        'genres': top_genres
    }
