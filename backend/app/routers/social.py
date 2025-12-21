from fastapi import Request
import datetime

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
            
    # 2. Call Model
    inference_service = request.app.state.inference_service
    user_vec = inference_service.get_user_embedding(user.id, history_items)
    
    # 3. Save
    user.embedding = user_vec
    user.embedding_version = 1
    db.commit()
    
    return user_vec

@router.get("/matches", response_model=List[schemas.UserMatchOut])
def get_user_matches(
    limit: int = 10,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db),
    request: Request = None 
):
    """
    Finds soulmates using Two-Tower Vectors + Hard Filters.
    """
    # 1. Ensure Vector Exists
    my_vector = current_user.embedding
    if not my_vector:
        # Auto-refresh
        my_vector = refresh_user_vector(current_user, db, request)
        
    my_vector = np.array(my_vector)
    
    # 2. Hard Filters (Candidate Selection)
    query = db.query(models.User).filter(
        models.User.id != current_user.id,
        models.User.embedding.isnot(None)
    )
    
    # Location Filter
    if current_user.location_city:
        query = query.filter(models.User.location_city == current_user.location_city)
        
    # Gender Preference Filter
    # Logic: If I am 'male' interested in 'female', show 'female' interested in 'male' (or 'both').
    if current_user.interested_in and current_user.interested_in != 'both':
        query = query.filter(models.User.gender == current_user.interested_in)
        
    # Reverse Gender Filter (They must be interested in ME)
    if current_user.gender:
        # They should be interested in my gender, or 'both'
        # SQL: interested_in == my_gender OR interested_in == 'both'
        pass # Complexity for MVP, skip strict reverse check for now, or implement simple OR
        
    candidates = query.all()
    
    matches = []
    
    for user in candidates:
        other_vector = np.array(user.embedding)
        
        if my_vector.shape != other_vector.shape:
            continue
            
        similarity = cosine_similarity(my_vector, other_vector)
        
        if similarity > 0.5: # 50% match minimum
            matches.append({
                "user_id": user.id,
                "username": user.username,
                "similarity": float(similarity)
            })
            
    # 3. Sort by similarity desc
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    # 4. Enrich with "Why Matched" (Only for top results to save DB calls)
    final_results = []
    
    for m in matches[:limit]:
         # Calculate Shared Interests
         common_data = calculate_common_interests(current_user.id, m['user_id'], db)
         
         m['common_movies'] = common_data['movies']
         m['shared_genres'] = common_data['genres']
         
         # Generate Icebreaker
         if common_data['movies']:
             m['match_reason'] = f"You both like {common_data['movies'][0]}"
         elif common_data['genres']:
             m['match_reason'] = f"You are both fans of {common_data['genres'][0]} movies."
         else:
             m['match_reason'] = "Your taste profiles have a strong mathematical correlation."
             
         final_results.append(m)
         
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
