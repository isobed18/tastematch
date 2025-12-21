from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List, Optional
import numpy as np
from .. import models, schemas, database
from .auth import get_current_user
 

router = APIRouter(
    prefix="/date",
    tags=["date"]
)

@router.get("/recommendations")
def get_date_recommendations(
    request: Request,
    partner_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    # 1. Get Partner
    partner = db.query(models.User).filter(models.User.id == partner_id).first()
    if not partner:
        raise HTTPException(status_code=404, detail="Partner not found")
        
    # 2. Get Histories
    # Helper to get history items
    def get_history(uid):
        swipes = db.query(models.Swipe).filter(
            models.Swipe.user_id == uid,
            models.Swipe.action == models.SwipeAction.like
        ).all()
        return [s.item_id for s in swipes]

    my_history = get_history(current_user.id)
    partner_history = get_history(partner.id)
    
    # 3. Get Embeddings
    inference_service = request.app.state.inference_service
    
    vec_a = inference_service.get_user_embedding(current_user.id, my_history)
    vec_b = inference_service.get_user_embedding(partner.id, partner_history)
    
    if not vec_a or not vec_b:
        raise HTTPException(status_code=500, detail="Could not compute vectors")
        
    # 4. Compute Mean
    mean_vec = (np.array(vec_a) + np.array(vec_b)) / 2.0
    
    # 5. Get Recommendations
    # Get more than needed to filter seen
    raw_recs = inference_service.recommend_for_vector(mean_vec, k=50)
    
    # 6. Filter Seen Items (unless we want rewatch?)
    # For now, filter out items seen by EITHER
    seen_set = set(my_history + partner_history)
    
    final_recs = []
    for item_id in raw_recs:
        if item_id not in seen_set:
            final_recs.append(item_id)
            if len(final_recs) >= 5:
                break
                
    # 7. Fetch Item Details
    items = db.query(models.Item).filter(models.Item.id.in_(final_recs)).all()
    
    # Sort by order in final_recs (which is score order)
    items_map = {i.id: i for i in items}
    ordered_items = []
    for rid in final_recs:
        if rid in items_map:
            ordered_items.append(items_map[rid])
            
    return ordered_items
