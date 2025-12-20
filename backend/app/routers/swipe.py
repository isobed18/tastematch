from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas, database
from .auth import get_current_user

router = APIRouter(
    prefix="/swipe",
    tags=["swipe"]
)

@router.post("/")
def create_swipe(
    swipe: schemas.SwipeCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    # Check if item exists
    item = db.query(models.Item).filter(models.Item.id == swipe.item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    # Check if already swiped
    existing_swipe = db.query(models.Swipe).filter(
        models.Swipe.user_id == current_user.id,
        models.Swipe.item_id == swipe.item_id
    ).first()
    
    if existing_swipe:
        raise HTTPException(status_code=400, detail="Already swiped on this item")

    new_swipe = models.Swipe(
        user_id=current_user.id,
        item_id=swipe.item_id,
        action=swipe.action
    )
    

    is_daily_match = (
        current_user.daily_match_ml_id is not None and 
        item.ml_id == current_user.daily_match_ml_id
    )
    
    # Map Action to Rating (NCF Integration)
    if is_daily_match:
        print(f"[Swipe] Daily Match Interaction detected for User {current_user.id} Item {item.id} (ML: {item.ml_id})")
        # Boosted Ratings for Daily Match (High Impact)
        if swipe.action in [models.SwipeAction.like, models.SwipeAction.superlike, "watchlist"]:
            new_swipe.rating = 5.0 # Max Score
        elif swipe.action == models.SwipeAction.dislike:
            new_swipe.rating = 0.5 # Strong Penalty (Min Score)
    else:
        # Standard Ratings
        if swipe.action == models.SwipeAction.like:
            new_swipe.rating = 4.0
        elif swipe.action == models.SwipeAction.superlike:
            new_swipe.rating = 5.0
        elif swipe.action == models.SwipeAction.dislike:
            new_swipe.rating = 2.0
        elif swipe.action == "watchlist":
            new_swipe.rating = 4.5
    
    db.add(new_swipe)
    db.commit()
    
    # Not: Burada eskiden user vektörü güncellenirdi. 
    # Yeni sistemde vektörler inference anında hesaplandığı için (veya asenkron batch job ile)
    # burada anlık bir işlem yapmamıza gerek yok. Sadece kaydetmek yeterli.
    
    return {"message": "Swipe recorded"}