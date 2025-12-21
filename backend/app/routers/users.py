from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import or_
from .. import models, schemas, database
from .auth import get_current_user

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.get("/{user_id}/profile", response_model=schemas.UserMatchOut)
def get_user_profile(
    user_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Get public profile of a user (for Profile Viewer).
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Check if blocked (either way)
    block_check = db.query(models.BlockedUser).filter(
        or_(
            (models.BlockedUser.blocker_id == current_user.id) & (models.BlockedUser.blocked_id == user_id),
            (models.BlockedUser.blocker_id == user_id) & (models.BlockedUser.blocked_id == current_user.id)
        )
    ).first()
    
    if block_check:
        raise HTTPException(status_code=403, detail="User is unavailable")

    # Reuse logic for enrichments if needed, or just return basic info
    return {
        "user_id": user.id,
        "username": user.username,
        "similarity": 0.0, # Not calculated here
        "common_movies": [],
        "shared_genres": [],
        "tags": user.tags if user.tags else [],
        "bio": user.bio,
        "match_reason": "Profile View"
    }

@router.post("/{user_id}/block")
def block_user(
    user_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot block yourself")
        
    existing = db.query(models.BlockedUser).filter(
        models.BlockedUser.blocker_id == current_user.id,
        models.BlockedUser.blocked_id == user_id
    ).first()
    
    if existing:
        return {"message": "User already blocked"}
        
    new_block = models.BlockedUser(
        blocker_id=current_user.id,
        blocked_id=user_id
    )
    db.add(new_block)
    
    # Optional: Delete existing match/swipe?
    # db.query(models.UserInteraction).filter(...)
    # For now, just blocking prevents new interactions/messaging via the 'get_candidates' filter update
    
    db.commit()
    return {"message": "User blocked"}

@router.post("/{user_id}/report")
def report_user(
    user_id: int,
    report_data: schemas.ReportCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot report yourself")
        
    new_report = models.Report(
        reporter_id=current_user.id,
        reported_id=user_id,
        reason=report_data.reason,
        details=report_data.details
    )
    db.add(new_report)
    db.commit()
    return {"message": "Report submitted"}
