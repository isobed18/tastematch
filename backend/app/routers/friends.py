from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List
from .. import models, schemas, database
from .auth import get_current_user
import datetime

router = APIRouter(
    prefix="/friends",
    tags=["friends"]
)

# --- Schemas (Local for now or move to schemas.py) ---
from pydantic import BaseModel

class FriendRequestCreate(BaseModel):
    receiver_id: int

class FriendRequestOut(BaseModel):
    id: int
    sender_id: int
    sender_username: str
    status: str
    timestamp: datetime.datetime

    class Config:
        orm_mode = True

class FriendOut(BaseModel):
    user_id: int
    username: str
    # status: str = "online" # Future
    
    class Config:
        orm_mode = True

# --- Endpoints ---

@router.post("/request", response_model=dict)
def send_friend_request(
    request: FriendRequestCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    if request.receiver_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot add yourself as friend")

    # Check existence
    receiver = db.query(models.User).filter(models.User.id == request.receiver_id).first()
    if not receiver:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if already friends
    existing_friendship = db.query(models.Friendship).filter(
        or_(
            (models.Friendship.user_id == current_user.id) & (models.Friendship.friend_id == request.receiver_id),
            (models.Friendship.user_id == request.receiver_id) & (models.Friendship.friend_id == current_user.id)
        )
    ).first()
    if existing_friendship:
        return {"message": "Already friends"}

    # Check pending request
    existing_request = db.query(models.FriendRequest).filter(
        models.FriendRequest.sender_id == current_user.id,
        models.FriendRequest.receiver_id == request.receiver_id,
        models.FriendRequest.status == "pending"
    ).first()
    if existing_request:
        return {"message": "Request already sent"}

    # Check if they already requested us
    reverse_request = db.query(models.FriendRequest).filter(
        models.FriendRequest.sender_id == request.receiver_id,
        models.FriendRequest.receiver_id == current_user.id,
        models.FriendRequest.status == "pending"
    ).first()
    
    if reverse_request:
        # Auto-accept if they already asked
        return accept_friend_request(reverse_request.id, current_user, db)

    new_req = models.FriendRequest(
        sender_id=current_user.id,
        receiver_id=request.receiver_id
    )
    db.add(new_req)
    db.commit()
    return {"message": "Friend request sent"}

@router.post("/accept/{request_id}")
def accept_friend_request(
    request_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    req = db.query(models.FriendRequest).filter(models.FriendRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
        
    if req.receiver_id != current_user.id:
         # Special case: called internally from send_friend_request (Auto-accept)
         if req.sender_id == current_user.id:
             pass # Logic error handling if calling manually? No, internal call passes objects not IDs usually.
             # Actually for internal reuse, we should extract logic.
             # But for API call validation:
             pass 

    if req.receiver_id != current_user.id and req.sender_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your request")

    req.status = "accepted"
    
    # Create bidirectional friendship (2 rows for easy querying 'my friends')
    f1 = models.Friendship(user_id=req.sender_id, friend_id=req.receiver_id)
    f2 = models.Friendship(user_id=req.receiver_id, friend_id=req.sender_id)
    
    db.add(f1)
    db.add(f2)
    
    # Delete request or keep as log? keeping as accepted log is better but cleaning up is cleaner for MVP
    db.delete(req) 
    
    db.commit()
    return {"message": "Friend request accepted"}

@router.post("/reject/{request_id}")
def reject_friend_request(
    request_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    req = db.query(models.FriendRequest).filter(
        models.FriendRequest.id == request_id,
        models.FriendRequest.receiver_id == current_user.id
    ).first()
    
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")

    db.delete(req) # Just delete it
    db.commit()
    return {"message": "Friend request rejected"}

@router.get("/list", response_model=List[FriendOut])
def get_friends(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    # Because we insert 2 rows, we just query where user_id == me
    friends = db.query(models.Friendship).filter(models.Friendship.user_id == current_user.id).all()
    
    results = []
    for f in friends:
        results.append({
            "user_id": f.friend.id,
            "username": f.friend.username
        })
    return results

@router.get("/requests", response_model=List[FriendRequestOut])
def get_pending_requests(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    reqs = db.query(models.FriendRequest).filter(
        models.FriendRequest.receiver_id == current_user.id,
        models.FriendRequest.status == "pending"
    ).all()
    
    results = []
    for r in reqs:
        results.append({
            "id": r.id,
            "sender_id": r.sender_id,
            "sender_username": r.sender.username,
            "status": r.status,
            "timestamp": r.timestamp
        })
    return results
