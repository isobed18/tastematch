from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas, database
from .auth import get_current_user
import numpy as np
from sqlalchemy.sql import text

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
    return dot_product / (norm_v1 * norm_v2)

@router.get("/matches", response_model=List[schemas.UserMatchOut])
def get_user_matches(
    limit: int = 10,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Finds other users with similar taste based on NCF embeddings.
    """
    # 1. Check if current user has an embedding (generated from daily feed)
    if not current_user.embedding:
        raise HTTPException(status_code=400, detail="You need to get your Daily Movie Match first to generate your taste profile.")
    
    my_vector = np.array(current_user.embedding)
    
    # 2. Fetch other users with embeddings
    # Using raw SQL or ORM to filter non-null embeddings is safer
    # SQLAlchemy JSON checking can be dialect specific, so lets fetch all and filter in python for MVP (assuming low user count)
    # For production: Use pgvector or similar.
    
    # Fetch ID, Username, Embedding
    others = db.query(models.User).filter(
        models.User.id != current_user.id,
        models.User.embedding.isnot(None)
    ).all()
    
    matches = []
    
    for user in others:
        if not user.embedding: continue
        
        other_vector = np.array(user.embedding)
        
        # Check dimensions
        if my_vector.shape != other_vector.shape:
            continue
            
        similarity = cosine_similarity(my_vector, other_vector)
        
        # Convert -1...1 to 0...100% (?) or just keep score
        # Cosine 1.0 = Identical
        # Cosine 0.0 = Orthogonal
        # Cosine -1.0 = Opposite
        
        # Let's filter reasonable matches (> 0.5?)
        if similarity > 0.0: # Show positive correlations only
            matches.append({
                "user_id": user.id,
                "username": user.username,
                "similarity": float(similarity)
            })
            
    # 3. Sort by similarity desc
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    return matches[:limit]
