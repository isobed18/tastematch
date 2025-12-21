import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from app.database import engine
from app import models

# Setup DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

def cosine_similarity_matrix(vectors):
    """
    Computes Cosine Similarity Matrix for a list of vectors.
    Vectors: numpy array [N, D]
    Returns: [N, N] matrix
    """
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norm + 1e-8)
    return np.dot(normalized, normalized.T)

def analyze_all_users():
    print("Fetching All Users with Embeddings...")
    users = db.query(models.User).filter(models.User.embedding.isnot(None)).all()
    
    if len(users) < 2:
        print("Not enough users with embeddings.")
        return

    # Handle JSON storage of lists - Filter out None or Empty
    valid_vectors = []
    valid_usernames = []
    for u in users:
        if isinstance(u.embedding, list) and len(u.embedding) > 0:
            valid_vectors.append(u.embedding)
            valid_usernames.append(u.username)

    if not valid_vectors:
        print("No valid embeddings found.")
        return

    vectors = np.array(valid_vectors)
    usernames = valid_usernames 
    
    print(f"Found {len(users)} users. Vector Shape: {vectors.shape}")
    
    # Compute Matrix
    sim_matrix = cosine_similarity_matrix(vectors)
    
    # Create DataFrame for better visualization
    df = pd.DataFrame(sim_matrix, index=usernames, columns=usernames)
    
    # Print Matrix
    print("\n--- FULL SIMILARITY MATRIX ---")
    print(df.round(4))
    
    # Statistics
    mask = np.ones(sim_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0) # Exclude self-similarity (1.0)
    
    off_diag_values = sim_matrix[mask]
    
    print("\n--- STATISTICS ---")
    print(f"Mean Similarity: {np.mean(off_diag_values):.4f}")
    print(f"Min Similarity:  {np.min(off_diag_values):.4f}")
    print(f"Max Similarity:  {np.max(off_diag_values):.4f}")
    print(f"Std Dev:         {np.std(off_diag_values):.4f}")
    
    # Check for "Isotropic" behavior (Centered around 0?)
    # If Mean is ~0.8, we have severe Anisotropy (Embedding Collapse / Cone Effect)

if __name__ == "__main__":
    analyze_all_users()
