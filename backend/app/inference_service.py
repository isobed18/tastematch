import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import Item, Swipe, SwipeAction

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_NCF_DIR = os.path.join(BASE_DIR, "project", "ncf")
RUNS_DIR = os.path.join(BASE_DIR, "runs_ncf", "run_20251216_154202") # BEST MODEL
MODEL_PATH = os.path.join(RUNS_DIR, "best_model.pth")
GENOME_PATH = os.path.join(RUNS_DIR, "genome_matrix.npy")
MOVIE_MAPPING_PATH = os.path.join(RUNS_DIR, "movie_mapping.pkl")
USER_MAPPING_PATH = os.path.join(RUNS_DIR, "user_mapping.pkl")

# Add NCF to path
import sys
sys.path.append(PROJECT_NCF_DIR)
import config
# DYNAMIC IMPORT to avoid issues if model.py changes
from model import HybridNCF 

class InferenceService:
    def __init__(self):
        self.model = None
        self.movie_mapping = None # tmdb_id/ml_id -> model_idx
        self.genome_matrix = None
        self.initialized = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        if self.initialized: return
        print(f"Loading NCF Model from {RUNS_DIR}...")
        try:
            # Load Mappings
            with open(MOVIE_MAPPING_PATH, 'rb') as f:
                self.movie_mapping = pickle.load(f)
            
            # Need User Mapping just for num_users dimension initialization
            with open(USER_MAPPING_PATH, 'rb') as f:
                user_mapping = pickle.load(f)
                num_train_users = len(user_mapping)
                
            # Load Genome
            self.genome_matrix = np.load(GENOME_PATH)
            
            # Initialize Model
            self.model = HybridNCF(
                num_users=num_train_users,
                num_movies=len(self.movie_mapping),
                genome_dim=config.GENOME_DIM, # 1128
                embedding_dim=config.EMBEDDING_DIM
            )
            
            # Load Weights
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Freeze weights to be safe (we only optimize user vec)
            for param in self.model.parameters():
                param.requires_grad = False
            
            # LOAD DB IDs (Validation Set)
            # Only recommend items that actually exist in our local DB
            try:
                db = SessionLocal()
                existing_ids = db.query(Item.ml_id).filter(Item.ml_id.isnot(None)).all()
                self.available_item_ids = set([i[0] for i in existing_ids])
                db.close()
                print(f"[NCF] Loaded {len(self.available_item_ids)} available items from DB.")
            except Exception as e:
                print(f"[NCF] Warning: Could not load DB items: {e}")
                self.available_item_ids = set()
                
            self.initialized = True
            print("NCF Model Loaded Successfully.")
            
        except Exception as e:
            print(f"FATAL: NCF Model load failed: {e}")

    def fold_in_user(self, liked_items_with_ratings, verbose=False):
        """
        Creates and trains a temporary user vector based on ratings.
        liked_items_with_ratings: list of dict {'ml_id': int, 'rating': float}
        """
        # If no ratings, return random vector
        if not liked_items_with_ratings:
            return torch.nn.Parameter(torch.normal(0, 0.1, size=(1, config.EMBEDDING_DIM)).to(self.device))

        # 1. Prepare Data
        movie_indices = []
        ratings_vals = []
        
        for item in liked_items_with_ratings:
            mid = item['ml_id']
            if mid in self.movie_mapping:
                movie_indices.append(self.movie_mapping[mid])
                ratings_vals.append(item['rating'])
                
        if not movie_indices:
             return torch.nn.Parameter(torch.normal(0, 0.1, size=(1, config.EMBEDDING_DIM)).to(self.device))
             
        # Convert to Tensor
        m_tensor = torch.tensor(movie_indices, dtype=torch.long).to(self.device)
        r_tensor = torch.tensor(ratings_vals, dtype=torch.float32).to(self.device)
        f_tensor = torch.tensor(self.genome_matrix[movie_indices], dtype=torch.float32).to(self.device)

        # 2. Initialize Vector (Random Normal)
        user_vector = torch.nn.Parameter(torch.normal(0, 0.1, size=(1, config.EMBEDDING_DIM)).to(self.device))
        
        # 3. Optimize
        # Use Weight Decay (L2 Regularization) to prevent vector magnitude explosion
        # This keeps predictions from saturating at 5.0
        optimizer = optim.Adam([user_vector], lr=0.01, weight_decay=1e-3)
        loss_fn = nn.MSELoss()
        
        steps = 1000
        
        if verbose:
            print(f"\n[Fold-In] Starting Optimization (Steps: {steps})")
            print(f"[Fold-In] Initial Loss check...")
            
        for s in range(steps):
            optimizer.zero_grad()
            
            # Manual Forward Pass reusing Model Layers
            item_emb = self.model.movie_embedding(m_tensor)
            
            # Expand user vector to batch size
            batch_user = user_vector.expand(len(r_tensor), -1)
            
            # Concat
            vector = torch.cat([batch_user, item_emb, f_tensor], dim=-1)
            
            # MLP
            x = self.model.mlp(vector)
            
            # Output
            logits = self.model.output_layer(x).view(-1)
            preds = torch.clamp(logits, min=config.MIN_RATING, max=config.MAX_RATING)
            
            loss = loss_fn(preds, r_tensor)
            loss.backward()
            optimizer.step()
            
            if verbose and (s % 10 == 0 or s == steps-1):
                print(f"  Step {s+1}/{steps} - Loss: {loss.item():.4f}")
        
        if verbose:
            print(f"[Fold-In] Final User Vector (First 5 dims): {user_vector.data[0][:5].cpu().numpy()}")
            print(f"[Fold-In] Vector Norm: {torch.norm(user_vector).item():.4f}\n")
            
        return user_vector.detach()

    # Updated to accept db session to avoid DetachedInstanceError in caller
    def get_recommendations(self, user_id: int, db: Session, limit=20, verbose=False):
        if not self.initialized: self.load_model()
        if not self.initialized: return [], [], [] # Return empty tuple matching signature

        # 1. Fetch User History (Ratings)
        swipes = db.query(Swipe).join(Item).filter(
            Swipe.user_id == user_id,
            Swipe.rating.isnot(None),
            Item.ml_id.isnot(None)
        ).all()
        
        history = [{'ml_id': s.item.ml_id, 'rating': s.rating} for s in swipes]
        
        # ... (rest of logic uses db) ...
        # (Fold-In logic remains same)
        # (Predict logic remains same)
        
        # ... fetch items ...
        
        # ... logic continues ...

        if verbose:
            print(f"\n[NCF] User {user_id}: Found {len(history)} rated items in history.")
        
        # 2. Fold-In (Calculate User Vector)
        user_vector = self.fold_in_user(history, verbose=verbose) # Pass verbose
        if verbose:
             print("[NCF] Fold-In Complete. User Vector Generated.")
        
        # 3. Predict for Candidates
        # Filter watched AND filter by availability
        watched_ids = set([h['ml_id'] for h in history])
        all_model_ids = list(self.movie_mapping.keys())
        
        # Only predict for items that (1) are in DB and (2) user hasn't seen
        candidates = []
        if self.available_item_ids:
            # Intersection is faster
            # Candidates = (Model Keys) INTERSECT (DB IDs) - (Watched)
            possible = set(all_model_ids).intersection(self.available_item_ids)
            candidates = list(possible - watched_ids)
        else:
             # Fallback if DB load failed (risky)
             candidates = [mid for mid in all_model_ids if mid not in watched_ids]

        if not candidates: 
            db.close()
            return []
            
        print(f"[NCF] Predicting for {len(candidates)} candidates.")
        
        # Batch Predict (Chunk if necessary, but 25k is okay)
        c_tensor = torch.tensor([self.movie_mapping[mid] for mid in candidates], dtype=torch.long).to(self.device)
        f_tensor = torch.tensor(self.genome_matrix[c_tensor.cpu().numpy()], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
             item_emb = self.model.movie_embedding(c_tensor)
             batch_user = user_vector.expand(len(c_tensor), -1)
             vector = torch.cat([batch_user, item_emb, f_tensor], dim=-1)
             x = self.model.mlp(vector)
             logits = self.model.output_layer(x).view(-1)
             preds = torch.clamp(logits, min=config.MIN_RATING, max=config.MAX_RATING)
        
        # 4. Rank
        scores = preds.cpu().numpy()
        top_indices = scores.argsort()[::-1][:limit]
        
        # CAST TO INT to avoid np.int64 issues in SQL
        top_ml_ids = [int(candidates[i]) for i in top_indices]
        top_scores = {int(candidates[i]): float(scores[i]) for i in top_indices}
        
        if verbose:
            print(f"[NCF] Top 5 Candidate ML IDs: {top_ml_ids[:5]}")
            
            # Print worst 5 too
            worst_indices = scores.argsort()[:5]
            worst_ml_ids = [int(candidates[i]) for i in worst_indices]
            worst_scores = [float(scores[i]) for i in worst_indices]
            print(f"[NCF] Worst 5 Candidates: {list(zip(worst_ml_ids, worst_scores))}")
        
        # 5. Fetch Items
        top_items = db.query(Item).filter(Item.ml_id.in_(top_ml_ids)).all()
        
        # Sort Top
        result_top = []
        item_map_top = {i.ml_id: i for i in top_items}
        for mid in top_ml_ids:
            if mid in item_map_top:
                it = item_map_top[mid]
                it.score = top_scores[mid]
                result_top.append(it)

        # Fetch Bottom Items
        # Get worst indices
        worst_indices = scores.argsort()[:limit] # Get same amount as limit to be safe
        worst_ml_ids = [int(candidates[i]) for i in worst_indices]
        worst_scores = {int(candidates[i]): float(scores[i]) for i in worst_indices}
        
        bottom_items = db.query(Item).filter(Item.ml_id.in_(worst_ml_ids)).all()
        result_bottom = []
        item_map_bottom = {i.ml_id: i for i in bottom_items}
        for mid in worst_ml_ids:
             if mid in item_map_bottom:
                 it = item_map_bottom[mid]
                 it.score = worst_scores[mid]
                 result_bottom.append(it)

        if verbose:
            print(f"[NCF] Top Recommendations for User {user_id}:")
            for i, item in enumerate(result_top[:5]): # Show top 5
                 print(f"  {i+1}. {item.title} (Score: {item.score:.2f})")
             
        # Return Top, Bottom, and Vector
        vector_list = user_vector.detach().view(-1).cpu().numpy().tolist()
        
        return result_top, result_bottom, vector_list

inference_engine = InferenceService()