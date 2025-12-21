import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import Modules
try:
    from project.two_tower.inference_utils import TwoTowerInference
    from project.two_tower.ranking_models import RankerModel
    from project.two_tower import config as tt_config
except ImportError as e:
    print(f"Import Error: {e}")
    raise e

class InferenceService:

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inference Service Device: {self.device}")
        
        # Define Production Models Directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.prod_models_dir = os.path.join(base_dir, "production_models")
        
        # --- 1. Load Two-Tower (Retrieval) ---
        print("Loading Two-Tower Retrieval Engine...")
        
        retrieval_path = os.path.join(self.prod_models_dir, "two_tower_retrieval_v1.pth")
        if not os.path.exists(retrieval_path):
             print(f"WARNING: Production model not found at {retrieval_path}. Falling back to default.")
             retrieval_path = None # Let utils decide or fail
             
        self.retrieval_engine = TwoTowerInference(model_path=retrieval_path) 
        
        # --- 2. Load Two-Tower Ranker ---
        print("Loading Two-Tower Ranker Engine...")
        self.ranker_model = self._load_ranker()
        
        print("Inference Service Initialized Successfully.")


    def _load_ranker(self):
        """
        Loads the pairwise RankerModel.
        """
        try:
            ranker_path = os.path.join(self.prod_models_dir, "two_tower_ranker_v1.pth")
            
            if not os.path.exists(ranker_path):
                print(f"WARNING: Ranker model not found at {ranker_path}. Ranking will be disabled (Cold Start friendly though!).")
                return None
                
            print(f"Loading Ranker from: {ranker_path}")
            
            # Initialize Model
            # Latent Dim matches Two Tower (1024 or 512?)
            # Retrieval Engine uses 512 internally (latent_dim=512 in inference_utils).
            # So Ranker must match.
            model = RankerModel(embedding_dim=512) 
            
            model.load_state_dict(torch.load(ranker_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"ERROR loading Ranker: {e}")
            return None

    def get_recommendations(self, user_id, history_items, k_final=10):
        """
        API Entry Point.
        Returns: List of Item IDs (int)
        """
        
        # 1. RETRIEVAL (Two-Tower)
        # ------------------------
        # Get Candidate IDs AND User Vector
        # user_vec is numpy array [1, 512]
        candidates, user_vec = self.retrieval_engine.get_recommendations(
            user_id=user_id,
            history_items=history_items,
            k=100 
        )
        
        if not candidates:
            return []

        # 2. RANKING (Feature Based)
        # ----------------
        # Now we use the User Vector (which exists even for new users!) to rank.
        
        if self.ranker_model is not None:
             # Returns list of (id, score)
            ranked_candidates = self._rank_candidates(user_vec, candidates)
            print(f"[Inference] Ranked {len(candidates)} candidates. Top 3: {ranked_candidates[:3]}")
            return ranked_candidates[:k_final]
        else:
            # Fallback if ranker missing
            print(f"[Inference] Ranker missing. Returning Retrieval results.")
            return [(cand, 0.0) for cand in candidates[:k_final]]

    def get_user_embedding(self, user_id, history_items):
        """
        Returns the 512-dim User Vector as a Python List.
        """
        # Call Retrieval Engine
        # user_vec is Tensor on Device [1, 512]
        user_vec_t = self.retrieval_engine.get_user_vector(user_id, history_items)
        
        # Convert to list
        user_vec_np = user_vec_t.cpu().numpy().flatten()
        return user_vec_np.tolist()

    def _rank_candidates(self, user_vec_np, candidate_ids):
        """
        Ranks candidates using UserVector + ItemVector.
        """
        # 1. Get Item Vectors from Retrieval Engine's Cache
        # Retrieval engine has 'item_embeddings' (Tensor on Device) and 'item_map' (ID -> Index)
        
        # Ensure item embeddings are ready
        if self.retrieval_engine.item_embeddings is None:
            self.retrieval_engine.index_items()
            
        valid_candidates = [] # (RawID, ItemIndex)
        
        for mid in candidate_ids:
            if mid in self.retrieval_engine.item_map:
                idx = self.retrieval_engine.item_map[mid]
                valid_candidates.append((mid, idx))
                
        if not valid_candidates:
             return [(c, 0.0) for c in candidate_ids]
             
        # 2. Batch Creation
        n = len(valid_candidates)
        
        # User Vector: [1, 512] -> [N, 512]
        u_vec_t = torch.tensor(user_vec_np, device=self.device) # [1, 512]
        if u_vec_t.dim() == 1: u_vec_t = u_vec_t.unsqueeze(0)
        u_batch = u_vec_t.repeat(n, 1) # [N, 512]
        
        # Item Vectors: Gather from cache
        item_indices = [x[1] for x in valid_candidates]
        i_indices_t = torch.tensor(item_indices, device=self.device, dtype=torch.long)
        i_batch = self.retrieval_engine.item_embeddings[i_indices_t] # [N, 512]
        
        # 3. Predict
        with torch.no_grad():
            scores = self.ranker_model(u_batch, i_batch) # [N]
            scores = scores.cpu().numpy()
            
        # 4. Sort
        scored_list = []
        for i in range(n):
            raw_id = valid_candidates[i][0]
            score = float(scores[i])
            scored_list.append((raw_id, score))
            
        scored_list.sort(key=lambda x: x[1], reverse=True)
        
        return scored_list

    def recommend_for_vector(self, vector_np, k=10):
        """
        Returns items closest to the given vector.
        vector_np: List or Numpy array of shape [Dim]
        """
        engine = self.retrieval_engine
        if engine.item_embeddings is None:
            engine.index_items()
            
        # Tensorize
        vec_t = torch.tensor(vector_np, device=self.device, dtype=torch.float)
        if vec_t.dim() == 1: vec_t = vec_t.unsqueeze(0) # [1, Dim]
        
        with torch.no_grad():
             scores = torch.matmul(vec_t, engine.item_embeddings.T)
             scores = scores.squeeze(0)
             top_scores, top_indices = torch.topk(scores, k)
             
        recs = []
        for idx in top_indices.cpu().numpy():
            recs.append(engine.idx_to_ml_id[idx])
            
        return recs
