import torch
import numpy as np
import os
import pickle

try:
    from . import config, models
except ImportError:
    import config, models

class TwoTowerInference:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Maps
        with open(config.ITEM_MAP_PATH, 'rb') as f:
            data = pickle.load(f)
            self.item_map = data['item_map'] # ml_id -> idx
            self.idx_to_ml_id = data['idx_to_ml_id']
            
        with open(config.USER_MAP_PATH, 'rb') as f:
            self.user_map = pickle.load(f) # user_id -> idx
            
        num_users = len(self.user_map)
        num_items = len(self.item_map)
        
        # Load Model
        self.model = models.TwoTowerModel(
            num_users, num_users, # User Num is used for Embedding
            config.EMBEDDING_DIM, config.LATENT_DIM
        )
        
        if model_path is None:
            model_path = os.path.join(config.DATA_DIR, "best_two_tower.pth")
            
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        else:
            print("Warning: No trained model found at", model_path)
            
        # Pre-compute Item Embeddings for Fast Retrieval
        self.item_embeddings = None
        
    def index_items(self):
        """Compute and cache all item embeddings"""
        if self.item_embeddings is not None:
            return
            
        print("Indexing items...")
        num_items = len(self.item_map)
        all_vecs = []
        batch_size = 2048
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, num_items, batch_size):
                end = min(i + batch_size, num_items)
                indices = torch.arange(i, end, device=self.device)
                vecs = self.model.item_tower(indices)
                all_vecs.append(vecs)
                
        self.item_embeddings = torch.cat(all_vecs, dim=0) # [NumItems, Dim]
        print(f"Indexed {num_items} items.")
        
    def get_recommendations(self, user_id, history_items, history_weights=None, k=10):
        """
        user_id: Raw UserID
        history_items: List of Raw ItemIDs (MovieLens IDs)
        history_weights: List of weights (optional, default 1.0)
        """
        if self.item_embeddings is None:
            self.index_items()
            
        # Map User
        u_idx = self.user_map.get(user_id, 0) # Default to 0 if new? Or handle Cold Start
        
        # Map History
        h_indices = []
        h_weights = []
        for i, item in enumerate(history_items):
            if item in self.item_map:
                h_indices.append(self.item_map[item])
                w = history_weights[i] if history_weights else 1.0
                h_weights.append(w)
                
        if not h_indices:
            # Cold start logic?
            # Return popular items?
            # For now, just use empty history
            h_indices = [0]
            h_weights = [0.0]
            
        # To Tensor
        u_idx_t = torch.tensor([u_idx], device=self.device)
        h_idx_t = torch.tensor([h_indices], device=self.device) # [1, Seq]
        h_w_t = torch.tensor([h_weights], device=self.device, dtype=torch.float)
        h_mask_t = torch.ones_like(h_idx_t, dtype=torch.float)
        
        with torch.no_grad():
            user_vec = self.model.user_tower(u_idx_t, h_idx_t, h_w_t, h_mask_t)
            
            # Dot Product
            scores = torch.matmul(user_vec, self.item_embeddings.T) # [1, NumItems]
            scores = scores.squeeze(0)
            
            # Top K
            top_scores, top_indices = torch.topk(scores, k)
            
        # Map back to IDs
        recs = []
        for idx in top_indices.cpu().numpy():
            recs.append(self.idx_to_ml_id[idx])
            
        return recs
