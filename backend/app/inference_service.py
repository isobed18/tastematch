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
    import chromadb
except ImportError as e:
    print(f"Import Error: {e}")
    # Don't raise, allow partial initialization
    pass

class InferenceService:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inference Service Device: {self.device}")

        # Define Production Models Directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.prod_models_dir = os.path.join(base_dir, "production_models")
        
        # --- 1. Load Domain-Specific Engines ---
        print("Loading Domain Engines...")
        
        # A. MOVIE (Two-Tower) - Existing
        retrieval_path = os.path.join(self.prod_models_dir, "two_tower_retrieval_v1.pth")
        if not os.path.exists(retrieval_path):
             print(f"WARNING: Production model not found at {retrieval_path}. Falling back to default.")
             retrieval_path = None
             
        try:
             self.movie_engine = TwoTowerInference(model_path=retrieval_path) 
             self.latent_dim = 512 # Standard
        except Exception as e:
             print(f"Failed to load Movie Engine: {e}")
             self.movie_engine = None

        # B. OTHER DOMAINS (ChromaDB / ANN)
        # We use a single ChromaDB instance for Books, Music, etc.
        chroma_path = os.path.join(base_dir, "backend", "chroma_db") 
        # Note: If running inside Docker, path might differ.
        if not os.path.exists(chroma_path):
             os.makedirs(chroma_path, exist_ok=True)
             
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            # Create/Get collections
            self.collections = {
                "book": self.chroma_client.get_or_create_collection("book_embeddings"),
                "music": self.chroma_client.get_or_create_collection("music_embeddings"),
                "food": self.chroma_client.get_or_create_collection("food_embeddings"),
                # Movies theoretically could be here too, but we use TwoTowerInference for now
            }
        except Exception as e:
            print(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collections = {}

        # --- 2. Load Unified Ranker ---
        print("Loading Ranker Engine...")
        self.ranker_model = self._load_ranker()
        
        print("Multi-Domain Inference Service Initialized.")

    def mix_tastes(self, user_taste_vectors: dict) -> np.ndarray:
        """
        Mixes domain-specific taste vectors into a single Composite Soul Vector.
        Strategy: Weighted Fusion (Normalize(Sum(Weight * Vector)))
        """
        # Default Weights
        weights = {
            "movie": 1.0,
            "book": 1.0,
            "music": 0.8,
            "food": 0.5
        }
        
        # Initialize Composite
        composite = np.zeros(self.latent_dim, dtype=np.float32)
        total_weight = 0.0
        
        for domain, vector in user_taste_vectors.items():
            if not vector or len(vector) != self.latent_dim:
                continue
                
            w = weights.get(domain, 0.5)
            
            # --- Advanced: Adjust weight by Recency/Confidence if available ---
            # (Assuming vector has metadata or passed separately, for now simple dict)
            
            vec_np = np.array(vector, dtype=np.float32)
            composite += vec_np * w
            total_weight += w
            
        if total_weight > 0:
            composite /= total_weight
            
        # L2 Normalize
        norm = np.linalg.norm(composite)
        if norm > 0:
            composite /= norm
            
        return composite

    def get_user_embedding(self, user_id, history_items):
        """
        Returns the 512-dim User Vector as a Python List.
        """
        # Call Retrieval Engine
        if self.movie_engine:
            user_vec_t = self.movie_engine.get_user_vector(user_id, history_items)
            user_vec_np = user_vec_t.cpu().numpy().flatten()
            return user_vec_np.tolist()
        return []

    def _rank_candidates(self, user_vec_np, candidate_ids):
        """
        Ranks candidates using UserVector + ItemVector.
        """
        if not self.movie_engine:
            return [(c, 0.0) for c in candidate_ids]

        # 1. Get Item Vectors from Movie Engine's Cache
        if self.movie_engine.item_embeddings is None:
            self.movie_engine.index_items()
            
        valid_candidates = [] # (RawID, ItemIndex)
        
        for mid in candidate_ids:
            if mid in self.movie_engine.item_map:
                idx = self.movie_engine.item_map[mid]
                valid_candidates.append((mid, idx))
                
        if not valid_candidates:
             return [(c, 0.0) for c in candidate_ids]
             
        # 2. Batch Creation
        n = len(valid_candidates)
        
        u_vec_t = torch.tensor(user_vec_np, device=self.device) # [1, 512]
        if u_vec_t.dim() == 1: u_vec_t = u_vec_t.unsqueeze(0)
        u_batch = u_vec_t.repeat(n, 1) # [N, 512]
        
        item_indices = [x[1] for x in valid_candidates]
        i_indices_t = torch.tensor(item_indices, device=self.device, dtype=torch.long)
        i_batch = self.movie_engine.item_embeddings[i_indices_t] # [N, 512]
        
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
        """
        if not self.movie_engine: return []
        
        engine = self.movie_engine
        if engine.item_embeddings is None:
            engine.index_items()
            
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

    def get_recommendations(self, user_id, history_items=None, domain="movie", k_final=10, user_obj=None):
        """
        Unified Entry Point.
        Returns: List of tuples (item_id, score)
        """
        user_vec = None
        
        # 1. Resolve User Vector
        if user_obj and getattr(user_obj, 'taste_vectors', None):
             # Just use the composite or mix on fly
             user_vec = self.mix_tastes(user_obj.taste_vectors) # returns np.array
        
        # Scenario B: Domain specific fallback (Legacy Movie Logic)
        if domain == "movie" and self.movie_engine:
             if history_items:
                  cands, u_vec = self.movie_engine.get_recommendations(user_id, history_items, k=100)
                  
                  # If we didn't have a composite vector, use this one
                  if user_vec is None:
                      user_vec = u_vec.flatten()
                  
                  # RANK using Feature Ranker
                  if self.ranker_model:
                       return self._rank_candidates(user_vec, cands)[:k_final]
                  else:
                       return [(c, 0.0) for c in cands[:k_final]]
                  
        # 2. Universal Retrieval using User Vector (ANN)
        if user_vec is None:
             # No history, no User Obj? 
             # For Movie domain if TwoTower failed, fallback to empty (Feed router handles popularity)
             return []
             
        # Retrieve from Chroma
        if domain in self.collections:
             collection = self.collections[domain]
             try:
                 results = collection.query(
                     query_embeddings=[user_vec.tolist()],
                     n_results=k_final
                 )
                 # Extract IDs
                 if results['ids']:
                     ids = results['ids'][0]
                     dists = results['distances'][0] if 'distances' in results and results['distances'] else [0.0]*len(ids)
                     
                     scored = []
                     for i, mid in enumerate(ids):
                         # Distance usually L2. Score = ?
                         # If Cosine distance: 0 is match, 1 is opposite.
                         d = dists[i]
                         score = 10.0 * (1.0 - d) # Approximation
                         scored.append((mid, score))
                     return scored
             except Exception as e:
                 print(f"[Inference] Chroma Query Error: {e}")
                 
        return []
