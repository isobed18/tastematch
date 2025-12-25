<<<<<<< HEAD
import os
import pickle
import numpy as np
from .database import SessionLocal
from .models import Item, Swipe, SwipeAction

# Model dosya yollarını ayarla (Project klasörüne gidip alacak)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "project", "models", "fast_svd_model.pkl")
MAPPING_PATH = os.path.join(BASE_DIR, "project", "models", "mappings.pkl")

# Pickle 'models' modülünü arayacağı için path'e ekle
import sys
sys.path.append(os.path.join(BASE_DIR, "project", "src"))

class InferenceService:
    def __init__(self):
        self.model = None
        self.mappings = None
        self.item_factors = None
        self.initialized = False
        
    def load_model(self):
        if self.initialized: return
        print("Model yükleniyor...")
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
                # Ensure item_factors is (N_items, 64)
                # If shape is (64, N), transpose it. If (N, 64), keep it.
                if self.model.item_vecs.shape[0] < self.model.item_vecs.shape[1]:
                     self.item_factors = self.model.item_vecs.T
                else:
                     self.item_factors = self.model.item_vecs
            
            with open(MAPPING_PATH, 'rb') as f:
                self.mappings = pickle.load(f)
                
            self.initialized = True
            print(f"Model Hazır. Item Factors Shape: {self.item_factors.shape}")
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")

    def get_recommendations(self, user_id: int, limit=20):
        if not self.initialized: self.load_model()
        
        # Eğer model yüklenemediyse boş liste dön (500 hatası verme)
        if not self.initialized or self.mappings is None:
            print("Model yüklenemediği için öneri yapılamıyor.")
            return []

        db = SessionLocal()
        
        # 1. Kullanıcının beğendiği (like/superlike) filmlerin ml_id'lerini çek
        liked_rows = db.query(Item.ml_id).join(Swipe).filter(
            Swipe.user_id == user_id,
            Swipe.action.in_([SwipeAction.like, SwipeAction.superlike]),
            Item.ml_id.isnot(None)
        ).all()
        
        liked_ml_ids = [r[0] for r in liked_rows]
        
        # 2. Kullanıcı Vektörü Oluştur
        user_vector = np.zeros(64) # 64 boyutlu eğitmiştik
        count = 0
        
        for mid in liked_ml_ids:
            # Modelin tanıdığı ID'ye (index) çevir
            if mid in self.mappings['movie2idx']:
                idx = self.mappings['movie2idx'][mid]
                # Item vektörünü ekle
                user_vector += self.item_factors[idx]
                count += 1
                
        if count > 0:
            user_vector /= count # Ortalama al
        else:
            # Soğuk başlangıç: Hiçbir şey beğenmediyse popülerleri döndür
            # (Bunu basitçe boş liste dönerek halledelim, router popülerleri doldursun)
            db.close()
            return []

        # 3. Skor Hesapla (Dot Product)
        # User (64,) . Items (N, 64).T -> (N,)
        # OR np.dot(Items, User) -> (N, 64) . (64,) -> (N,)
        scores = np.dot(self.item_factors, user_vector)
        
        # 4. En yüksek skorlu indexleri bul
        top_indices = scores.argsort()[::-1][:limit*3] # Filtrelemek için fazla al
        
        recommended_ml_ids = []
        for idx in top_indices:
            real_ml_id = self.mappings['idx2movie'][idx]
            
            # Zaten izlediklerini önerme
            if real_ml_id not in liked_ml_ids:
                recommended_ml_ids.append(real_ml_id)
                if len(recommended_ml_ids) >= limit:
                    break
        
        # 5. DB'den Itemları Çek (ml_id'ye göre)
        items = db.query(Item).filter(Item.ml_id.in_(recommended_ml_ids)).all()
        
        # Sıralamayı koru (SQL 'IN' sorgusu sıralamayı bozar)
        item_map = {item.ml_id: item for item in items}
        sorted_items = []
        
        # Score map oluştur (ml_id -> score)
        score_map = {}
        for idx in top_indices:
             real_ml_id = self.mappings['idx2movie'][idx]
             score_map[real_ml_id] = float(scores[idx]) # float'a çevir

        for mid in recommended_ml_ids:
            if mid in item_map:
                item = item_map[mid]
                # Skoru item üzerine ekle (geçici olarak)
                item.score = score_map.get(mid, 0.0)
                sorted_items.append(item)
                
        db.close()
        return sorted_items

inference_engine = InferenceService()
=======
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
>>>>>>> feature/multi-domain-architecture
