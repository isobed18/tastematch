import torch
import numpy as np
import pandas as pd
import os
import pickle
import sys


# Proje kök dizinini yola ekle
# backend/app/inference_service.py -> backend/app -> backend -> tastematch (root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Modülleri import et
try:
    from project.two_tower.inference_utils import TwoTowerInference
    from project.ncf.model import HybridNCF
    from project.ncf import config as ncf_config
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback or re-raise
    raise e

class InferenceService:

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inference Service Device: {self.device}")
        
        # --- 1. Load Two-Tower (Retrieval) ---
        print("Loading Two-Tower Retrieval Engine...")
        # Load specific model if provided, else default
        self.retrieval_engine = TwoTowerInference(model_path=model_path) 
        
        # --- 2. Load NCF (Ranker) ---
        print("Loading NCF Ranking Engine...")
        self.ranker_model, self.ncf_user_map, self.ncf_movie_map = self._load_ncf()
        self.genome_matrix = self._load_genome()
        
        print("Inference Service Initialized Successfully.")


    def _load_ncf(self):
        """
        En son eğitilen NCF modelini ve maplerini yükler.
        """
        # runs_ncf klasöründeki en son run'ı bul
        # base_dir is root (tastematch/)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        run_root = os.path.join(base_dir, "runs_ncf") # DIRECTLY in root
        
        if not os.path.exists(run_root):
            print(f"WARNING: NCF runs directory not found at {run_root}. Ranking will be disabled.")
            return None, {}, {}

        runs = sorted([d for d in os.listdir(run_root) if d.startswith("run_")])
        if not runs:
            print("WARNING: No NCF training runs found. Ranking will be disabled.")
            return None, {}, {}
            
        latest_run = os.path.join(run_root, runs[-1])
        print(f"Using NCF Model from: {latest_run}")
        
        # Mappings Yükle
        try:
            with open(os.path.join(latest_run, 'user_mapping.pkl'), 'rb') as f:
                user_map = pickle.load(f)
            with open(os.path.join(latest_run, 'movie_mapping.pkl'), 'rb') as f:
                movie_map = pickle.load(f)
                
            # Model Başlat
            model = HybridNCF(
                num_users=len(user_map),
                num_movies=len(movie_map),
                genome_dim=ncf_config.GENOME_DIM,
                embedding_dim=ncf_config.EMBEDDING_DIM
            )
            
            # Ağırlıkları Yükle
            model_path = os.path.join(latest_run, 'best_model.pth')
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            return model, user_map, movie_map
            
        except Exception as e:
            print(f"ERROR loading NCF: {e}")
            return None, {}, {}

    def _load_genome(self):
        # Genome matrix yükle
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        run_root = os.path.join(base_dir, "runs", "ncf") # Changed from "runs_ncf" to "runs/ncf"
        
        if not os.path.exists(run_root): return None
        runs = sorted([d for d in os.listdir(run_root) if d.startswith("run_")])
        if not runs: return None
        
        latest_run = os.path.join(run_root, runs[-1])
        path = os.path.join(latest_run, 'genome_matrix.npy')
        
        if os.path.exists(path):
            return np.load(path)
        return None

    def get_recommendations(self, user_id, history_items, k_final=10):
        """
        API'nin çağıracağı ana fonksiyon.
        user_id: Veritabanındaki gerçek User ID (int)
        history_items: Kullanıcının beğendiği/izlediği filmlerin ID listesi [int]
        """
        
        # 1. RETRIEVAL (Two-Tower)
        # ------------------------
        # Two-Tower, kullanıcının ID'si yoksa bile 'history_items' kullanarak
        # içerik tabanlı (Content-Based) bir vektör oluşturur ve aday getirir.
        
        # Daha fazla aday çağırıyoruz ki Ranker'a eleme şansı kalsın (örn. 100)
        candidates = self.retrieval_engine.get_recommendations(
            user_id=user_id,
            history_items=history_items,
            k=100 
        )
        
        # Eğer Two-Tower hiç aday bulamazsa (çok nadir), boş dön.
        if not candidates:
            return []


        # 2. RANKING (NCF)
        # ----------------
        
        is_known_user = (self.ranker_model is not None) and (user_id in self.ncf_user_map)
        print(f"[Inference] User {user_id} Known to Ranker? {is_known_user}")
        
        if is_known_user:
            # Kullanıcı Biliniyor -> Hassas Sıralama Yap (Re-Ranking)
            # Returns list of (id, score)
            ranked_candidates = self._rank_with_ncf(user_id, candidates)
            print(f"[Inference] Ranked {len(candidates)} candidates. Top 3: {ranked_candidates[:3]}")
            return ranked_candidates[:k_final]
        else:
            # Kullanıcı Yeni -> Two-Tower sonuçlarını olduğu gibi döndür
            print(f"[Inference] Cold Start / Unknown User. Returning Retrieval results.")
            return [(cand, 0.0) for cand in candidates[:k_final]]

    def _rank_with_ncf(self, user_id, candidates):
        """
        NCF modelini kullanarak adaylara 0.5 - 5.0 arası puan verir.
        """
        ncf_user_idx = self.ncf_user_map[user_id]
        
        # Adayları NCF ID'lerine çevir
        valid_candidates = [] # (RawID, NCF_ID)
        
        for mid in candidates:
            if mid in self.ncf_movie_map:
                valid_candidates.append((mid, self.ncf_movie_map[mid]))
        
        print(f"[Ranker] {len(valid_candidates)}/{len(candidates)} candidates found in NCF map.")
        
        if not valid_candidates:
            return [(c, 0.0) for c in candidates] # Eşleşme yoksa orijinalleri dön
            
        # Batch Hazırlığı
        n = len(valid_candidates)
        u_tensor = torch.tensor([ncf_user_idx] * n, dtype=torch.long).to(self.device)
        m_tensor = torch.tensor([x[1] for x in valid_candidates], dtype=torch.long).to(self.device)
        
        # Genome Features
        if self.genome_matrix is not None:
            # Numpy -> Tensor
            features = torch.tensor(self.genome_matrix[m_tensor.cpu().numpy()], dtype=torch.float32).to(self.device)
        else:
            # Fallback if genome missing (should not happen in prod)
            features = torch.zeros((n, ncf_config.GENOME_DIM), dtype=torch.float32).to(self.device)
            
        # Tahmin
        with torch.no_grad():
            scores = self.ranker_model(u_tensor, m_tensor, features)
            scores = scores.cpu().numpy()
            
        # Skorlara göre sırala
        scored_list = []
        for i in range(n):
            raw_id = valid_candidates[i][0]
            score = float(scores[i]) # Convert to python float
            scored_list.append((raw_id, score))
            
        # Puanı yüksek olan en başa
        scored_list.sort(key=lambda x: x[1], reverse=True)
        
        return scored_list
