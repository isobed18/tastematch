import os
import torch
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config
from models import NCFModel, FastSVDModel

class RecommenderSystem:
    def __init__(self):
        self.mappings = None
        self.fast_model = None
        self.deep_model = None
        self.device = config.DEVICE
        self.load_artifacts()

    def load_artifacts(self):
        print("Modeller ve Mappingler yükleniyor...")
        
        # Mappingleri yükle
        with open(os.path.join(config.MODELS_DIR, "mappings.pkl"), 'rb') as f:
            self.mappings = pickle.load(f)
            
        # Hızlı Modeli yükle
        with open(os.path.join(config.MODELS_DIR, "fast_svd_model.pkl"), 'rb') as f:
            self.fast_model = pickle.load(f)
            
        # Derin Modeli Yükle (Sadece ağırlıklar)
        num_users = len(self.mappings['user2idx'])
        num_movies = len(self.mappings['movie2idx'])
        
        self.deep_model = NCFModel(num_users, num_movies, embedding_dim=config.EMBEDDING_DIM)
        self.deep_model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, "ncf_model.pth"), map_location=self.device))
        self.deep_model.to(self.device)
        self.deep_model.eval()

    def find_similar_users(self, user_id, k=5):
        # Deep modelin embeddinglerini kullanır
        if user_id not in self.mappings['user2idx']:
            return "User bulunamadı"
            
        idx = self.mappings['user2idx'][user_id]
        
        # Embeddingleri al
        all_users = self.deep_model.user_embedding.weight.data.cpu().numpy()
        target_vec = all_users[idx].reshape(1, -1)
        
        # Benzerlik
        sim = cosine_similarity(target_vec, all_users)[0]
        sim_indices = sim.argsort()[::-1][1:k+1]
        
        similar_ids = [self.mappings['idx2user'][i] for i in sim_indices]
        return similar_ids

if __name__ == "__main__":
    # Test
    rec_sys = RecommenderSystem()
    
    # Rastgele bir user ID seç (Veri setindeki gerçek ID'lerden biri olmalı)
    # Mapping dosyasından ilk user'ı alalım örnek olarak
    test_user = list(rec_sys.mappings['user2idx'].keys())[0]
    
    print(f"\nUser ID: {test_user} için analiz:")
    
    # Benzer Userlar
    similars = rec_sys.find_similar_users(test_user)
    print(f"En Benzer 5 User: {similars}")