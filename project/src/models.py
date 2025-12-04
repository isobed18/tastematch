import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
import numpy as np

# --- 1. HIZLI MODEL (SVD - Scikit-Learn) ---
class FastSVDModel:
    def __init__(self, n_components=64):
        # TruncatedSVD, sparse matrix üzerinde çalışabilen çok hızlı bir yöntemdir.
        # Netflix Prize yarışmasında popülerleşmiştir.
        self.model = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
        self.user_vecs = None
        self.item_vecs = None
        
    def fit(self, sparse_user_item):
        print("Hızlı Model (SVD) eğitiliyor...")
        # SVD bize User matrisini verir
        self.user_vecs = self.model.fit_transform(sparse_user_item)
        # Components_ ise Item (Film) matrisidir
        self.item_vecs = self.model.components_.T
        print("SVD Eğitimi bitti.")
        
    def predict(self, user_idx, movie_idx):
        # Basit Dot Product: User Vektörü . Film Vektörü
        u = self.user_vecs[user_idx]
        m = self.item_vecs[movie_idx]
        return np.dot(u, m)

# --- 2. KAPSAMLI MODEL (NCF - PyTorch) ---
class NCFModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super(NCFModel, self).__init__()
        
        # User ve Movie Embeddingleri
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # MLP Katmanları
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, user_idx, movie_idx):
        user_embed = self.user_embedding(user_idx)
        movie_embed = self.movie_embedding(movie_idx)
        
        # Vektörleri birleştir
        x = torch.cat([user_embed, movie_embed], dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        
        out = self.output(x)
        return out.squeeze()