import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from data_loader import MovieLensDataLoader
from models import FastALSModel, NCFModel

def train_ncf(model, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    model.to(config.DEVICE)
    
    print(f"\n--- NCF Modeli ({config.DEVICE}) üzerinde eğitiliyor ---")
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for user_idx, movie_idx, rating in train_loader:
            user_idx, movie_idx, rating = user_idx.to(config.DEVICE), movie_idx.to(config.DEVICE), rating.to(config.DEVICE)
            
            optimizer.zero_grad()
            prediction = model(user_idx, movie_idx)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{config.EPOCHS} - Loss: {total_loss / len(train_loader):.4f}")
        
    return model

def find_similar_users(model, target_user_idx, k=5):
    """
    NCF modelinin öğrendiği embedding uzayında en yakın userları bulur.
    """
    model.eval()
    # Tüm user embeddinglerini CPU'ya al
    all_users = model.user_embedding.weight.data.cpu().numpy()
    target_vector = all_users[target_user_idx].reshape(1, -1)
    
    # Cosine Similarity
    sim_scores = cosine_similarity(target_vector, all_users)[0]
    
    # Kendisi hariç en yüksek skorlu k kişiyi bul
    similar_indices = sim_scores.argsort()[::-1][1:k+1]
    return similar_indices

if __name__ == "__main__":
    # 1. Veriyi Yükle
    loader = MovieLensDataLoader()
    df = loader.load_data()
    
    # 2. Hızlı Model (ALS) Eğitimi
    sparse_matrix = loader.get_sparse_matrix()
    als_model = FastALSModel(factors=config.EMBEDDING_DIM)
    als_model.fit(sparse_matrix)
    
    # ALS Test
    test_user_idx = 100 # Örnek bir user index
    ids, scores = als_model.recommend(test_user_idx, sparse_matrix)
    print(f"\n[Hızlı Model] User idx {test_user_idx} için önerilen film indexleri: {ids}")
    
    # 3. Kapsamlı Model (NCF) Eğitimi
    train_loader, val_loader = loader.get_pytorch_dataloader()
    ncf_model = NCFModel(loader.num_users, loader.num_movies, embedding_dim=config.EMBEDDING_DIM)
    ncf_model = train_ncf(ncf_model, train_loader, val_loader)
    
    # 4. User Similarity (Taste Matching) Testi
    similar_users = find_similar_users(ncf_model, target_user_idx=test_user_idx)
    
    print(f"\n[Kapsamlı Model] User idx {test_user_idx} ile en benzer zevke sahip User Indexleri:")
    print(similar_users)
    
    # Gerçek User ID'lerini görmek istersek:
    real_user_id = loader.idx2user[test_user_idx]
    similar_real_ids = [loader.idx2user[idx] for idx in similar_users]
    print(f"(Gerçek ID'ler) User {real_user_id} -> Benzerleri: {similar_real_ids}")
    
    # 5. Modelleri Kaydet
    torch.save(ncf_model.state_dict(), os.path.join(config.MODELS_DIR, "ncf_model.pth"))
    print("\nModeller kaydedildi.")