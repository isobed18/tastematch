import os
import pickle
import numpy as np
import pandas as pd
import config

def test_fast_model():
    print("--- Hızlı Model (SVD) Testi Başlıyor ---")
    
    # 1. Dosyaları Yükle
    model_path = os.path.join(config.MODELS_DIR, "fast_svd_model.pkl")
    map_path = os.path.join(config.MODELS_DIR, "mappings.pkl")
    movies_path = config.MOVIES_PATH
    
    if not os.path.exists(model_path):
        print("HATA: Model dosyası bulunamadı. Önce 'python src/train_fast.py' çalıştırın.")
        return

    print("Model ve Mappingler yükleniyor...")
    with open(model_path, 'rb') as f:
        svd_model = pickle.load(f)
        
    with open(map_path, 'rb') as f:
        mappings = pickle.load(f)
        
    # Film İsimlerini Yükle (Sonuçları okuyabilmek için)
    movies_df = pd.read_csv(movies_path)
    # Kolay erişim için movieId -> Title sözlüğü
    movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

    # 2. Rastgele Bir User Seç
    # Mapping dosyasındaki userlardan rastgele biri
    import random
    random_user_idx = random.choice(list(mappings['idx2user'].keys()))
    real_user_id = mappings['idx2user'][random_user_idx]
    
    print(f"\nTEST EDİLEN KULLANICI: ID {real_user_id} (Index: {random_user_idx})")
    
    # 3. Tüm filmler için skor üret (Dot Product)
    # User Vektörü (1, 64) * Item Matrisi (64, Movie_Count) = (1, Movie_Count)
    user_vector = svd_model.user_vecs[random_user_idx]
    # .T ekleyerek matrisi (64, 83239) haline getiriyoruz
    scores = np.dot(user_vector, svd_model.item_vecs.T)
    
    # 4. En yüksek skorlu 10 filmi bul
    # argsort küçükten büyüğe sıralar, ters çevirip ilk 10'u alıyoruz
    top_10_indices = scores.argsort()[::-1][:10]
    
    print("\n--- ÖNERİLEN TOP 10 FİLM ---")
    for i, movie_idx in enumerate(top_10_indices):
        real_movie_id = mappings['idx2movie'][movie_idx]
        title = movie_id_to_title.get(real_movie_id, "Bilinmeyen Film")
        score = scores[movie_idx]
        print(f"{i+1}. [{score:.2f}] {title}")

if __name__ == "__main__":
    test_fast_model()