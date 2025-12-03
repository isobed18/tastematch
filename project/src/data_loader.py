import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import config

# --- DÜZELTME: Sınıfı en dışa aldık (Windows Multiprocessing hatası için) ---
class RatingDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.movies = torch.tensor(movies, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

class MovieLensDataLoader:
    def __init__(self):
        self.df = None
        self.user2idx = {}
        self.movie2idx = {}
        self.idx2user = {}
        self.idx2movie = {}
        self.num_users = 0
        self.num_movies = 0
        
    def load_data(self):
        print(f"Veri yükleniyor: {config.RATINGS_PATH}...")
        # RAM tasarrufu için dtype
        self.df = pd.read_csv(config.RATINGS_PATH, 
                              usecols=['userId', 'movieId', 'rating'],
                              dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        
        print(f"Toplam {len(self.df)} satır veri yüklendi.")
        
        # Mapping
        user_ids = self.df['userId'].unique()
        movie_ids = self.df['movieId'].unique()
        
        self.num_users = len(user_ids)
        self.num_movies = len(movie_ids)
        
        self.user2idx = {id: i for i, id in enumerate(user_ids)}
        self.movie2idx = {id: i for i, id in enumerate(movie_ids)}
        
        self.idx2user = {i: id for i, id in enumerate(user_ids)}
        self.idx2movie = {i: id for i, id in enumerate(movie_ids)}
        
        self.df['user_idx'] = self.df['userId'].map(self.user2idx)
        self.df['movie_idx'] = self.df['movieId'].map(self.movie2idx)
        
        print(f"Mapping Tamamlandı. Users: {self.num_users}, Movies: {self.num_movies}")
        return self.df

    def get_sparse_matrix(self):
        row = self.df['user_idx'].values
        col = self.df['movie_idx'].values
        data = self.df['rating'].values
        return sparse.csr_matrix((data, (row, col)), shape=(self.num_users, self.num_movies))

    def get_pytorch_dataloader(self):
        train_df, val_df = train_test_split(self.df, test_size=0.1, random_state=42)
        
        train_ds = RatingDataset(train_df['user_idx'].values, train_df['movie_idx'].values, train_df['rating'].values)
        val_ds = RatingDataset(val_df['user_idx'].values, val_df['movie_idx'].values, val_df['rating'].values)
        
        # Windows'ta bazen num_workers > 0 sorun yaratabilir, hata devam ederse burayı 0 yap.
        # Şimdilik 4 olarak bırakıyoruz, yukarıdaki fix yetmeli.
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
        
        return train_loader, val_loader