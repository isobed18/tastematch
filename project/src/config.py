import os
import torch

# Dizin Ayarları (project/src içinden çalıştırılacağı varsayımıyla)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'ml-latest')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Dosya Yolları
RATINGS_PATH = os.path.join(DATA_DIR, 'ratings.csv')
MOVIES_PATH = os.path.join(DATA_DIR, 'movies.csv')

# Model Ayarları
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 64  # Hem ALS hem NCF için vektör boyutu
BATCH_SIZE = 10240  # GPU belleğine göre artırılabilir (33M veri için yüksek tutuyoruz)
EPOCHS = 5
LR = 0.001

# Çıktı klasörü yoksa oluştur
os.makedirs(MODELS_DIR, exist_ok=True)