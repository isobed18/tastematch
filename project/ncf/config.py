import torch
import os

# --- PATHS ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ml-latest')
RATINGS_PATH = os.path.join(DATA_DIR, 'ratings.csv')
MOVIES_PATH = os.path.join(DATA_DIR, 'movies.csv')
GENOME_SCORES_PATH = os.path.join(DATA_DIR, 'genome-scores.csv')


# Paths
DATA_PATH = "data/ml-latest"
MODEL_DIR = "runs/ncf/"

# --- HYPERPARAMETERS ---
EMBEDDING_DIM = 64
LAYERS = [128, 64, 32] # MLP Layer sizes (Input layer dim will be calculated dynamically: emb + emb + genome)
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 16384 # Increased for 24GB VRAM (was 2048)
EPOCHS = 50
PATIENCE = 5 # Early Stopping
# NEGATIVES_RATIO removed for Regression (we use all existing ratings)

# --- SETTINGS ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
NUM_WORKERS = 0 # Set to 0 for Windows to avoid "OSError: Invalid argument" with large datasets
PIN_MEMORY = True if DEVICE == 'cuda' else False

# --- DATA SETTINGS ---
# POSITIVE_THRESHOLD removed (we predict actual ratings)
GENOME_DIM = 1128 # Fixed dimension of genome tags
MIN_RATING = 0.5
MAX_RATING = 5.0
