import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'project', 'data', 'two_tower')
ML_DATA_DIR = os.path.join(PROJECT_ROOT, 'project', 'data', 'ml-latest')
BACKEND_DATA_DIR = os.path.join(PROJECT_ROOT, 'backend', 'data')

# Input Files
MOVIES_CSV = os.path.join(ML_DATA_DIR, 'movies.csv')
RATINGS_CSV = os.path.join(ML_DATA_DIR, 'ratings.csv')
LINKS_CSV = os.path.join(ML_DATA_DIR, 'links.csv')
TAGS_CSV = os.path.join(ML_DATA_DIR, 'tags.csv')
GENOME_SCORES_CSV = os.path.join(ML_DATA_DIR, 'genome-scores.csv')
TMDB_MOVIES_CSV = os.path.join(BACKEND_DATA_DIR, 'tmdb-movies.csv')

# Output Artifacts
TEXT_EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'text_embeddings.npy')
GENOME_MATRIX_PATH = os.path.join(DATA_DIR, 'genome_matrix.npy')
GENRE_MATRIX_PATH = os.path.join(DATA_DIR, 'genre_matrix.npy')
# Mapping from internal ID to (MovieLensID, TMDBID)
ITEM_MAP_PATH = os.path.join(DATA_DIR, 'item_map.pkl') 
TRAIN_INTERACTIONS_PATH = os.path.join(DATA_DIR, 'train_interactions.pkl')
VAL_INTERACTIONS_PATH = os.path.join(DATA_DIR, 'val_interactions.pkl')
USER_MAP_PATH = os.path.join(DATA_DIR, 'user_map.pkl')

# Hyperparameters
EMBEDDING_DIM = 64 # Dimension for IDs
PROJECTION_DIM = 128 # Dimension for separate towers before concat

LATENT_DIM = 1024 # Final vector dimension (Increased from 256)
TEXT_EMBEDDING_DIM = 384 # Output of MiniLM
GENOME_DIM = 1128 # Standard ML Genome


BATCH_SIZE = 4096 # Increased for higher GPU Util
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 10
TEMPERATURE = 0.05 # InfoNCE Temp (Lowered from 0.07 for sharper predictions)

# Interaction Weights (Swipe Logic)
WEIGHT_SUPERLIKE = 2.0
WEIGHT_LIKE = 1.0
WEIGHT_DISLIKE = 0.5
