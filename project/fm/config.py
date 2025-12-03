import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.join(BASE_DIR, "project")
DATA_DIR = os.path.join(PROJECT_DIR, "data", "ml-latest")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
FM_DIR = os.path.join(PROJECT_DIR, "fm")

# Paths
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
GENOME_SCORES_PATH = os.path.join(DATA_DIR, "genome-scores.csv")
GENOME_TAGS_PATH = os.path.join(DATA_DIR, "genome-tags.csv")

# Train Config Defaults
DEFAULT_OUTPUT_DIR = os.path.join(FM_DIR, "runs")
DROP_THRESHOLD_PCT = 0.20  # 20% drop for densification

# Model Defaults
NO_COMPONENTS = 64
LEARNING_RATE = 0.05
LOSS = 'warp' # warp or bpr
EPOCHS = 10
BATCH_SIZE = 2048 # LightFM doesn't use batch size in fit directly but good for custom loops if needed
NUM_THREADS = 10
