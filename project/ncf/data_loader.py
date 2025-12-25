import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import config

class NCFDataProcessor:
    def __init__(self):
        self.user_mapping = {}
        self.movie_mapping = {}
        self.reverse_movie_mapping = {}
        self.genome_matrix = None
        self.num_users = 0
        self.num_movies = 0

    def load_data(self):
        print("Loading data...")
        # Load Ratings
        ratings = pd.read_csv(config.RATINGS_PATH)
        
        # Load Genome
        genome_scores = pd.read_csv(config.GENOME_SCORES_PATH)
        
        # 1. Map IDs
        print("Mapping IDs...")
        unique_users = ratings['userId'].unique()
        unique_movies = ratings['movieId'].unique()
        
        self.user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_mapping = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.reverse_movie_mapping = {idx: mid for mid, idx in self.movie_mapping.items()}
        
        self.num_users = len(unique_users)
        self.num_movies = len(unique_movies)
        
        # Apply Mapping to Ratings
        ratings['user_idx'] = ratings['userId'].map(self.user_mapping)
        ratings['movie_idx'] = ratings['movieId'].map(self.movie_mapping)
        
        # 2. Build Genome Matrix (M x 1128)
        print("Building Genome Matrix...")
        # Initialize with zeros
        # genome-scores.csv: movieId, tagId, relevance
        # TagIds should be 1..1128. Let's map them to 0..1127 just in case.
        tag_ids = genome_scores['tagId'].unique()
        tag_mapping = {tid: idx for idx, tid in enumerate(sorted(tag_ids))}
        
        # Create a matrix of shape (NumMovies, NumTags)
        # Note: We only care about movies that exist in our ratings dataset
        self.genome_matrix = np.zeros((self.num_movies, len(tag_ids)), dtype=np.float32)
        
        # Filter genome scores for relevant movies
        relevant_gs = genome_scores[genome_scores['movieId'].isin(self.movie_mapping.keys())]
        
        # Iterate and fill (Optimization: use pivoting)
        print("  Pivoting Genome Scores...")
        pivot = relevant_gs.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)
        
        # Fill the tensor in order of movie_idx
        # This ensures row i of genome_matrix corresponds to movie_idx=i
        print("  Aligning Genome Matrix...")
        for mid, idx in tqdm(self.movie_mapping.items(), desc="Aligning Genome"):
            if mid in pivot.index:
                # Assuming pivot columns are sorted by tagId, if we strictly used sorted tags above
                # But pivot columns are tagIds. We need strict alignment.
                # Safe way:
                vals = pivot.loc[mid].values
                # We need to ensure vals order matches our tensor columns.
                # If pivot columns are already ordered tagIds, we are good.
                # Pivot automatically sorts columns (tagIds).
                self.genome_matrix[idx, :] = vals
            # Else remains 0
            
        return ratings

    def split_train_test(self, ratings):
        print("Splitting Train/Test (Regression Mode)...")
        # Use all available ratings for regression
        
        # Train/Test Split
        train, test = train_test_split(ratings, test_size=0.2, random_state=config.SEED)
        
        return train, test, None # No "positives_set" needed for regression

class HybridNCFDataset(Dataset):
    def __init__(self, ratings_df, genome_matrix):
        self.ratings_df = ratings_df
        self.genome_matrix = genome_matrix # Numpy array (M, 1128)
        
        self.data = []
        self._generate_data()

    def _generate_data(self):
        print(f"Generating Data...")
        
        self.users_list = self.ratings_df['user_idx'].values
        self.items_list = self.ratings_df['movie_idx'].values
        # Use actual ratings as labels
        self.labels_list = self.ratings_df['rating'].values.astype(np.float32)
        
        self.length = len(self.users_list)
        print(f"  Total Samples: {self.length}")
                            
        self.length = len(self.users_list)
        print(f"  Total Samples: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        user_idx = self.users_list[idx]
        movie_idx = self.items_list[idx]
        label = self.labels_list[idx]
        
        # Get Genome Features
        features = self.genome_matrix[movie_idx] # Shape (1128,)
        
        return {
            'user': torch.tensor(user_idx, dtype=torch.long),
            'movie': torch.tensor(movie_idx, dtype=torch.long),
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }
