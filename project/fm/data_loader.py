import pandas as pd
import numpy as np
import scipy.sparse as sp
from lightfm.data import Dataset
import config
import os

class FMDataLoader:
    def __init__(self, drop_pct=0.20):
        self.drop_pct = drop_pct
        self.ratings_df = None
        self.genome_scores_df = None
        self.movies_df = None
        self.dataset = None
        
    def load_and_process(self):
        print(f"Loading data from {config.DATA_DIR}...")
        
        # 1. Load Ratings (Optimized dtype)
        self.ratings_df = pd.read_csv(config.RATINGS_PATH, 
                                      usecols=['userId', 'movieId', 'rating'],
                                      dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        print(f"Initial Ratings: {len(self.ratings_df)}")

        # 2. Densification (Drop sparse users/items)
        self._densify_data()
        
        # 3. Load Genome Scores (Content Features)
        # Filter genome scores to keep only relevant movies
        print("Loading Genome Scores...")
        self.genome_scores_df = pd.read_csv(config.GENOME_SCORES_PATH)
        
        # Only keep movies that survived filtering
        valid_movies = self.ratings_df['movieId'].unique()
        self.genome_scores_df = self.genome_scores_df[self.genome_scores_df['movieId'].isin(valid_movies)]
        
        print(f"Genome Scores Filtered: {len(self.genome_scores_df)}")

        # 4. Build LightFM Dataset
        self._build_dataset()
        
        return self

    def _densify_data(self):
        # STRATEGY: Keep strictly the top `keep_top_pct` (e.g., 0.1) of users and items.
        # This creates a small, super-dense core dataset for debugging/training.
        target_pct = 0.005 # Keep top 0.5% (Ultra-fast for metric verification)
        print(f"Applying Aggressive Densification (Keep Top {target_pct*100}% of Users/Items)...")
        
        # 1. Keep Top Users
        user_counts = self.ratings_df['userId'].value_counts()
        # Find the count threshold for the top 10%
        top_user_cutoff = user_counts.quantile(1.0 - target_pct) 
        print(f"User Activity Threshold (Top 10%): > {top_user_cutoff} ratings")
        
        active_users = user_counts[user_counts >= top_user_cutoff].index
        self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(active_users)]
        print(f"Ratings after filtering users: {len(self.ratings_df)}")
        
        # 2. Keep Top Movies (from the remaining ratings)
        movie_counts = self.ratings_df['movieId'].value_counts()
        top_movie_cutoff = movie_counts.quantile(1.0 - target_pct)
        print(f"Movie Popularity Threshold (Top 10%): > {top_movie_cutoff} ratings")
        
        active_movies = movie_counts[movie_counts >= top_movie_cutoff].index
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(active_movies)]
        print(f"Ratings after filtering movies: {len(self.ratings_df)}")

    def _build_dataset(self):
        print("Building LightFM Dataset...")
        self.dataset = Dataset()
        
        # Unique features (tags)
        # Genome scores format: movieId, tagId, relevance
        unique_tags = self.genome_scores_df['tagId'].unique()
        
        feature_list = [f"tag:{t}" for t in unique_tags]
        
        print(f"Fitting Dataset with unique entities...")
        self.dataset.fit(
            users=self.ratings_df['userId'].unique(),
            items=self.ratings_df['movieId'].unique(),
            item_features=feature_list
        )

    def get_matrices(self):
        # 1. Interaction Matrix
        print("Building Interaction Matrix...")
        (interactions, weights) = self.dataset.build_interactions(
            ((row.userId, row.movieId, row.rating) for row in self.ratings_df.itertuples())
        )
        
        # ENFORCE CSR AND FLOAT32
        interactions = interactions.tocsr().astype(np.float32)
        
        # 2. Item Features Matrix
        print("Building Item Features Matrix...")
        
        # Pre-compute pivot for faster lookup
        unique_movies = self.ratings_df['movieId'].unique()
        # Filter genome scores to only relevant movies first to save memory
        relevant_genome = self.genome_scores_df[self.genome_scores_df['movieId'].isin(unique_movies)]
        
        if relevant_genome.empty:
            print("WARNING: No genome scores found for the selected top movies! Features will be empty.")
            gs_pivot = pd.DataFrame()
        else:
            gs_pivot = relevant_genome.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)
        
        def item_features_generator():
            for movie_id in unique_movies:
                if movie_id in gs_pivot.index:
                    tags = gs_pivot.loc[movie_id]
                    feats = {f"tag:{tag_id}": score for tag_id, score in tags.items() if score > 0}
                    yield (movie_id, feats)
                else:
                    yield (movie_id, {})

        item_features = self.dataset.build_item_features(
            item_features_generator()
        )
        
        # ENFORCE CSR AND FLOAT32
        item_features = item_features.tocsr().astype(np.float32)
        
        return interactions, weights, item_features, self.dataset.mapping()

if __name__ == "__main__":
    # Test run
    loader = FMDataLoader(drop_pct=0.2)
    loader.load_and_process()
    print("Test Complete")
