import pickle
import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from project.two_tower import config

def check_alignment():

    with open('debug_result.txt', 'w', encoding='utf-8') as log:
        def log_print(s):
            print(s)
            log.write(s + '\n')

        log_print("Checking alignment/shapes...")
        
        # 1. Check Item Map
        if not os.path.exists(config.ITEM_MAP_PATH):
            log_print("Item map not found.")
            return

        with open(config.ITEM_MAP_PATH, 'rb') as f:
            data = pickle.load(f)
            item_map = data['item_map']
            log_print(f"Item Map Size (Num Items): {len(item_map)}")
            
        # 2. Check Text Embeddings
        if os.path.exists(config.TEXT_EMBEDDINGS_PATH):
            text_emb = np.load(config.TEXT_EMBEDDINGS_PATH)
            log_print(f"Text Embeddings Shape: {text_emb.shape}")
            
            if text_emb.shape[0] != len(item_map):
                log_print("MISMATCH DETECTED: Text Embeddings rows != Item Map size")
                
                # Check for duplicates in data source logic
                log_print("Simulating merge to find duplicates...")
                movies_df = pd.read_csv(config.MOVIES_CSV)
                links_df = pd.read_csv(config.LINKS_CSV)
                tmdb_df = pd.read_csv(config.TMDB_MOVIES_CSV)
                
                # Replicate Preprocess Logic
                movies_df['item_idx'] = range(len(movies_df))
                
                links_df = links_df.dropna(subset=['tmdbId'])
                links_df['tmdbId'] = links_df['tmdbId'].astype(int)
                
                tmdb_subset = tmdb_df[['id', 'overview', 'title']].dropna(subset=['id'])
                tmdb_subset['id'] = tmdb_subset['id'].astype(int)
                tmdb_subset = tmdb_subset.rename(columns={'title': 'tmdb_title'})
                
                log_print(f"Links DF rows: {len(links_df)}")
                log_print(f"Links unique movieIds: {links_df['movieId'].nunique()}")
                
                merged_tmdb = links_df.merge(tmdb_subset, left_on='tmdbId', right_on='id', how='left')
                full_meta = movies_df.merge(merged_tmdb, on='movieId', how='left')
                
                log_print(f"Movies DF rows: {len(movies_df)}")
                log_print(f"Full Meta rows: {len(full_meta)}")
                
                if len(full_meta) > len(movies_df):
                    log_print("DUPLICATE ROWS FOUND IN MERGE!")
                    dup_counts = full_meta['movieId'].value_counts()
                    log_print(f"Movies with duplicates: {dup_counts[dup_counts > 1].head()}")
            else:
                log_print("Text Embeddings aligned.")
                
        # 3. Check Genome
        if os.path.exists(config.GENOME_MATRIX_PATH):
            genome = np.load(config.GENOME_MATRIX_PATH)
            log_print(f"Genome Matrix Shape: {genome.shape}")
            if genome.shape[0] != len(item_map):
                 log_print("MISMATCH: Genome rows != Item Map size")
                 
        # 4. Check Interactions
        if os.path.exists(config.TRAIN_INTERACTIONS_PATH):
            with open(config.TRAIN_INTERACTIONS_PATH, 'rb') as f:
                interactions = pickle.load(f)
            log_print(f"Train Users: {len(interactions)}")
            
            # Check max item index in interactions
            max_idx = 0
            for user in interactions:
                for event in user['events']:
                    if event[0] > max_idx:
                        max_idx = event[0]
            log_print(f"Max Item Index in Interactions: {max_idx}")
            log_print(f"Expected Max Index < {len(item_map)}")

if __name__ == "__main__":
    check_alignment()
