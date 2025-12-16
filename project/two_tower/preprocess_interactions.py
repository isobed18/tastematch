import pandas as pd
import numpy as np
import os
import sys
import pickle
from tqdm import tqdm

try:
    from . import config
except ImportError:
    import config

def preprocess_interactions():
    print("Loading Ratings...")
    ratings_df = pd.read_csv(config.RATINGS_CSV)
    
    # Optional: Load Tags if they exist and treat as weak likes
    # tags_df = pd.read_csv(config.TAGS_CSV)
    
    # Load Item Map to ensure consistent IDs
    with open(config.ITEM_MAP_PATH, 'rb') as f:
        data = pickle.load(f)
        item_map = data['item_map']
        
    # Filter ratings for items that exist in our map
    ratings_df = ratings_df[ratings_df['movieId'].isin(item_map)]
    ratings_df['item_idx'] = ratings_df['movieId'].map(item_map)
    
    # --- Weight Logic (Per Requirements) ---
    # Superlike (5.0) -> 2.0
    # Like (>= 4.0) -> 1.0
    # Dislike (<= 1.5) -> 0.5 (Used for history or negative mining)
    # Others (2.0 - 3.5) -> Neutral/Ignore? We'll treat as weak positive or 0.
    
    def get_weight(r):
        if r >= 5.0: return config.WEIGHT_SUPERLIKE # 2.0
        if r >= 4.0: return config.WEIGHT_LIKE      # 1.0
        if r <= 1.5: return config.WEIGHT_DISLIKE   # 0.5
        return 0.0 # Neutral
    
    ratings_df['weight'] = ratings_df['rating'].apply(get_weight)
    
    # Filter out Neutrals (0.0) from being targets, but keep them if needed?
    # For simplicity, we drop 0.0 weights to reduce noise
    ratings_df = ratings_df[ratings_df['weight'] > 0.0]

    # Sort by User, Timestamp (Strict Time Split)
    print("Sorting interactions...")
    ratings_df = ratings_df.sort_values(['userId', 'timestamp'])
    
    # Generate User Mapping
    unique_users = ratings_df['userId'].unique()
    user_map = {uid: i for i, uid in enumerate(unique_users)}
    ratings_df['user_idx'] = ratings_df['userId'].map(user_map)
    
    print(f"Total Users: {len(unique_users)}")
    
    train_data = []
    val_data = []
    
    print("Splitting Time-based...")
    grouped = ratings_df.groupby('user_idx')
    
    min_interactions = 5
    
    for uid, group in tqdm(grouped):
        if len(group) < min_interactions:
            continue
            
        # Events list: (item_idx, weight, timestamp)
        # We also keep the raw rating to distinguish Dislike from Like if needed later
        events = list(zip(group['item_idx'], group['weight'], group['timestamp']))
        
        # Strict Time Split (e.g., last 20% for Validation)
        split_idx = int(len(events) * 0.8)
        
        # Train Part
        train_events = events[:split_idx]
        # Val Part
        val_events = events[split_idx:]
        
        # Skip if train or val is empty
        if not train_events or not val_events:
            continue
            
        train_data.append({
            'user_idx': uid,
            'events': train_events
        })
        
        val_data.append({
            'user_idx': uid,
            'events': val_events, 
            'context': train_events # Context for validation history
        })

    # Save
    with open(config.TRAIN_INTERACTIONS_PATH, 'wb') as f:
        pickle.dump(train_data, f)
        
    with open(config.VAL_INTERACTIONS_PATH, 'wb') as f:
        pickle.dump(val_data, f)
        
    with open(config.USER_MAP_PATH, 'wb') as f:
        pickle.dump(user_map, f)
        
    print("Interaction Processing Complete.")

if __name__ == "__main__":
    if not os.path.exists(config.ITEM_MAP_PATH):
        print("Error: content not processed. Run preprocess_content first.")
    else:
        preprocess_interactions()