import pandas as pd
import numpy as np
import os
import sys
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Add project root to path
try:
    from . import config
except ImportError:
    import config

def preprocess_content():
    print("Loading Data...")
    movies_df = pd.read_csv(config.MOVIES_CSV)
    links_df = pd.read_csv(config.LINKS_CSV)
    tmdb_df = pd.read_csv(config.TMDB_MOVIES_CSV)
    genome_scores_df = pd.read_csv(config.GENOME_SCORES_CSV)
    
    # 1. Create Internal Item Map (Continuous ID: 0..N-1)
    # Filter only movies that exist in movies.csv (ML)
    # We will prioritize items that have TMDB data, but keep all ML items?
    # Better to keep items that have User Interactions. 
    # For now, we take all movies in movies.csv as the base universe.
    
    movies_df['item_idx'] = range(len(movies_df))
    item_map = dict(zip(movies_df['movieId'], movies_df['item_idx']))
    # Inverse map
    idx_to_ml_id = dict(zip(movies_df['item_idx'], movies_df['movieId']))
    
    num_items = len(movies_df)
    print(f"Total Movies: {num_items}")
    
    # 2. Process TMDB Overviews (Text Embeddings)
    print("Processing Text Embeddings...")
    try:
        # Link ML -> TMDB
        links_df = links_df.dropna(subset=['tmdbId'])
        links_df['tmdbId'] = links_df['tmdbId'].astype(int)
        
        # Merge TMDB data onto Links
        # tmdb_df: id, overview
        tmdb_subset = tmdb_df[['id', 'overview', 'title']].dropna(subset=['id'])
        tmdb_subset['id'] = tmdb_subset['id'].astype(int)
        
        # Rename TMDB title to avoid collision before merge
        tmdb_subset = tmdb_subset.rename(columns={'title': 'tmdb_title'})
        
        merged_tmdb = links_df.merge(tmdb_subset, left_on='tmdbId', right_on='id', how='left')
        
        # Now merge back to movies_df to align with item_idx
        full_meta = movies_df.merge(merged_tmdb, on='movieId', how='left')
        
        print(f"Columns after merge: {full_meta.columns.tolist()}")
        full_meta = full_meta.sort_values('item_idx')
        # Fill missing overviews with Title or Placeholder
        # Note: 'title' should come from movies_df (MovieLens)
        full_meta['overview'] = full_meta['overview'].fillna('')
        # Ensure 'overview' is string
        full_meta['overview'] = full_meta['overview'].astype(str)
        
        # Use movie title if overview is missing
        full_meta['text_input'] = full_meta.apply(
            lambda x: x['overview'] if len(x['overview']) > 5 else f"{x['title']} {x['genres']}", 
            axis=1
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error columns: {full_meta.columns.tolist() if 'full_meta' in locals() else 'Not created'}")
        raise e

    # Encode
    print("Loading SBERT Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Batch encode
    texts = full_meta['text_input'].tolist()
    print(f"Encoding {len(texts)} texts...")
    text_embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Save Text Embeddings
    np.save(config.TEXT_EMBEDDINGS_PATH, text_embeddings)
    print(f"Saved Text Embeddings: {text_embeddings.shape}")
    
    # 3. Process Genome (Dense Matrix)
    print("Processing Genome...")
    # genome_scores: movieId, tagId, relevance
    # Pivot: Index=movieId, Columns=tagId
    
    # Filter genome for movies we have
    genome_scores_df = genome_scores_df[genome_scores_df['movieId'].isin(item_map.keys())]
    
    # Create empty matrix [NumItems, 1128]
    # We rely on tagIds being 1..1128. Let's verify max tag Id.
    max_tag = genome_scores_df['tagId'].max()
    print(f"Max Tag ID: {max_tag}")
    
    # We need to map movieId -> item_idx for the pivot
    genome_scores_df['item_idx'] = genome_scores_df['movieId'].map(item_map)
    
    # Manual pivot to ensure alignment
    # Init zero matrix
    genome_matrix = np.zeros((num_items, max_tag + 1), dtype=np.float32) # +1 for 1-based index (will drop col 0)
    
    # Fill
    # Using numpy indexing for speed
    rows = genome_scores_df['item_idx'].values
    cols = genome_scores_df['tagId'].values
    vals = genome_scores_df['relevance'].values
    
    genome_matrix[rows, cols] = vals
    genome_matrix = genome_matrix[:, 1:] # Drop col 0 (unused)
    
    np.save(config.GENOME_MATRIX_PATH, genome_matrix)
    print(f"Saved Genome Matrix: {genome_matrix.shape}")
    
    # 4. Process Genres
    print("Processing Genres...")
    # Expand genres
    all_genres = set()
    for g_str in movies_df['genres']:
        for g in g_str.split('|'):
            all_genres.add(g)
            
    genres_list = sorted(list(all_genres))
    genre_map = {g: i for i, g in enumerate(genres_list)}
    
    genre_matrix = np.zeros((num_items, len(genres_list)), dtype=np.float32)
    
    for idx, g_str in zip(movies_df['item_idx'], movies_df['genres']):
        for g in g_str.split('|'):
            if g in genre_map:
                genre_matrix[idx, genre_map[g]] = 1.0
                
    np.save(config.GENRE_MATRIX_PATH, genre_matrix)
    print(f"Saved Genre Matrix: {genre_matrix.shape}")
    
    # 5. Save Map
    with open(config.ITEM_MAP_PATH, 'wb') as f:
        pickle.dump({'item_map': item_map, 'idx_to_ml_id': idx_to_ml_id, 'genre_list': genres_list}, f)
        
    print("Pre-processing Complete.")

if __name__ == "__main__":
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    preprocess_content()
