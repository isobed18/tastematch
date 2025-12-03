import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import pickle
import config
from model import HybridNCF
from tabulate import tabulate

def load_environment():
    # Load Paths
    run_dir = "runs_ncf/run_20251216_154202" # Hardcoded best run for debug
    
    print(f"Loading environment from {run_dir}...")
    
    # Load Mappings
    with open(os.path.join(run_dir, 'movie_mapping.pkl'), 'rb') as f:
        movie_mapping = pickle.load(f)
    print(f"Movies Loaded: {len(movie_mapping)}")
        
    # Load Genome
    genome_matrix = np.load(os.path.join(run_dir, 'genome_matrix.npy'))
    
    # Load Model structure
    # Note: We need NumUsers from training to initialize the model layers correctly, 
    # even if we don't use the user embedding layer for the new user.
    with open(os.path.join(run_dir, 'user_mapping.pkl'), 'rb') as f:
        user_mapping = pickle.load(f)
        
    model = HybridNCF(
        num_users=len(user_mapping),
        num_movies=len(movie_mapping),
        genome_dim=config.GENOME_DIM,
        embedding_dim=config.EMBEDDING_DIM
    )
    
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth')))
    model.to(config.DEVICE)
    model.eval()
    
    # Freeze Model Weights (We don't want to break the trained model)
    for param in model.parameters():
        param.requires_grad = False
        
    return model, movie_mapping, genome_matrix, run_dir

def get_movie_title(mid, movies_df):
    if mid in movies_df.index:
        return movies_df.loc[mid, 'title']
    return f"Movie {mid}"

def solve_user_vector(model, ratings_data, genome_matrix, device='cuda'):
    """
    Fold-In Strategy:
    Create a new random vector and `train` ONLY that vector to match the user's ratings.
    """
    print(f"\n--- FOLD-IN: Training New User Vector ({len(ratings_data)} ratings) ---")
    
    # 1. Initialize Random User Vector (Recycling standard embedding logic)
    # Shape: (1, 32)
    user_vector = torch.nn.Parameter(torch.normal(0, 0.1, size=(1, config.EMBEDDING_DIM)).to(device))
    
    # Optimizer: Only optimize this new vector
    optimizer = optim.Adam([user_vector], lr=0.05) # High LR for fast adaptation
    
    # Data to Tensor
    movie_indices = torch.tensor([r['movie_idx'] for r in ratings_data], dtype=torch.long).to(device)
    ratings = torch.tensor([r['rating'] for r in ratings_data], dtype=torch.float32).to(device)
    features = torch.tensor(genome_matrix[movie_indices.cpu().numpy()], dtype=torch.float32).to(device)
    
    # Training Loop for single vector
    steps = 50
    for i in range(steps):
        optimizer.zero_grad()
        
        # We need to hack the model forward pass to accept a vector instead of a user_id
        # Standard forward: user_emb = self.user_embedding(user_input)
        # We will bypass that.
        
        # Manual Forward Logic (Partial)
        # 1. Get Item Embeddings
        item_emb = model.movie_embedding(movie_indices)
        
        # 2. Concat: UserVec + ItemEmb + Features
        # user_vector is (1, 32), need to broadcast to (N, 32)
        batch_user_vec = user_vector.expand(len(ratings), -1)
        
        vector = torch.cat([batch_user_vec, item_emb, features], dim=-1)
        
        # 3. MLP
        x = model.mlp(vector)
        
        # 4. Output
        logits = model.output_layer(x).view(-1)
        pred = torch.clamp(logits, min=config.MIN_RATING, max=config.MAX_RATING)
        
        loss = nn.MSELoss()(pred, ratings)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i}: Loss = {loss.item():.4f}")
            
    print(f"Final Loss: {loss.item():.4f}")
    return user_vector.detach()

def recommend(model, user_vector, movie_mapping, genome_matrix, exclude_ids, top_k=10, device='cuda'):
    print("\n--- Generating Recommendations ---")
    
    all_movie_indices = np.array(list(movie_mapping.values()))
    
    # Filter out excluded (watched)
    candidate_indices = [idx for idx in all_movie_indices if idx not in exclude_ids]
    candidate_indices = torch.tensor(candidate_indices, dtype=torch.long).to(device)
    
    # Genome
    features = torch.tensor(genome_matrix[candidate_indices.cpu().numpy()], dtype=torch.float32).to(device)
    
    # Forward Pass
    with torch.no_grad():
         item_emb = model.movie_embedding(candidate_indices)
         batch_user_vec = user_vector.expand(len(candidate_indices), -1)
         vector = torch.cat([batch_user_vec, item_emb, features], dim=-1)
         x = model.mlp(vector)
         logits = model.output_layer(x).view(-1)
         preds = torch.clamp(logits, min=config.MIN_RATING, max=config.MAX_RATING)
         
    # Top K
    scores = preds.cpu().numpy()
    top_arg = scores.argsort()[::-1][:top_k]
    
    recs = []
    
    # Load titles for display
    movies_df = pd.read_csv(config.MOVIES_PATH).set_index('movieId')
    reverse_map = {v: k for k, v in movie_mapping.items()}
    
    for idx_pos in top_arg:
        original_idx = candidate_indices[idx_pos].item()
        score = scores[idx_pos]
        real_id = reverse_map[original_idx]
        title = get_movie_title(real_id, movies_df)
        recs.append([real_id, title, score])
        
    return recs

def main():
    device = config.DEVICE
    model, movie_mapping, genome_matrix, _ = load_environment()
    
    # --- SIMULATE NEW BACKEND USER ---
    # User likes Action and Sci-Fi, Hates Romance
    print("Simulating User: 'Sci-Fi Fan'")
    
    # Find IDs for some movies
    # Star Wars (260), Matrix (2571), Inception (79132) -> Rated 5.0
    # Notebook (8533), Titanic (1721) -> Rated 1.0
    
    simulated_ratings = [
        {'id': 260, 'rating': 5.0, 'name': 'Star Wars IV'},
        {'id': 2571, 'rating': 5.0, 'name': 'The Matrix'},
        {'id': 1721, 'rating': 2.0, 'name': 'Titanic'}, # Hates Titanic
        {'id': 1196, 'rating': 5.0, 'name': 'Star Wars V'},
    ]
    
    # Convert to model indices
    input_data = []
    for item in simulated_ratings:
        if item['id'] in movie_mapping:
            input_data.append({
                'movie_idx': movie_mapping[item['id']],
                'rating': item['rating'],
                'name': item['name']
            })
            
    # Show Input
    print(tabulate(simulated_ratings, headers="keys"))
    
    # 1. OPTIMIZE VECTOR
    user_vector = solve_user_vector(model, input_data, genome_matrix, device)
    
    # 2. RECOMMEND
    exclude_indices = [x['movie_idx'] for x in input_data]
    recs = recommend(model, user_vector, movie_mapping, genome_matrix, exclude_indices, top_k=10, device=device)
    
    print("\nTop 10 Recommendations:")
    print(tabulate(recs, headers=['ID', 'Title', 'Pred Score'], floatfmt=".2f"))

if __name__ == "__main__":
    main()
