import argparse
import os
import torch
import pandas as pd
import numpy as np
import pickle
import config
from model import HybridNCF
from tabulate import tabulate # Requires tabulate, usually available or I can print simple table

def parse_args():
    parser = argparse.ArgumentParser(description="Test NCF Model Predictions")
    parser.add_argument('--run_dir', type=str, default='runs_ncf/run_20251216_154202', help='Path to the run directory containing best_model.pth and mappings')
    parser.add_argument('--user_id', type=int, default=None, help='Specific User ID to test (Original ID from CSV)')
    parser.add_argument('--num_users', type=int, default=3, help='Number of random users to test if user_id is not provided')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    return parser.parse_args()

def load_artifacts(run_dir):
    print(f"Loading artifacts from {run_dir}...")
    
    with open(os.path.join(run_dir, 'user_mapping.pkl'), 'rb') as f:
        user_mapping = pickle.load(f)
        
    with open(os.path.join(run_dir, 'movie_mapping.pkl'), 'rb') as f:
        movie_mapping = pickle.load(f)
        
    genome_matrix = np.load(os.path.join(run_dir, 'genome_matrix.npy'))
    
    return user_mapping, movie_mapping, genome_matrix

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Artifacts
    user_mapping, movie_mapping, genome_matrix = load_artifacts(args.run_dir)
    
    # 2. Load Data for Ground Truth
    print("Loading Ratings and Movies for verification...")
    ratings_df = pd.read_csv(config.RATINGS_PATH)
    movies_df = pd.read_csv(config.MOVIES_PATH).set_index('movieId')
    
    # 3. Select Users
    if args.user_id:
        if args.user_id not in user_mapping:
            print(f"Error: User {args.user_id} not found in model mappings.")
            return
        target_users = [args.user_id]
        print(f"Targeting User: {args.user_id}")
    else:
        # Filter users who have at least 10 ratings for better visibility
        user_counts = ratings_df['userId'].value_counts()
        eligible_users = user_counts[user_counts >= 10].index.tolist()
        # Filter eligible users to those in mapping
        eligible_users = [u for u in eligible_users if u in user_mapping]
        
        target_users = np.random.choice(eligible_users, size=args.num_users, replace=False)
        print(f"Selected Random Users: {target_users}")

    # 4. Initialize Model
    num_users = len(user_mapping)
    num_movies = len(movie_mapping)
    
    model = HybridNCF(
        num_users=num_users,
        num_movies=num_movies,
        genome_dim=config.GENOME_DIM,
        embedding_dim=config.EMBEDDING_DIM
        # Config layers/dropout assumed same as config.py
    )
    
    model_path = os.path.join(args.run_dir, 'best_model.pth')
    print(f"Loading Model Weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Move genome to GPU
    genome_tensor = torch.tensor(genome_matrix, dtype=torch.float32).to(device)

    # 5. Predict and Compare
    for uid in target_users:
        print(f"\n{'='*60}")
        print(f"User ID: {uid}")
        print(f"{'='*60}")
        
        # Get User's Actual Ratings
        user_ratings = ratings_df[ratings_df['userId'] == uid].copy()
        
        # Filter movies that are in our mapping (model only knows known movies)
        user_ratings = user_ratings[user_ratings['movieId'].isin(movie_mapping)]
        
        if len(user_ratings) == 0:
            print("No valid movies found for this user in the model's scope.")
            continue
            
        # Limit to top 20 latest ratings or random schema? Let's take 15 random ratings to show variety
        if len(user_ratings) > 20:
            user_ratings = user_ratings.sample(20, random_state=42)
            
        # Prepare Batch
        u_idx = user_mapping[uid]
        m_idxs = [movie_mapping[mid] for mid in user_ratings['movieId'].values]
        
        u_tensor = torch.tensor([u_idx] * len(m_idxs), dtype=torch.long).to(device)
        m_tensor = torch.tensor(m_idxs, dtype=torch.long).to(device)
        f_tensor = genome_tensor[m_tensor] # Gather features
        
        # Predict
        with torch.no_grad():
            preds = model(u_tensor, m_tensor, f_tensor)
            # Ensure output is in expected range (model might already contain clamp/sigmoid depending on version)
            # User reverted to Sigmoid+Scaling model. The output is already scaled 0.5-5.0.
            preds = preds.cpu().numpy()
            
        # Display
        results = []
        ae_sum = 0
        se_sum = 0
        
        for i, (idx, row) in enumerate(user_ratings.iterrows()):
            mid = row['movieId']
            actual = row['rating']
            pred = preds[i]
            
            title = movies_df.loc[mid, 'title'] if mid in movies_df.index else f"Movie {mid}"
            # Truncate title
            if len(title) > 40:
                title = title[:37] + "..."
                
            diff = pred - actual
            ae_sum += abs(diff)
            se_sum += diff**2
            
            results.append([mid, title, actual, f"{pred:.2f}", f"{diff:+.2f}"])
            
        # Print Table
        headers = ["MovID", "Title", "Actual", "Pred", "Diff"]
        try:
            print(tabulate(results, headers=headers, tablefmt="github"))
        except ImportError:
            # Fallback if tabulate not installed
            print(f"{'MovID':<8} {'Title':<40} {'Actual':<6} {'Pred':<6} {'Diff':<6}")
            print("-" * 70)
            for r in results:
                print(f"{r[0]:<8} {r[1]:<40} {r[2]:<6} {r[3]:<6} {r[4]:<6}")
                
        # Metrics
        n = len(results)
        rmse = np.sqrt(se_sum / n)
        mae = ae_sum / n
        print(f"\nUser Specific RMSE: {rmse:.4f} | MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
