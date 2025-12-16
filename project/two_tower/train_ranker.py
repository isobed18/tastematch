import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import pickle
from tqdm import tqdm

try:
    from . import config, ranking_models, inference_utils
except ImportError:
    import config, ranking_models, inference_utils

# Pairwise Dataset
class PairwiseRankingDataset(Dataset):
    def __init__(self, samples):
        # samples: List of (user_idx, pos_item_idx, neg_item_idx, weight)
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def generate_ranking_data(inference_engine, train_interactions_path, num_negatives=4):
    """
    Generate training data for ranker.
    Uses Retrieval Model to find Hard Negatives.
    """
    print("Generating Ranking Data (Hard Negatives)...")
    ranking_samples = []
    
    with open(train_interactions_path, 'rb') as f:
        all_users = pickle.load(f)
        
    # Limit for demo speed if needed, but intended for full training
    # all_users = all_users[:5000] 

    inference_engine.index_items()
    
    for user_data in tqdm(all_users, desc="Generating Candidates"):
        user_idx = user_data['user_idx']
        events = user_data['events'] # (item, weight, ts)
        
        if len(events) < 5:
            continue
            
        # Split History/Target for Ranker Training
        # We pretend the last 30% of TRAIN events are the 'future' for the ranker
        split = int(len(events) * 0.7)
        history_events = events[:split]
        target_events = events[split:]
        
        if not target_events:
            continue
            
        # 1. Build User Vector from History
        # inference_utils expects Raw IDs usually, but we have internal indices here.
        # We need to bridge this. Ideally, inference_utils should support internal indices.
        # We will assume we can pass internal indices if we bypass the map lookups.
        
        # Extract history item indices
        hist_items = [e[0] for e in history_events]
        # Only use Positive/Like items for history generation (ignore dislikes)
        # Assuming weights: 2.0 (Super), 1.0 (Like), 0.5 (Dislike)
        # We filter >= 1.0 for history
        hist_items_pos = [e[0] for e in history_events if e[1] >= 1.0]
        
        if not hist_items_pos:
            continue

        # Get Candidates (Retrieval Stage)
        # We manually call the model's retrieval logic to get internal indices directly
        k = 50
        with torch.no_grad():
            # Create tensors
            u_idx_t = torch.tensor([user_idx], device=inference_engine.device)
            h_idx_t = torch.tensor([hist_items_pos], device=inference_engine.device)
            h_w_t = torch.tensor([[1.0]*len(hist_items_pos)], device=inference_engine.device) # Simple avg
            h_mask_t = torch.ones_like(h_idx_t)
            
            user_vec = inference_engine.model.user_tower(u_idx_t, h_idx_t, h_w_t, h_mask_t)
            scores = torch.matmul(user_vec, inference_engine.item_embeddings.T).squeeze(0)
            top_scores, top_indices = torch.topk(scores, k)
            candidates = top_indices.cpu().numpy().tolist()

        # 2. Labeling
        target_items = {e[0] for e in target_events if e[1] >= 1.0} # Only predict positives
        
        # Create Pairs: (User, Pos, Neg)
        # Pos: From Target Items
        # Neg: From Candidates (Hard Negative) if not in Target
        
        hard_negatives = [c for c in candidates if c not in target_items]
        
        if not hard_negatives:
            continue
            
        for pos_item, weight, _ in target_events:
            if weight < 1.0: continue # Skip dislikes as positive targets
            
            # Sample Negatives
            curr_negs = random.sample(hard_negatives, min(len(hard_negatives), num_negatives))
            
            for neg_item in curr_negs:
                ranking_samples.append({
                    'user_idx': user_idx,
                    'pos_item': pos_item,
                    'neg_item': neg_item,
                    'weight': weight, # Higher weight for Superlike pairs
                    'user_vec': user_vec.cpu().numpy()[0] # Cache User Vec
                })
                
    return ranking_samples

def train_ranker():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Inference Engine (for Candidates & Vectors)
    inference = inference_utils.TwoTowerInference()
    
    # 2. Generate Data
    raw_data = generate_ranking_data(inference, config.TRAIN_INTERACTIONS_PATH)
    print(f"Generated {len(raw_data)} pairwise samples.")
    
    # 3. Prepare Dataset
    # We need item vectors.
    item_vectors = inference.item_embeddings.cpu().numpy() # [NumItems, 256]
    
    # Convert raw_data to Tensor dataset
    # Input: UserVec, PosItemVec, NegItemVec
    # But UserVec varies per sample.
    
    class RankerTrainDataset(Dataset):
        def __init__(self, data, item_vecs):
            self.data = data
            self.item_vecs = item_vecs
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            row = self.data[idx]
            # Features
            u_vec = row['user_vec']
            pos_vec = self.item_vecs[row['pos_item']]
            neg_vec = self.item_vecs[row['neg_item']]
            w = row['weight']
            return u_vec, pos_vec, neg_vec, w

    dataset = RankerTrainDataset(raw_data, item_vectors)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # 4. Model
    # Input Dim = EmbeddingDim(256) * 4 + Features
    ranker = ranking_models.RankerModel(embedding_dim=config.LATENT_DIM).to(device)
    optimizer = optim.Adam(ranker.parameters(), lr=1e-3)
    
    # 5. Training Loop
    print("Training Ranker...")
    ranker.train()
    
    for epoch in range(5): # Fast training for ranker
        total_loss = 0
        for u_vec, pos_vec, neg_vec, w in tqdm(dataloader):
            u_vec = u_vec.to(device)
            pos_vec = pos_vec.to(device)
            neg_vec = neg_vec.to(device)
            w = w.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pos
            pos_score = ranker(u_vec, pos_vec)
            # Forward Neg
            neg_score = ranker(u_vec, neg_vec)
            
            # Pairwise Logistic Loss: log(1 + exp(s_neg - s_pos))
            # Weighted by importance (Superlike > Like)
            loss = torch.mean(w * torch.log1p(torch.exp(neg_score - pos_score)))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
        
    # Save
    torch.save(ranker.state_dict(), os.path.join(config.DATA_DIR, "ranker_model.pth"))
    print("Ranker Saved.")

if __name__ == "__main__":
    train_ranker()