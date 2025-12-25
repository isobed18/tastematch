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

def generate_ranking_data(inference_engine, train_interactions_path, num_negatives=4, batch_size_gen=512):
    """
    Generate training data for ranker.
    Uses Retrieval Model to find Hard Negatives (Batched Version).
    """
    print("Generating Ranking Data (Hard Negatives)...")
    ranking_samples = []
    
    with open(train_interactions_path, 'rb') as f:
        all_users = pickle.load(f)
        
    inference_engine.index_items()
    
    # Filter valid users first to avoid complexity in batch loop
    valid_users = []
    for u in all_users:
        if len(u['events']) >= 5:
             # Just basic filter, history splitting logic applied inside loop
             valid_users.append(u)
             
    print(f"Processing {len(valid_users)} valid users for training data...")
    
    # Process in Batches
    for i in tqdm(range(0, len(valid_users), batch_size_gen), desc="Generating Candidates (Batched)"):
        batch_users = valid_users[i : i + batch_size_gen]
        
        # Prepare Batch Tensors
        u_indices = []
        h_indices_list = []
        h_weights_list = []
        
        user_meta_list = [] # Store metadata to process results later
        
        for user_data in batch_users:
            user_idx = user_data['user_idx']
            events = user_data['events']
            
            split = int(len(events) * 0.7)
            history_events = events[:split]
            target_events = events[split:]
            
            if not target_events:
                 continue
                 
            # Extract history (Positive only)
            hist_items_pos = [e[0] for e in history_events if e[1] >= 1.0]
            if not hist_items_pos:
                hist_items_pos = [0] # Padding if empty but valid?
            
            u_indices.append(user_idx)
            h_indices_list.append(hist_items_pos)
            h_weights_list.append([1.0] * len(hist_items_pos))
            
            user_meta_list.append({
                'user_idx': user_idx,
                'target_events': target_events
            })
            
        if not u_indices:
            continue
            
        # Pad Sequences
        max_len = max([len(h) for h in h_indices_list])
        h_idx_padded = np.zeros((len(u_indices), max_len), dtype=int)
        h_w_padded = np.zeros((len(u_indices), max_len), dtype=float)
        h_mask_padded = np.zeros((len(u_indices), max_len), dtype=float)
        
        for idx, (h, w) in enumerate(zip(h_indices_list, h_weights_list)):
            l = len(h)
            h_idx_padded[idx, :l] = h
            h_w_padded[idx, :l] = w
            h_mask_padded[idx, :l] = 1.0
            
        # To Tensor
        u_idx_t = torch.tensor(u_indices, device=inference_engine.device)
        h_idx_t = torch.tensor(h_idx_padded, device=inference_engine.device, dtype=torch.long)
        h_w_t = torch.tensor(h_w_padded, device=inference_engine.device, dtype=torch.float)
        h_mask_t = torch.tensor(h_mask_padded, device=inference_engine.device, dtype=torch.float)
        
        # Batch Inference
        with torch.no_grad():
            user_vecs = inference_engine.model.user_tower(u_idx_t, h_idx_t, h_w_t, h_mask_t) # [B, Dim]
            # Dot Product with ALL Items
            # [B, Dim] x [NumItems, Dim]^T = [B, NumItems]
            scores = torch.matmul(user_vecs, inference_engine.item_embeddings.T)
            
            # Top K
            top_scores, top_indices = torch.topk(scores, k=50, dim=1)
            candidates_batch = top_indices.cpu().numpy() # [B, 50]
            user_vecs_cpu = user_vecs.cpu().numpy() # [B, Dim]
            
        # Process Results
        # Be careful: user_meta_list matches indices of batch result IF we didn't skip any inside logic.
        # We collected u_indices aligned with user_meta_list. So i-th output corresponds to i-th meta.
        
        for j, meta in enumerate(user_meta_list):
            user_idx = meta['user_idx']
            target_events = meta['target_events']
            
            candidates = candidates_batch[j]
            user_vec = user_vecs_cpu[j]
            
            # Identify Hard Negatives
            target_item_ids = {e[0] for e in target_events if e[1] >= 1.0}
            hard_negatives = [c for c in candidates if c not in target_item_ids]
            
            if not hard_negatives: continue
            
            for pos_item, weight, _ in target_events:
                 if weight < 1.0: continue
                 curr_negs = random.sample(hard_negatives, min(len(hard_negatives), num_negatives))
                 for neg_item in curr_negs:
                     ranking_samples.append({
                        'user_idx': user_idx,
                        'pos_item': pos_item,
                        'neg_item': neg_item,
                        'weight': weight,
                        'user_vec': user_vec
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
    # Input Dim = EmbeddingDim * 4 + Features
    actual_embedding_dim = item_vectors.shape[1]
    print(f"Initializing Ranker with Embedding Dim: {actual_embedding_dim}")
    
    ranker = ranking_models.RankerModel(embedding_dim=actual_embedding_dim).to(device)
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