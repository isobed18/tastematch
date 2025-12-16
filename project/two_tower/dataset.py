import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import random

class TwoTowerDataset(Dataset):
    def __init__(self, interactions_path, item_map_path, max_history=20, is_train=True):
        """
        interactions_path: Path to .pkl file containing list of user dicts.
                           [{'user_idx': u, 'events': [(item, weight, ts), ...]}, ...]
        max_history: Number of recent items to use for user representation.
        """
        with open(interactions_path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.max_history = max_history
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        user_data = self.data[idx]
        user_idx = user_data['user_idx']
        events = user_data['events'] # list of (item, weight, ts)
        
        # 1. Sample Positive Target
        # For training, we want to predict the *next* item given history.
        # Or just 'an' item given 'other' items (Masked Language Model style?)
        # Standard Recommender: Predict Item[t] given Item[0...t-1].
        # In this strict time-split setup: 
        # We should iterate over ALL events as targets? 
        # Or sample one per epoch per user? Sampling is faster for large datasets.
        
        if self.is_train:
            # Randomly pick one event as 'target', use previous events as history.
            # Must have at least 1 prior event to have history? 
            # If only 1 event, history is empty (cold start).

            # Filter valid target indices (must have weight >= 1.0)
            valid_indices = [i for i in range(1, len(events)) if events[i][1] >= 1.0]
            
            if valid_indices:
                target_idx = random.choice(valid_indices)
            else:
                # Fallback: if all valid interactions are dislikes, just pick any random one
                # or better, pick the one with highest weight
                target_idx = random.randint(1, len(events) - 1)
                
            target_event = events[target_idx]
            history_events = events[:target_idx]
            
        else:
            # Validation
            # Target is from 'events' (which are val_events).
            # History is from 'context' (train_events) + prior val_events?
            # Usually validation is: Predict next item given ONLY training history?
            # The preprocessing stored 'context' which is the full train history.
            # If we predict a specific val event, we technically know the validation events before it.
            # But simpler: History = Train Items. Target = Masked Validation Item.

            # Let's sample a target from Validation set.
            # Filter valid (Like/Superlike)
            valid_indices = [i for i in range(len(events)) if events[i][1] >= 1.0]
            if valid_indices:
                target_idx = random.choice(valid_indices)
            else:
                target_idx = random.randint(0, len(events) - 1)
                
            target_event = events[target_idx]
            
            # History comes from the 'context' field we saved + events before target
            # Use .get('context', [])
            history_events = user_data.get('context', []) + events[:target_idx]
            
        # 2. Process Target
        pos_item_idx = target_event[0]
        pos_weight = target_event[1]
        

        # 3. Process History
        # Filter out dislikes from history (keep only weight >= 1.0)
        # We want the User Vector to represent "What I like", not "What I hate".
        # (Unless we had a specific "Hated Tower", but for now, ignore dislikes)
        positive_history = [e for e in history_events if e[1] >= 1.0]
        
        # Take last N items from Positive History
        recent_history = positive_history[-self.max_history:]
        
        if len(recent_history) == 0:
            hist_indices = [0] # Placeholder
            hist_weights = [0.0]
            hist_mask = [0.0] # Empty
        else:
            hist_indices = [e[0] for e in recent_history]
            hist_weights = [float(e[1]) for e in recent_history]
            hist_mask = [1.0] * len(hist_indices)
            
        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'item_idx': torch.tensor(pos_item_idx, dtype=torch.long), # The Positive Target
            'weight': torch.tensor(pos_weight, dtype=torch.float),
            'hist_indices': torch.tensor(hist_indices, dtype=torch.long),
            'hist_weights': torch.tensor(hist_weights, dtype=torch.float),
            'hist_mask': torch.tensor(hist_mask, dtype=torch.float) # 1 for valid, 0 for padding
        }

def collate_fn(batch):
    # Determine max history length in this batch
    max_len = 0
    for item in batch:
        max_len = max(max_len, item['hist_indices'].shape[0])
        
    user_idxs = []
    item_idxs = []
    weights = []
    hist_indices_padded = []
    hist_weights_padded = []
    hist_masks_padded = []
    
    for item in batch:
        user_idxs.append(item['user_idx'])
        item_idxs.append(item['item_idx'])
        weights.append(item['weight'])
        
        curr_len = item['hist_indices'].shape[0]
        pad_len = max_len - curr_len
        
        # Pad with 0
        h_idx = torch.cat([item['hist_indices'], torch.zeros(pad_len, dtype=torch.long)])
        h_w = torch.cat([item['hist_weights'], torch.zeros(pad_len, dtype=torch.float)])
        h_m = torch.cat([item['hist_mask'], torch.zeros(pad_len, dtype=torch.float)])
        
        hist_indices_padded.append(h_idx)
        hist_weights_padded.append(h_w)
        hist_masks_padded.append(h_m)
        
    return {
        'user_idx': torch.stack(user_idxs),
        'item_idx': torch.stack(item_idxs),
        'weight': torch.stack(weights),
        'hist_indices': torch.stack(hist_indices_padded),
        'hist_weights': torch.stack(hist_weights_padded),
        'hist_mask': torch.stack(hist_masks_padded)
    }
