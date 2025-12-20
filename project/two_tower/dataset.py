import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import random



class TwoTowerDataset(Dataset):
    def __init__(self, interactions_path, item_map_path, max_history=50, is_train=True):
        """
        interactions_path: Path to .pkl file containing list of user dicts.
        """
        with open(interactions_path, 'rb') as f:
            raw_data = pickle.load(f)
            
        self.max_history = max_history
        self.is_train = is_train
        self.samples = []
        

        # Optimization: Store ONLY positive events for each user to avoid filtering in __getitem__
        # Format: self.user_positives[user_idx] = [(item_idx, weight), ...]
        self.user_positives = []
        self.user_ids = [] # Store actual User IDs corresponding to the user_positives list
        
        print("Pre-processing dataset for fast access...")
        
        for i, user_data in enumerate(raw_data):
            events = user_data['events'] # raw events
            real_user_id = user_data['user_idx']
            
            # Pre-filter positives once
            # If validation, we also need to include 'context' in the history timeline
            if not is_train and 'context' in user_data:
                # Validation: Prepend context to events for history purposes
                # Context is train history.
                full_timeline = user_data['context'] + events
            else:
                full_timeline = events
                

            # Filter duplicates or just filter by weight?
            # Filter Skips/Ignores (keep Dislikes with weight 0.5)
            # Reverting logic as requested: Include Dislikes.
            pos_events = [e for e in full_timeline if e[1] > 0.0]
            self.user_positives.append(pos_events)
            self.user_ids.append(real_user_id)
            
            # Now generate samples pointing to indices in pos_events
            # For Training: Sliding window over pos_events
            if self.is_train:
                if len(pos_events) < 2:
                    continue
                    
                # We need at least 1 history item.
                # pos_events = [p0, p1, p2, ...]
                # Target p1 -> Hist [p0]
                # Target p2 -> Hist [p0, p1]
                for k in range(1, len(pos_events)):
                    # (user_index_in_list, target_index_in_pos_events)
                    self.samples.append((i, k))
                    
            else:
                # Validation logic
                # Target must be from the 'events' part (not context)
                # But 'pos_events' is merged. We need to find which ones are valid targets.
                # Valid targets are those that came from 'events'.
                # Since we concatenated, they are at the end.
                
                # Simplified Validation: Just pick the LAST positive event as target.
                if len(pos_events) > 1:
                    self.samples.append((i, len(pos_events) - 1))
                # If no positives, skip or placeholder? 
                
        # We don't need raw_data anymore to save RAM, unless needed for something else?
        # self.data = raw_data 
        del raw_data
        
        print(f"Generated {len(self.samples)} samples. Optimized structure ready.")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # 1. Retrieve Pre-processed Metadata
        user_idx_in_list, target_idx = self.samples[idx]
        
        # Access the user's positive sequence and Real User ID
        user_seq = self.user_positives[user_idx_in_list]
        real_user_id = self.user_ids[user_idx_in_list]
        
        # 2. Get Target
        target_event = user_seq[target_idx]
        pos_item_idx = target_event[0]
        pos_weight = target_event[1]
        
        # 3. Get History (Efficient Slice)
        # History is everything before target in this pre-filtered list
        # Take last N immediately
        start_idx = max(0, target_idx - self.max_history)
        recent_history = user_seq[start_idx : target_idx]
        
        # No filtering needed here anymore!
        
        if len(recent_history) == 0:
            hist_indices = [0]
            hist_weights = [0.0]
            hist_mask = [0.0]
        else:
            hist_indices = [e[0] for e in recent_history]
            hist_weights = [float(e[1]) for e in recent_history]
            hist_mask = [1.0] * len(hist_indices)
            
        # Return REAL User ID, not the list index
        
        return {
            'user_idx': torch.tensor(real_user_id, dtype=torch.long),
            'item_idx': torch.tensor(pos_item_idx, dtype=torch.long),
            'weight': torch.tensor(pos_weight, dtype=torch.float),
            'hist_indices': torch.tensor(hist_indices, dtype=torch.long),
            'hist_weights': torch.tensor(hist_weights, dtype=torch.float),
            'hist_mask': torch.tensor(hist_mask, dtype=torch.float)
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
