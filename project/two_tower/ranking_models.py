import torch
import torch.nn as nn

class RankerModel(nn.Module):
    def __init__(self, embedding_dim=256, feature_dim=0):
        super().__init__()
        
        # Inputs:
        # User Vec (256)
        # Item Vec (256)
        # Element-wise Product (256)
        # Element-wise Diff Abs (256)
        # Extra Features (Pop, Recency...)
        
        input_dim = embedding_dim * 4 + feature_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Logit
            # No Sigmoid here, use BCEWithLogitsLoss or Pairwise Logic
        )
        
    def forward(self, user_vec, item_vec, extra_features=None):
        # Interaction Features
        hadamard = user_vec * item_vec
        diff = torch.abs(user_vec - item_vec)
        
        # Concat
        features = [user_vec, item_vec, hadamard, diff]
        if extra_features is not None:
            features.append(extra_features)
            
        concat = torch.cat(features, dim=1)
        
        score = self.mlp(concat)
        return score.squeeze(-1)
