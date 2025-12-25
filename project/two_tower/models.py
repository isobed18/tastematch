import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from . import config
except ImportError:
    import config

class ItemTower(nn.Module):
    def __init__(self, num_items, embedding_dim, latent_dim, text_emb_path, genome_emb_path, genre_emb_path):
        super().__init__()
        
        # 1. ID Embedding
        self.id_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 2. Load Static Features
        text_matrix = np.load(text_emb_path).astype(np.float32)
        genome_matrix = np.load(genome_emb_path).astype(np.float32)
        genre_matrix = np.load(genre_emb_path).astype(np.float32)
        
        # Register buffers
        self.register_buffer("text_matrix", torch.from_numpy(text_matrix))
        self.register_buffer("genome_matrix", torch.from_numpy(genome_matrix))
        self.register_buffer("genre_matrix", torch.from_numpy(genre_matrix))
        
        # 3. Projections
        self.text_proj = nn.Linear(text_matrix.shape[1], config.PROJECTION_DIM)
        self.genome_proj = nn.Linear(genome_matrix.shape[1], config.PROJECTION_DIM)
        self.genre_proj = nn.Linear(genre_matrix.shape[1], config.PROJECTION_DIM)
        
        # 4. Fusion
        fusion_dim = embedding_dim + 3 * config.PROJECTION_DIM
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, latent_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.id_embedding.weight)
        
    def forward(self, item_indices):
        id_emb = self.id_embedding(item_indices)
        text_vec = self.text_matrix[item_indices]
        genome_vec = self.genome_matrix[item_indices]
        genre_vec = self.genre_matrix[item_indices]
        
        p_text = F.relu(self.text_proj(text_vec))
        p_genome = F.relu(self.genome_proj(genome_vec))
        p_genre = F.relu(self.genre_proj(genre_vec))
        
        concated = torch.cat([id_emb, p_text, p_genome, p_genre], dim=-1)
        vector = self.fusion_mlp(concated)
        return F.normalize(vector, p=2, dim=-1)


class UserTower(nn.Module):
    def __init__(self, num_users, embedding_dim, latent_dim, item_tower):
        super().__init__()
        self.id_embedding = nn.Embedding(num_users, embedding_dim)
        
        # CRITICAL CHANGE: We save the item_tower to use its forward pass
        self.item_tower = item_tower
        
        # Input to fusion is UserID + Pooled History Vector (which is size latent_dim)
        fusion_dim = embedding_dim + latent_dim 
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, latent_dim)
        )
        
    def forward(self, user_indices, hist_indices, hist_weights, hist_mask):
        u_emb = self.id_embedding(user_indices)
        
        # CRITICAL FIX: Pass history IDs through the Item Tower to get Content Vectors
        # Flatten batch to pass through item tower
        b, l = hist_indices.shape
        flat_hist = hist_indices.view(-1)
        
        # Get content-rich vectors for history items
        flat_h_emb = self.item_tower(flat_hist)
        h_emb = flat_h_emb.view(b, l, -1)
        
        # Weighted Pooling
        weights = hist_weights.unsqueeze(-1)
        mask = hist_mask.unsqueeze(-1)
        
        weighted_sum = torch.sum(h_emb * weights * mask, dim=1)
        sum_weights = torch.sum(weights * mask, dim=1) + 1e-8
        pooled_history = weighted_sum / sum_weights
        
        concated = torch.cat([u_emb, pooled_history], dim=1)
        return F.normalize(self.fusion_mlp(concated), p=2, dim=1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, latent_dim):
        super().__init__()
        self.item_tower = ItemTower(
            num_items, embedding_dim, latent_dim,
            config.TEXT_EMBEDDINGS_PATH,
            config.GENOME_MATRIX_PATH,
            config.GENRE_MATRIX_PATH
        )
        
        self.user_tower = UserTower(
            num_users, embedding_dim, latent_dim,
            self.item_tower
        )
        
        self.temperature = config.TEMPERATURE
        
    def forward(self, batch):
        pos_item_vec = self.item_tower(batch['item_idx'])
        user_vec = self.user_tower(
            batch['user_idx'],
            batch['hist_indices'],
            batch['hist_weights'],
            batch['hist_mask']
        )
        return user_vec, pos_item_vec
        
    def compute_loss(self, user_vec, item_vec, weights):
        logits = torch.matmul(user_vec, item_vec.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
        return (loss_per_sample * weights).mean()