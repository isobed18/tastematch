import torch
import torch.nn as nn
from . import config

class HybridNCF(nn.Module):
    def __init__(self, num_users, num_movies, genome_dim=config.GENOME_DIM, embedding_dim=config.EMBEDDING_DIM, layers=config.LAYERS, dropout=config.DROPOUT):
        super(HybridNCF, self).__init__()
        
        # 1. Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Initialize weights (Xavier/He initialization is good practice)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)

        # 2. Input Dimensions for MLP
        # Logic: 
        # Hybrid Item Vector = MovieEmb (64) + Genome (1128)
        # MLP Input = UserEmb (64) + Hybrid Item Vector
        # Total Input Dim = UserEmb + MovieEmb + Genome
        self.input_dim = embedding_dim + embedding_dim + genome_dim
        
        # 3. MLP Layers
        self.mlp_layers = nn.ModuleList()
        interaction_layers = []
        
        input_size = self.input_dim
        for output_size in layers:
            interaction_layers.append(nn.Linear(input_size, output_size))
            interaction_layers.append(nn.ReLU())
            interaction_layers.append(nn.Dropout(p=dropout))
            input_size = output_size
            
        self.mlp = nn.Sequential(*interaction_layers)
        
        # 4. Output Layer
        # Projects last layer to 1 output (Probability)
        self.output_layer = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_indices, movie_indices, content_features):
        """
        user_indices: (Batch,)
        movie_indices: (Batch,)
        content_features: (Batch, GenomeDim) - The Tag Genome vector
        """
        
        # A. Embeddings
        user_emb = self.user_embedding(user_indices) # (Batch, EmbDim)
        movie_emb = self.movie_embedding(movie_indices) # (Batch, EmbDim)
        
        # B. Creating Hybrid Item Vector (Conceptually)
        # But for MLP input, we just concat everything: User || Movie || Content
        
        # Concatenate: [User, Movie, Content]
        vector = torch.cat([user_emb, movie_emb, content_features], dim=-1)
        
        # C. MLP
        x = self.mlp(vector)
        
        # D. Output (Scaled Sigmoid)
        # Logits -> Sigmoid [0,1] -> Scale [0.5, 5.0]
        logits = self.output_layer(x)
        sigmoid_val = torch.sigmoid(logits)
        output = sigmoid_val * (config.MAX_RATING - config.MIN_RATING) + config.MIN_RATING
        
        return output.view(-1) # Flatten to (Batch,)
