import torch
import torch.nn as nn
import torch.nn.functional as F

class RLPolicy(nn.Module):
    """
    Scoring model for candidate actions.
    Matches 384-d embeddings from all-MiniLM-L6-v2.
    """

    def __init__(self, embed_dim=384, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: shape (K, embed_dim)
        returns: (K, 1) scores
        """
        h = F.relu(self.fc1(x))
        return self.fc2(h)
