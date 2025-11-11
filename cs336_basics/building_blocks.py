import math

import torch

# import torch.nn as nn
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        sigma = math.sqrt(2.0 / (in_features + out_features))
        self.W = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(out_features, in_features), mean=0.0, std=sigma, a=-3, b=-3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embeds = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim), mean=0.0, std=1.0))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeds[token_ids]
