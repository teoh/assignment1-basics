import math

import einx
import torch
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


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.gain = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_times_x = einx.dot("... d_model, ... d_model -> ... c", x, x, c=1)
        rms = torch.sqrt(x_times_x / self.d_model + self.eps)
        result = x * self.gain / rms

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        self.w3 = Linear(self.d_model, self.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x) * x
