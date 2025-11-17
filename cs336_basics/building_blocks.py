import math

import einx
import torch
from einops import rearrange
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


class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        i_vec = torch.arange(max_seq_len, dtype=torch.float32)
        k_vec = torch.arange(d_k // 2, dtype=torch.float32)
        denom = torch.float_power(theta, -2.0 * k_vec / d_k)
        thetas = torch.outer(i_vec, denom)

        # shape: (max_seq_len, d_k//2)
        self.register_buffer("sin_thetas", torch.sin(thetas))
        self.register_buffer("cos_thetas", torch.cos(thetas))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # shape: (..., seq_len, d_k) ->
        x_split = rearrange(x, "... seq_len (k two) -> ... seq_len k two", two=2)
        top, bottom = x_split[..., 0], x_split[..., 1]

        sin_values, cos_values = self.sin_thetas[token_positions, :], self.cos_thetas[token_positions, :]
        x_prime_top = top * cos_values - bottom * sin_values
        x_prime_bottom = top * sin_values + bottom * cos_values

        x_prime = torch.stack((x_prime_top, x_prime_bottom), dim=-1)
        return rearrange(x_prime, "... seq_len k two -> ... seq_len (k two)", two=2)
