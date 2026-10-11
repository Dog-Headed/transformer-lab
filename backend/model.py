"""A trainable decoder, with explicit Q/K/V and causal attention weights."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import TrainConfig


class CausalAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_size = hidden_size // heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch, length, hidden = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, self.head_size).transpose(1, 2)
                   for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        weights = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        context = (weights @ v).transpose(1, 2).contiguous().view(batch, length, hidden)
        return self.projection(context), weights


class Block(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attention = CausalAttention(config.hidden_size, config.heads)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(nn.Linear(config.hidden_size, 4 * config.hidden_size),
                                 nn.GELU(), nn.Linear(4 * config.hidden_size, config.hidden_size))

    def forward(self, x):
        attended, weights = self.attention(self.norm1(x))
        x = x + attended
        return x + self.ffn(self.norm2(x)), weights


def parameter_count(config: TrainConfig, vocabulary_size: int):
    h = config.hidden_size
    return (vocabulary_size * h + config.sequence_length * h
            + config.layers * (12 * h * h + 13 * h) + 2 * h
            + h * vocabulary_size + vocabulary_size)


class TinyTransformer(nn.Module):
    def __init__(self, config: TrainConfig, vocabulary_size: int):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocabulary_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.sequence_length, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, vocabulary_size)

    def forward(self, tokens, *, return_attention=False):
        length = tokens.shape[1]
        if length > self.config.sequence_length:
            raise ValueError("上下文超过模型 sequence_length")
        x = self.token_embedding(tokens) + self.position_embedding(torch.arange(length, device=tokens.device))
        attentions = []
        for block in self.blocks:
            x, weights = block(x)
            if return_attention:
                attentions.append(weights)
        return self.lm_head(self.norm(x)), attentions
