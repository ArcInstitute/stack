"""Attention building blocks for the StateICL models."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation used across the project."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply multi-head self-attention."""

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            mask = attn_mask
            while mask.ndim < 4:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1:
                mask = mask.expand(batch_size, -1, -1, -1)
            if mask.shape[1] == 1:
                mask = mask.expand(-1, self.n_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn = self.dropout(F.softmax(attn_scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        out = self.proj(out)

        if return_attn:
            return out, attn
        return out, None


class TabularAttentionLayer(nn.Module):
    """Single layer of tabular attention for gene expression modelling."""

    def __init__(
        self,
        token_dim: int,
        n_cells: int,
        n_hidden: int,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        cell_attn_dim = token_dim
        cell_attn_heads = 8
        # The model dimension must be divisible by the number of attention heads.
        # If not, find the largest valid number of heads smaller than or equal to n_heads.
        if cell_attn_dim % cell_attn_heads != 0:
            for h in range(min(n_heads, cell_attn_dim), 0, -1):
                if cell_attn_dim % h == 0:
                    cell_attn_heads = h
                    break
            else:  # This fallback should ideally not be reached if cell_attn_dim >= 1
                cell_attn_heads = 1
        self.cell_attn = MultiHeadAttention(cell_attn_dim, cell_attn_heads, dropout)
        self.cell_norm = nn.LayerNorm(cell_attn_dim)

        gene_attn_dim = n_hidden * token_dim
        self.gene_attn = MultiHeadAttention(gene_attn_dim, n_heads, dropout)
        self.gene_norm = nn.LayerNorm(gene_attn_dim)

        hidden_dim = token_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.Dropout(dropout),
        )
        self.mlp_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        x: torch.Tensor,
        gene_pos_emb: torch.Tensor,
        gene_attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, n_cells, n_genes, token_dim = x.shape

        x_cell = x.reshape(batch_size * n_cells, n_genes, token_dim)
        x_cell_with_pos = x_cell + gene_pos_emb.unsqueeze(0)
        cell_attn_out, _ = self.cell_attn(x_cell_with_pos)
        x_cell = self.cell_norm(x_cell + cell_attn_out)

        x = x_cell.reshape(batch_size, n_cells, n_genes, token_dim)
        x_gene = x.reshape(batch_size, n_cells, n_genes * token_dim)

        if return_attn:
            gene_attn_out, attn = self.gene_attn(x_gene, attn_mask=gene_attn_mask, return_attn=True)
        else:
            gene_attn_out, attn = self.gene_attn(x_gene, attn_mask=gene_attn_mask)
        x_gene = self.gene_norm(x_gene + gene_attn_out)

        x = x_gene.reshape(batch_size, n_cells, n_genes, token_dim)
        mlp_input = x.reshape(-1, token_dim)
        mlp_out = self.mlp(mlp_input)
        x = self.mlp_norm(mlp_input + mlp_out).reshape(batch_size, n_cells, n_genes, token_dim)

        return x, attn
