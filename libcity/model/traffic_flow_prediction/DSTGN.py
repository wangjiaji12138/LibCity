"""
DSTGN: Dual Spatio-Temporal Graph Network

时空图预测模型，参考 STAEformer 架构：
- 先所有时间自注意力层，再所有空间自注意力层
- Pre-LN Transformer 块
- 原始值投影 + spatial embedding
- Mixed-projection 输出
"""

from __future__ import annotations

from logging import getLogger

import torch
import torch.nn as nn
from torch import Tensor

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class AttentionLayer(nn.Module):
    """在指定维度（-2）上执行多头自注意力。"""

    def __init__(self, model_dim: int, num_heads: int = 4, mask: bool = False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        B = query.shape[0]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim ** 0.5

        if self.mask:
            tgt_len = query.shape[-2]
            src_len = key.shape[-1]
            m = torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~m, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value

        out = torch.cat(torch.split(out, B, dim=0), dim=-1)
        return self.out_proj(out)


class SelfAttentionLayer(nn.Module):
    """Pre-LN Transformer 块：Multi-Head Attention + FFN"""

    def __init__(self, model_dim: int, feed_forward_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.1, mask: bool = False):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, dim: int = -2) -> Tensor:
        x = x.transpose(dim, -2)
        residual = x
        x = self.attn(x, x, x)
        x = self.dropout1(x)
        x = self.ln1(residual + x)

        residual = x
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = self.ln2(residual + x)

        return x.transpose(dim, -2)


class DSTGN(AbstractTrafficStateModel):
    """
    双重时空图网络，参考 STAEformer 架构。

    架构：Encoder -> [T-Attn x N] -> [S-Attn x N] -> Output(proj)
    - Encoder: 原始值投影 + spatial embedding
    - 时间层：所有层先在 T 维度做自注意力
    - 空间层：所有层再在 N 维度做自注意力
    - 输出：Mixed-projection
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.num_nodes = data_feature.get('num_nodes')
        self._scaler = data_feature.get('scaler')
        self._logger = getLogger()

        self.in_window = config.get('input_window', 12)
        self.out_window = config.get('output_window', 12)

        self.input_embedding_dim = config.get('input_embedding_dim', 96)
        self.spatial_embedding_dim = config.get('spatial_embedding_dim', 96)
        self.model_dim = self.input_embedding_dim + self.spatial_embedding_dim

        self.feed_forward_dim = config.get('feed_forward_dim', 256)
        self.num_heads = config.get('num_heads', 4)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        self.use_mixed_proj = config.get('use_mixed_proj', True)
        self.out_dim = self.data_feature.get('output_dim', 1)
        self.device = config.get('device', torch.device('cpu'))

        self._logger.info(
            f'DSTGN | nodes={self.num_nodes}, model_dim={self.model_dim}, '
            f'layers={self.num_layers}, mixed_proj={self.use_mixed_proj}'
        )

        # 原始值投影
        self.input_proj = nn.Linear(1, self.input_embedding_dim)

        # Spatial embedding
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
        nn.init.xavier_uniform_(self.node_emb)

        # 时间注意力层
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, self.feed_forward_dim,
                               self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        # 空间注意力层
        self.attn_layers_s = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, self.feed_forward_dim,
                               self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        # 输出投影
        if self.use_mixed_proj:
            self.output_proj = nn.Linear(
                self.in_window * self.model_dim,
                self.out_window * self.out_dim
            )
        else:
            self.temporal_proj = nn.Linear(self.in_window, self.out_window)
            self.output_proj = nn.Linear(self.model_dim, self.out_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, batch: dict) -> Tensor:
        B, T, N, _ = batch['X'].shape
        x = batch['X'][..., 0:1]  # (B, T, N, 1)

        # 原始值投影
        x = self.input_proj(x)  # (B, T, N, input_embedding_dim)

        # Spatial embedding
        spatial_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        x = torch.cat([x, spatial_emb], dim=-1)  # (B, T, N, model_dim)

        # 时间注意力
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)

        # 空间注意力
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)

        # 输出
        if self.use_mixed_proj:
            x = x.transpose(1, 2)  # (B, N, T, D)
            x = x.reshape(B, self.num_nodes, self.in_window * self.model_dim)
            x = self.output_proj(x).view(B, self.num_nodes, self.out_window, self.out_dim)
            x = x.transpose(1, 2)  # (B, T, N, out_dim)
        else:
            x = x.transpose(1, 3)
            x = self.temporal_proj(x)
            x = x.transpose(1, 3)
            x = self.output_proj(x)

        return x

    def predict(self, batch: dict) -> Tensor:
        x = batch['X'][:, :self.in_window]
        return self.forward({'X': x})

    def calculate_loss(self, batch: dict) -> Tensor:
        y_pred = self.predict(batch)
        y_true = batch['y'][:, :self.out_window]
        y_pred = y_pred[..., :self.out_dim]
        y_true = y_true[..., :self.out_dim]
        y_pred = self._scaler.inverse_transform(y_pred)
        y_true = self._scaler.inverse_transform(y_true)
        return loss.masked_mae_torch(y_pred, y_true, 0)
