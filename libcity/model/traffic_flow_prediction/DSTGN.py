"""
DSTGN: Dual Spatio-Temporal Graph Network

时空图预测模型，采用 GCN 和交替时空注意力双层堆叠架构。
- 空间图卷积（GCN）：稀疏图上高效的消息传递
- 时间/空间自注意力（参考 STAEformer）：Pre-LN Transformer 块
- 残差连接和归一化
"""

from __future__ import annotations

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Q: (B, ..., tgt_len, D), K/V: (B, ..., src_len, D)
        B = query.shape[0]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # 拆分多头：split 后 concat 到 batch 维度
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        # (H*B, ..., T, H) @ (H*B, ..., H, S) -> (H*B, ..., T, S)
        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim ** 0.5

        if self.mask:
            tgt_len = query.shape[-2]
            src_len = key.shape[-1]
            m = torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~m, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value

        # 合并多头
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


class GCN(nn.Module):
    """图卷积：消息传递聚合邻居特征"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_idx: Tensor) -> Tensor:
        # x: (B, N, C), edge_idx: (2, E)
        x = self.linear(x)
        src, tgt = edge_idx[0], edge_idx[1]
        B, N, C = x.shape
        out = torch.zeros_like(x)
        for b in range(B):
            out[b].index_add_(0, tgt, x[b, src])
        return out + self.bias


class SBlock(nn.Module):
    """空间块：GCN + LayerNorm + ReLU"""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.gcn = GCN(in_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_idx: Tensor) -> Tensor:
        x = self.gcn(x, edge_idx)
        x = self.norm(x)
        return self.dropout(F.relu(x))


class Encoder(nn.Module):
    """特征编码层"""
    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(feat_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, F, T, N) -> (B, H, T, N)
        B, F, T, N = x.shape
        x = x.permute(0, 3, 2, 1)  # (B, N, T, F)
        x = self.linear(x)
        x = self.norm(x)
        return x.permute(0, 3, 2, 1)


class STBlock(nn.Module):
    """时空块：GCN + 时间注意力 + GCN + 时间注意力"""
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, num_nodes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.s1 = SBlock(in_ch, hidden_ch, dropout)
        self.t1 = SelfAttentionLayer(hidden_ch, hidden_ch * 4, dropout=dropout)
        self.s2 = SBlock(hidden_ch, hidden_ch, dropout)
        self.t2 = SelfAttentionLayer(hidden_ch, hidden_ch * 4, dropout=dropout)
        self.proj = nn.Linear(hidden_ch, out_ch) if hidden_ch != out_ch else nn.Identity()
        self.norm = nn.LayerNorm([num_nodes, out_ch])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_idx: Tensor) -> Tensor:
        # x: (B, C, T, N) -> (B, C', T', N)
        x = x.permute(0, 2, 3, 1)  # (B, T, N, C)
        B, T, N, C = x.shape

        # 空间卷积
        x = x.reshape(B * T, N, C)
        x = self.s1(x, edge_idx)
        x = x.reshape(B, T, N, -1)

        # 时间注意力
        x = self.t1(x, dim=1)

        # 再次空间卷积
        x = x.reshape(B * T, N, -1)
        x = self.s2(x, edge_idx)
        x = x.reshape(B, T, N, -1)

        # 再次时间注意力
        x = self.t2(x, dim=1)

        x = self.proj(x)
        x = self.norm(x)
        return self.dropout(x.permute(0, 3, 1, 2))


class Decoder(nn.Module):
    """输出层"""
    def __init__(self, ch: int, num_nodes: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.tblock = SelfAttentionLayer(ch, ch * 4, dropout=dropout)
        self.proj = nn.Conv2d(ch, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm([num_nodes, ch])

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, N) -> (B, out_dim, 1, N)
        x = self.tblock(x.permute(0, 2, 3, 1), dim=1)
        x = self.norm(x)
        return self.proj(x.permute(0, 3, 1, 2)[:, :, -1:, :])


class DSTGN(AbstractTrafficStateModel):
    """
    双重时空图网络

    架构：Encoder -> [STBlock x N] -> Decoder
    - Encoder: 特征维度投影到隐藏维度
    - STBlock: GCN（空间）+ SelfAttention（时间），重复两次
    - Decoder: 时间注意力 + 投影到输出维度
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.feat_dim = self.data_feature.get('feature_dim', 1)
        self.out_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()

        self.hidden_dim = config.get('hidden_dim', 64)
        self.in_window = config.get('input_window', 12)
        self.out_window = config.get('output_window', 1)
        self.dropout = config.get('dropout', 0.1)
        self.num_layers = config.get('num_layers', 2)
        self.num_heads = config.get('num_heads', 4)
        self.device = config.get('device', torch.device('cpu'))

        adj_mx = data_feature['adj_mx']
        self.edge_idx = self._build_edge_idx(adj_mx).to(self.device)
        self._logger.info(f'DSTGN | nodes={self.num_nodes}, edges={self.edge_idx.shape[1]}')

        self.encoder = Encoder(self.feat_dim, self.hidden_dim)

        self.st_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            ch = self.hidden_dim * (2 ** i)
            self.st_blocks.append(
                STBlock(ch, ch, ch * 2, self.num_nodes, self.dropout)
            )

        decoder_ch = self.hidden_dim * (2 ** self.num_layers)
        self.decoder = Decoder(decoder_ch, self.num_nodes, self.out_dim, self.dropout)

        self.apply(self._init_weights)

    def _build_edge_idx(self, adj_mx):
        adj_mx = torch.from_numpy(adj_mx) if not isinstance(adj_mx, torch.Tensor) else adj_mx
        rows, cols = torch.where(adj_mx > 0)
        return torch.stack([rows, cols], dim=0).long()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, batch: dict) -> Tensor:
        # batch['X']: (B, T_in, N, F)
        x = batch['X'].permute(0, 3, 1, 2)  # (B, F, T, N)
        x = self.encoder(x)
        for block in self.st_blocks:
            x = block(x, self.edge_idx)
        return self.decoder(x)

    def predict(self, batch: dict) -> Tensor:
        x_cur = batch['X'][:, :self.in_window]
        return self.forward({'X': x_cur})

    def calculate_loss(self, batch: dict) -> Tensor:
        y_pred = self.predict(batch)
        y_true = batch['y'][:, :self.out_window]
        y_pred = self._scaler.inverse_transform(y_pred)
        y_true = self._scaler.inverse_transform(y_true)
        return loss.masked_mae_torch(y_pred, y_true, 0)
