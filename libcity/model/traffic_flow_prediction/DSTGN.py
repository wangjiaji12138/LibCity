"""
DSTGN: Dual Spatio-Temporal Graph Network (Clean Baseline)

架构：Encoder -> [T-Attn x N] -> [S-Attn x N] -> Output(proj)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from logging import getLogger

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class SpatialPrototypeModule(nn.Module):
    """
    空间范式原型模块

    功能：
    1. 构建M个可学习的原型（空间范式）
    2. 根据节点特征分配节点到原型（Sinkhorn-Knopp 正则化）
    3. 原型和编码器联合更新（无 detach）
    """

    def __init__(self, num_nodes: int, model_dim: int,
                 num_prototypes: int = 16, prototype_dim: int = 32,
                 temperature: float = 0.1, geo_adj: Tensor = None,
                 sinkhorn_iterations: int = 3, sinkhorn_epsilon: float = 0.03):
        super().__init__()
        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)

        self.proto_proj = nn.Linear(model_dim, prototype_dim)
        self.geo_adj = geo_adj

    def _sinkhorn_knopp(self, logits: Tensor) -> Tensor:
        """Sinkhorn-Knopp 算法：让分配矩阵按行和列均匀分布，使不同原型被均匀使用"""
        Q = logits
        for _ in range(self.sinkhorn_iterations):
            Q = Q - torch.logsumexp(Q, dim=-1, keepdim=True)
            Q = Q - torch.logsumexp(Q, dim=-2, keepdim=True)
        return torch.exp(Q)

    def forward(self, node_features: Tensor):
        B, T, N, D = node_features.shape
        node_features_flat = node_features.reshape(B * T, N, D)

        node_proj = self.proto_proj(node_features_flat)
        node_proj = F.normalize(node_proj, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        proto_logits = torch.matmul(node_proj, prototypes_norm.transpose(0, 1)) / self.temperature

        proto = self._sinkhorn_knopp(proto_logits)

        return self.prototypes, proto.reshape(B, T, N, self.num_prototypes)

    def get_contrastive_loss(self, node_features: Tensor) -> Tensor:
        B, T, N, D = node_features.shape
        node_features_flat = node_features.reshape(B * T, N, D)

        node_proj = self.proto_proj(node_features_flat)
        node_proj = F.normalize(node_proj, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        sim = torch.matmul(node_proj, prototypes_norm.transpose(0, 1)) / self.temperature
        pos_sim = sim.max(dim=-1)[0]
        exp_sim = torch.exp(sim)
        loss = -torch.log(pos_sim / (exp_sim.sum(dim=-1) + 1e-8))

        return loss.mean()

    def get_sinkhorn_reg_loss(self, proto: Tensor) -> Tensor:
        """Sinkhorn 正则化损失：促进原型均匀分布"""
        B, T, N, M = proto.shape
        proto_usage = proto.reshape(B * T * N, M).mean(dim=0) + 1e-8
        uniform = torch.ones_like(proto_usage) / self.num_prototypes
        return F.kl_div(proto_usage.log(), uniform, reduction='batchmean')


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

    当前恢复 prototype 分支，用于对比实验。
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.num_nodes = data_feature.get('num_nodes')
        self._scaler = data_feature.get('scaler')
        self._logger = getLogger()

        self.in_window = config.get('input_window', 24)
        self.out_window = config.get('output_window', 24)

        self.input_embedding_dim = config.get('input_embedding_dim', 64)
        self.spatial_embedding_dim = config.get('spatial_embedding_dim', 32)
        self.tod_embedding_dim = config.get('tod_embedding_dim', 32)
        self.dow_embedding_dim = config.get('dow_embedding_dim', 32)
        self.steps_per_day = config.get('steps_per_day', 48)

        self.model_dim = (
            self.input_embedding_dim +
            self.spatial_embedding_dim +
            self.tod_embedding_dim +
            self.dow_embedding_dim
        )

        self.feed_forward_dim = config.get('feed_forward_dim', 256)
        self.num_heads = config.get('num_heads', 4)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        self.out_dim = self.data_feature.get('output_dim', 1)
        self.device = config.get('device', torch.device('cpu'))

        self.num_prototypes = config.get('num_prototypes', 8)
        self.prototype_dim = config.get('prototype_dim', 64)
        self.prototype_temperature = config.get('prototype_temperature', 1)
        self.prototype_loss_weight = config.get('prototype_loss_weight', 0.01)

        self.geo_adj = self._build_geo_adj(data_feature)

        self._logger.info(
            f'DSTGN | nodes={self.num_nodes}, model_dim={self.model_dim}, '
            f'(input={self.input_embedding_dim}, spatial={self.spatial_embedding_dim}, '
            f'tod={self.tod_embedding_dim}, dow={self.dow_embedding_dim}), '
            f'layers={self.num_layers}'
        )
        self._logger.info(
            f'Prototype | M={self.num_prototypes}, dim={self.spatial_embedding_dim}, '
            f'assign_temp={self.prototype_temperature}, proto_loss={self.prototype_loss_weight}'
        )

        self.input_proj = nn.Linear(1, self.input_embedding_dim)
        self.tod_proj = nn.Linear(1, self.tod_embedding_dim)
        self.dow_proj = nn.Linear(1, self.dow_embedding_dim)

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
        nn.init.xavier_uniform_(self.node_emb)

        self.prototype_module = SpatialPrototypeModule(
            num_nodes=self.num_nodes,
            model_dim=self.model_dim,
            num_prototypes=self.num_prototypes,
            prototype_dim=self.spatial_embedding_dim,
            temperature=self.prototype_temperature,
            geo_adj=self.geo_adj
        )

        # 时间注意力层（纯时间建模）
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, self.feed_forward_dim,
                              self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        # 空间注意力层（在 proto 注入后，全部使用 D+M 维度）
        self.attn_layers_s = nn.ModuleList([
            SelfAttentionLayer(self.model_dim + self.num_prototypes, self.feed_forward_dim,
                              self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        # 空间注意力块输出投影: D+M -> D
        self.spatial_out_proj = nn.Linear(self.model_dim + self.num_prototypes, self.model_dim)

        self.temporal_proj = nn.Conv1d(self.out_dim, self.out_window, kernel_size=1)
        self.output_proj = nn.Linear(self.model_dim, self.out_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _build_geo_adj(self, data_feature):
        adj_mx = data_feature.get('adj_mx')
        if adj_mx is not None:
            adj = torch.from_numpy(adj_mx).float()
            return (adj > 0).float()
        return None

    def forward(self, batch: dict, return_prototype_info: bool = False):
        # ==============================================================
        # 输入: batch['X'] shape = (B, T, N, 3)
        # 其中 3 = [value, time_of_day, day_of_week]
        # ==============================================================
        B, T, N, _ = batch['X'].shape

        # 特征嵌入
        # X[..., 0:1]: (B, T, N, 1) -> (B, T, N, input_embedding_dim)
        x_val = self.input_proj(batch['X'][..., 0:1])
        # X[..., 1:2]: (B, T, N, 1) -> (B, T, N, tod_embedding_dim)
        x_tod = self.tod_proj(batch['X'][..., 1:2])
        # X[..., 2:3]: (B, T, N, 1) -> (B, T, N, dow_embedding_dim)
        x_dow = self.dow_proj(batch['X'][..., 2:3])
        # node_emb: (N, spatial_embedding_dim) -> (B, T, N, spatial_embedding_dim)
        spatial_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        # 拼接初始特征: (B, T, N, model_dim)
        # model_dim = input_embedding_dim + spatial_embedding_dim + tod_embedding_dim + dow_embedding_dim
        x = torch.cat([x_val, spatial_emb, x_tod, x_dow], dim=-1)

        # ==============================================================
        # 空间原型模块 (Spatial Prototype Module)
        # ==============================================================
        prototypes, proto = self.prototype_module(x)
        proto_info = {
            'prototypes': prototypes,
            'proto': proto,
            'embedded_features': x,
        }

        # ==============================================================
        # 时间注意力块（Temporal Attention Block）
        # 3层纯时间建模: (B, T, N, D)
        # ==============================================================
        for attn_t in self.attn_layers_t:
            x = attn_t(x, dim=1)

        # ==============================================================
        # 注入原型信息（Inject Prototype Information）
        # proto: (B, T, N, M) 拼接到 x: (B, T, N, D+M)
        # ==============================================================
        x = torch.cat([x, proto], dim=-1)

        # ==============================================================
        # 空间注意力块（Spatial Attention Block）
        # 3层空间建模: (B, T, N, D+M)
        # ==============================================================
        for attn_s in self.attn_layers_s:
            x = attn_s(x, dim=2)

        # 投影回 D 维度: (B, T, N, D+M) -> (B, T, N, D)
        x = self.spatial_out_proj(x)

        # ==============================================================
        # 输出映射 (Output Projection)
        # ==============================================================

        # Step 1: 转置 (B, T, N, D) -> (B, N, T, D)
        x = x.transpose(1, 2)

        # Step 2: 重塑 (B, N, T, D) -> (B*N, T, D)
        x = x.reshape(B * N, T, self.model_dim)

        # Step 3: 特征投影 (B*N, T, D) -> (B*N, T, out_dim)
        # output_proj: Linear(D -> out_dim)
        x = self.output_proj(x.reshape(-1, self.model_dim))
        x = x.view(B * N, T, self.out_dim)

        # Step 4: 时间投影 (B*N, T, out_dim) -> (B*N, T, out_window)
        # temporal_proj: Conv1d(in_window, out_window, kernel_size=1)
        # 输入格式 (B*N, C, L) = (B*N, out_dim, T)
        x = x.permute(0, 2, 1)  # (B*N, T, out_dim) -> (B*N, out_dim, T)
        x = self.temporal_proj(x)  # (B*N, out_dim, T) -> (B*N, out_window, T)

        # Step 5: 重塑回四维 (B*N, out_window, T) -> (B, T, N, out_window)
        x = x.permute(0, 2, 1)  # (B*N, out_window, T) -> (B*N, T, out_window)
        x = x.reshape(B, N, T, self.out_window)
        x = x.permute(0, 2, 1, 3)  # (B, N, T, out_window) -> (B, T, N, out_window)

        if return_prototype_info:
            return x, proto_info
        return x

    def predict(self, batch: dict) -> Tensor:
        x = batch['X'][:, :self.in_window]
        return self.forward({'X': x})

    def calculate_loss(self, batch: dict) -> Tensor:
        y_pred, proto_info = self.forward(batch, return_prototype_info=True)

        y_true = batch['y'][:, :self.out_window]
        y_pred_for_loss = y_pred[..., :self.out_dim]
        y_true_for_loss = y_true[..., :self.out_dim]
        y_pred_inv = self._scaler.inverse_transform(y_pred_for_loss)
        y_true_inv = self._scaler.inverse_transform(y_true_for_loss)
        pred_loss = loss.masked_mae_torch(y_pred_inv, y_true_inv)

        proto = proto_info['proto']
        contrastive_loss = self.prototype_module.get_contrastive_loss(proto_info['embedded_features'])
        sinkhorn_reg_loss = self.prototype_module.get_sinkhorn_reg_loss(proto)

        hard_proto = torch.argmax(proto, dim=-1)
        proto_counts = torch.bincount(hard_proto.view(-1), minlength=self.num_prototypes).float()
        proto_ratio = proto_counts / proto_counts.sum()
        self._last_losses = {
            'pred': pred_loss.item(),
            'contrastive': contrastive_loss.item(),
            'sinkhorn_reg': sinkhorn_reg_loss.item(),
            'proto_loss_weight': self.prototype_loss_weight,
            'proto_dist': proto_ratio.cpu().tolist()
        }

        total_loss = pred_loss
        total_loss = total_loss + self.prototype_loss_weight * contrastive_loss
        total_loss = total_loss + self.prototype_loss_weight * sinkhorn_reg_loss

        return total_loss

    def get_last_losses(self) -> dict:
        return getattr(self, '_last_losses', {})
