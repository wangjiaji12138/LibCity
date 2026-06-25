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


def _info_nce_from_assignments(assignments: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if assignments.ndim == 3:
        assignments = assignments.unsqueeze(0)
    if assignments.ndim != 4:
        raise ValueError(f'assignments must be (B,T,N,M) or (S,N,M), got {assignments.shape}')

    b, t, n, m = assignments.shape
    z = F.normalize(assignments, p=2, dim=-1)
    z = z.reshape(b * t, n, m)
    sim = torch.matmul(z, z.transpose(1, 2)) / temperature
    labels = torch.arange(n, device=z.device).unsqueeze(0).expand(b * t, -1)
    return F.cross_entropy(sim.reshape(-1, n), labels.reshape(-1))


class SpatialPrototypeModule(nn.Module):
    """
    空间范式原型模块

    功能：
    1. 构建M个可学习的原型（空间范式）
    2. 根据节点特征分配节点到原型
    3. 使用原型增强节点表示
    """

    def __init__(self, num_nodes: int, model_dim: int,
                 num_prototypes: int = 16, prototype_dim: int = 32,
                 temperature: float = 0.1, geo_adj: Tensor = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.temperature = temperature

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)

        self.assign_proj = nn.Linear(model_dim, prototype_dim)
        self.reconstruct_proj = nn.Linear(prototype_dim, model_dim)
        self.geo_adj = geo_adj
        self.stop_gradient = True

    def forward(self, node_features: Tensor):
        B, T, N, D = node_features.shape
        node_features_flat = node_features.reshape(B * T, N, D)

        node_proj = self.assign_proj(node_features_flat)
        node_proj = F.normalize(node_proj, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        assign_logits = torch.matmul(node_proj, prototypes_norm.transpose(0, 1)) / self.temperature
        assignments_flat = F.softmax(assign_logits, dim=-1)

        if self.stop_gradient:
            prototypes_stop = prototypes_norm.detach()
        else:
            prototypes_stop = prototypes_norm

        prototype_features_flat = torch.matmul(assignments_flat, prototypes_stop)
        reconstructed_flat = self.reconstruct_proj(prototype_features_flat)
        enhanced_flat = node_features_flat + 0.1 * reconstructed_flat

        return enhanced_flat.reshape(B, T, N, D), {
            'enhanced_features': enhanced_flat.reshape(B, T, N, D),
            'assignments': assignments_flat.reshape(B, T, N, self.num_prototypes),
        }

    def get_contrastive_loss(self, node_features: Tensor) -> Tensor:
        B, T, N, D = node_features.shape
        node_features_flat = node_features.reshape(B * T, N, D)

        node_proj = self.assign_proj(node_features_flat).detach()
        node_proj = F.normalize(node_proj, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        sim = torch.matmul(node_proj, prototypes_norm.transpose(0, 1)) / self.temperature
        pos_sim = sim.max(dim=-1)[0]
        exp_sim = torch.exp(sim)
        loss = -torch.log(pos_sim / (exp_sim.sum(dim=-1) + 1e-8))

        return loss.mean()

    def get_adjacency_consistency_loss(self, assignments: Tensor) -> Tensor:
        if self.geo_adj is None:
            return torch.tensor(0.0, device=assignments.device)

        B, T, N, M = assignments.shape
        assignments_flat = assignments.reshape(B * T, N, M)

        adj = self.geo_adj.to(assignments.device)
        diff = torch.cdist(assignments_flat, assignments_flat, p=2) ** 2
        mask = (adj > 0).float()
        return (diff * mask).sum() / (mask.sum() + 1e-8)

    def get_distribution_reg_loss(self, assignments: Tensor) -> Tensor:
        B, T, N, M = assignments.shape
        proto_usage = assignments.reshape(B * T * N, M).mean(dim=0) + 1e-8
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

        self.input_embedding_dim = config.get('input_embedding_dim', 96)
        self.spatial_embedding_dim = config.get('spatial_embedding_dim', 96)
        self.tod_embedding_dim = config.get('tod_embedding_dim', 16)
        self.dow_embedding_dim = config.get('dow_embedding_dim', 16)
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
        self.use_mixed_proj = config.get('use_mixed_proj', True)
        self.out_dim = self.data_feature.get('output_dim', 1)
        self.device = config.get('device', torch.device('cpu'))

        self.num_prototypes = config.get('num_prototypes', 8)
        self.prototype_dim = config.get('prototype_dim', 128)
        self.prototype_temperature = config.get('prototype_temperature', 0.5)
        self.info_nce_temperature = config.get('info_nce_temperature', 0.07)
        self.prototype_loss_weight = config.get('prototype_loss_weight', 0.0)
        self.entropy_loss_weight = config.get('entropy_loss_weight', 0.0)

        self.geo_adj = self._build_geo_adj(data_feature)

        self._logger.info(
            f'DSTGN | nodes={self.num_nodes}, model_dim={self.model_dim}, '
            f'(input={self.input_embedding_dim}, spatial={self.spatial_embedding_dim}, '
            f'tod={self.tod_embedding_dim}, dow={self.dow_embedding_dim}), '
            f'layers={self.num_layers}, mixed_proj={self.use_mixed_proj}'
        )
        self._logger.info(
            f'Prototype | M={self.num_prototypes}, dim={self.prototype_dim}, '
            f'assign_temp={self.prototype_temperature}, nce_temp={self.info_nce_temperature}, '
            f'proto_loss={self.prototype_loss_weight}, entropy={self.entropy_loss_weight}'
        )

        self.input_proj = nn.Linear(1, self.input_embedding_dim)
        self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
        nn.init.xavier_uniform_(self.node_emb)

        self.prototype_module = SpatialPrototypeModule(
            num_nodes=self.num_nodes,
            model_dim=self.spatial_embedding_dim,
            num_prototypes=self.num_prototypes,
            prototype_dim=self.prototype_dim,
            temperature=self.prototype_temperature,
            geo_adj=self.geo_adj
        )

        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, self.feed_forward_dim,
                               self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        self.attn_layers_s = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, self.feed_forward_dim,
                               self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

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
        B, T, N, _ = batch['X'].shape

        x_val = self.input_proj(batch['X'][..., 0:1])
        tod_idx = (batch['X'][..., 1] * self.steps_per_day).long()
        x_tod = self.tod_embedding(tod_idx)
        dow_idx = batch['X'][..., 2].long()
        x_dow = self.dow_embedding(dow_idx)
        spatial_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        x = torch.cat([x_val, spatial_emb, x_tod, x_dow], dim=-1)

        spatial_features = x[..., self.input_embedding_dim:self.input_embedding_dim + self.spatial_embedding_dim]
        B, T, N, D_spatial = spatial_features.shape
        spatial_features_flat = spatial_features.reshape(B * T, N, D_spatial)
        enhanced_spatial, proto_info = self.prototype_module(spatial_features)
        proto_info = {
            'enhanced_features': proto_info['enhanced_features'],
            'assignments': proto_info['assignments'],
        }

        x = torch.cat([x_val, enhanced_spatial, x_tod, x_dow], dim=-1)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)

        if self.use_mixed_proj:
            x = x.transpose(1, 2)
            x = x.reshape(B, self.num_nodes, self.in_window * self.model_dim)
            x = self.output_proj(x).view(B, self.num_nodes, self.out_window, self.out_dim)
            x = x.transpose(1, 2)
        else:
            x = x.transpose(1, 3)
            x = self.temporal_proj(x)
            x = x.transpose(1, 3)
            x = self.output_proj(x)

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

        assignments = proto_info['assignments']
        info_nce = _info_nce_from_assignments(assignments, self.info_nce_temperature)
        contrastive_loss = self.prototype_module.get_contrastive_loss(proto_info['enhanced_features'])
        entropy_loss = self.prototype_module.get_distribution_reg_loss(assignments)

        hard_assign = torch.argmax(assignments, dim=-1)
        proto_counts = torch.bincount(hard_assign.view(-1), minlength=self.num_prototypes).float()
        proto_ratio = proto_counts / proto_counts.sum()
        self._last_losses = {
            'pred': pred_loss.item(),
            'info_nce': info_nce.item(),
            'contrastive': contrastive_loss.item(),
            'entropy': entropy_loss.item(),
            'proto_loss_weight': self.prototype_loss_weight,
            'entropy_loss_weight': self.entropy_loss_weight,
            'proto_dist': proto_ratio.cpu().tolist()
        }

        total_loss = pred_loss
        total_loss = total_loss + self.prototype_loss_weight * info_nce
        total_loss = total_loss + self.entropy_loss_weight * entropy_loss

        return total_loss

    def get_last_losses(self) -> dict:
        return getattr(self, '_last_losses', {})
