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
    2. 根据节点特征分配节点到原型（Sinkhorn-Knopp 正则化 + 地理约束）
    3. 原型和编码器联合更新（无 detach）
    """

    def __init__(self, num_nodes: int, model_dim: int,
                 num_prototypes: int = 16, prototype_dim: int = 32,
                 temperature: float = 0.1, geo_adj: Tensor = None,
                 sinkhorn_iterations: int = 3, sinkhorn_epsilon: float = 0.03,
                 geo_smooth_weight: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.geo_smooth_weight = geo_smooth_weight

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)

        self.proto_proj = nn.Linear(model_dim, prototype_dim)
        self.geo_adj = geo_adj

    def _sinkhorn_knopp(self, logits: Tensor, geo_adj: Tensor = None) -> Tensor:
        """
        Sinkhorn-Knopp 算法：让分配矩阵按行和列均匀分布，使不同原型被均匀使用。
        如果提供 geo_adj，则在每次迭代后对相邻节点进行加权平滑。
        """
        Q = logits
        for i in range(self.sinkhorn_iterations):
            Q = Q - torch.logsumexp(Q, dim=-1, keepdim=True)
            Q = Q - torch.logsumexp(Q, dim=-2, keepdim=True)
            if geo_adj is not None and i < self.sinkhorn_iterations - 1:
                Q = self._geo_smooth(Q, geo_adj)
        return torch.exp(Q)

    def _geo_smooth(self, Q: Tensor, geo_adj: Tensor) -> Tensor:
        """
        地理平滑：对 Sinkhorn 分配矩阵进行空间平滑
        让相邻节点倾向于有相似的原型分配

        Q: (B*T, N, M) - Sinkhorn 分配
        geo_adj: (N, N) - 二值邻接矩阵
        """
        BT, N, M = Q.shape
        geo_adj_2d = geo_adj.to(Q.device)

        deg = geo_adj_2d.sum(dim=-1, keepdim=True).clamp(min=1)
        D_inv = 1.0 / deg
        W = (geo_adj_2d * D_inv).unsqueeze(0).expand(BT, N, N)

        Q_smoothed = (Q.transpose(-2, -1) @ W.transpose(-2, -1)).transpose(-2, -1)
        alpha = self.geo_smooth_weight
        Q = (1 - alpha) * Q + alpha * Q_smoothed
        return Q
 
    def forward(self, node_features: Tensor):
        B, T, N, D = node_features.shape
        node_features_flat = node_features.reshape(B * T, N, D)

        node_proj = self.proto_proj(node_features_flat)
        node_proj = F.normalize(node_proj, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        proto_logits = torch.matmul(node_proj, prototypes_norm.transpose(0, 1)) / self.temperature

        proto = self._sinkhorn_knopp(proto_logits, self.geo_adj)

        return self.prototypes, proto.reshape(B, T, N, self.num_prototypes)

    def get_contrastive_loss(self, node_features: Tensor) -> Tensor:
        B, T, N, D = node_features.shape
        node_features_flat = node_features.reshape(B * T, N, D)

        node_proj = self.proto_proj(node_features_flat)
        node_proj = F.normalize(node_proj, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        sim = torch.matmul(node_proj, prototypes_norm.transpose(0, 1)) / self.temperature
        
        sim = torch.clamp(sim, min=-50, max=50)
        exp_sim = torch.exp(sim)
        
        pos_sim = sim.max(dim=-1)[0]
        pos_sim = torch.clamp(pos_sim, min=-50, max=50)
        
        loss = -pos_sim + torch.log(exp_sim.sum(dim=-1) + 1e-8)

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
        B, T, D = query.shape

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(B * T, self.num_heads, self.head_dim)
        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(B * T, self.num_heads, self.head_dim)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(B * T, self.num_heads, self.head_dim)

        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim ** 0.5

        if self.mask:
            tgt_len = query.shape[-2]
            src_len = key.shape[-1]
            m = torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~m, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value

        out = out.view(B, self.num_heads, T, self.head_dim).transpose(1, 2).contiguous().view(B, T, D)
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

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            residual = x
            x = self.attn(x, x, x)
            x = self.dropout1(x)
            x = self.ln1(residual + x)
            residual = x
            x = self.feed_forward(x)
            x = self.dropout2(x)
            x = self.ln2(residual + x)
            return x
        elif x.dim() == 4:
            B, T, N, D = x.shape
            x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
            residual = x
            x = self.attn(x, x, x)
            x = self.dropout1(x)
            x = self.ln1(residual + x)
            residual = x
            x = self.feed_forward(x)
            x = self.dropout2(x)
            x = self.ln2(residual + x)
            x = x.reshape(B, N, T, D).permute(0, 2, 1, 3)
            return x
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}, expected 3 or 4")


class GraphConvLayer(nn.Module):
    """
    图卷积层：使用对称归一化拉普拉斯进行空间信息聚合

    输入：
        x: (B, T, N, D) - 节点特征
        adj: (N, N) 或 (B, T, N, N) - 动态加权邻接矩阵
    输出：
        (B, T, N, D) - 聚合后的特征
    """

    def __init__(self, model_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        B, T, N, D = x.shape
        residual = x

        x = x.transpose(1, 2).reshape(B * T, N, D)

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).unsqueeze(0).expand(B, T, N, N)
        adj = adj.reshape(B * T, N, N)

        A_hat = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
        deg = A_hat.sum(dim=-1, keepdim=True).clamp(min=1)
        A_norm = A_hat / deg
        x = A_norm @ x
        x = x.reshape(B, T, N, D)
        
        x = self.linear(x)
        x = x + residual
        x = F.layer_norm(x, (D,))
        x = self.dropout(F.relu(x))
        return x


class DSTGN(AbstractTrafficStateModel):
    """
    双重时空图网络

    架构：特征嵌入 -> 时间注意力 -> 原型生成 + 动态邻接矩阵 -> 图卷积 -> 输出投影
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
        self.prototype_loss_weight = config.get('prototype_loss_weight', 0.5)
        self.geo_smooth_weight = config.get('geo_smooth_weight', 0.1)

        self.geo_adj = self._build_geo_adj(data_feature)

        self._logger.info(
            f'DSTGN | nodes={self.num_nodes}, model_dim={self.model_dim}, '
            f'layers={self.num_layers}'
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
            prototype_dim=self.prototype_dim,
            temperature=self.prototype_temperature,
            geo_adj=self.geo_adj,
            geo_smooth_weight=self.geo_smooth_weight
        )

        self.time_linear = nn.Linear(self.model_dim, self.model_dim)
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, self.feed_forward_dim,
                              self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        self.graph_conv = GraphConvLayer(self.model_dim, self.dropout)
        self.skip_proj = nn.Linear(self.model_dim, self.model_dim)
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
            return adj
        return None

    def forward(self, batch: dict, return_prototype_info: bool = False):
        B, T, N, _ = batch['X'].shape

        x_val = self.input_proj(batch['X'][..., 0:1])
        x_tod = self.tod_proj(batch['X'][..., 1:2])
        x_dow = self.dow_proj(batch['X'][..., 2:3])
        spatial_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(B, T, -1, -1)
        x = torch.cat([x_val, spatial_emb, x_tod, x_dow], dim=-1)

        x = self.time_linear(x)
        skip = self.skip_proj(x)

        for attn_t in self.attn_layers_t:
            residual = x
            x = attn_t(x)
            s = self.skip_proj(x)
            skip = s + skip
            x = x + residual[:, :, :, -x.size(3):]

        prototypes, proto = self.prototype_module(x)
        proto_info = {
            'prototypes': prototypes,
            'proto': proto,
            'embedded_features': x,
        }

        sim = torch.matmul(proto, proto.transpose(-2, -1))
        sim = F.relu(sim)

        geo_adj = self.geo_adj
        if geo_adj is not None:
            geo_adj = geo_adj.to(proto.device)
            geo_adj_3d = geo_adj.unsqueeze(0).unsqueeze(0)
            dynamic_adj = (sim * geo_adj_3d).relu()
        else:
            dynamic_adj = sim

        BT = B * T
        dynamic_adj = dynamic_adj.reshape(BT, N, N)

        x = self.graph_conv(x, dynamic_adj)

        skip = self.skip_proj(x) + skip
        x = F.relu(skip)
        x = self.output_proj(x)
        output = x

        if return_prototype_info:
            return output, proto_info
        return output

    def predict(self, batch: dict) -> Tensor:
        return self.forward(batch, return_prototype_info=False)

    def calculate_loss(self, batch: dict) -> Tensor:
        y_pred, proto_info = self.forward(batch, return_prototype_info=True)

        y_true = batch['y']
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
