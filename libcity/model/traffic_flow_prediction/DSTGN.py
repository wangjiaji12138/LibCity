"""
DSTGN: Dynamic Spatial-Temporal Graph Network

适配 LibCity 框架的时空图预测模型。

核心功能：
- 时空编码：同时建模空间依赖和时间动态
- Period Embedding：学习时段/周期的正常模式基线
- 空间侧：双向图卷积
- 时间侧：Multi-Head Self-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


def row_normalize(adjacency):
    """按行归一化邻接矩阵"""
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def expand_static_adjacency(A_static, num_replicas, device, dtype, num_nodes):
    """将 [N,N] 静态邻接扩展为 [num_replicas, N, N]"""
    if A_static is None:
        return torch.ones(num_replicas, num_nodes, num_nodes, device=device, dtype=dtype)
    A = torch.relu(A_static)
    A = (A + A.T) / 2.0
    return A.unsqueeze(0).expand(num_replicas, -1, -1).to(dtype=dtype)


class GraphAggregation(nn.Module):
    """图消息传递：聚合邻居节点特征"""

    def forward(self, x, adj):
        """
        Args:
            x: [B, C, N, T] 节点特征
            adj: [B, T, N, N] 邻接矩阵
        Returns:
            [B, C, N, T] 聚合后的特征
        """
        batch, channels, num_nodes, time_steps = x.shape
        x_reshaped = x.permute(0, 3, 1, 2).reshape(batch * time_steps, channels, num_nodes)
        adj_reshaped = adj.reshape(batch * time_steps, num_nodes, num_nodes)
        aggregated = torch.bmm(x_reshaped, adj_reshaped.transpose(1, 2))
        return aggregated.reshape(batch, time_steps, channels, num_nodes).permute(0, 2, 3, 1)


class BidirectionalGraphConvLayer(nn.Module):
    """双向图卷积，同时考虑正向和反向邻居的信息传播"""

    def __init__(self, channels, dropout=0.1, use_bid=True):
        super().__init__()
        self.use_bid = use_bid
        if use_bid:
            self.forward_conv = GraphConvLayer(channels, channels, dropout)
            self.backward_conv = GraphConvLayer(channels, channels, dropout)
        else:
            self.graph_conv = GraphConvLayer(channels, channels, dropout)

    def forward(self, enc, adjacency):
        if self.use_bid:
            forward_out = self.forward_conv(enc, adjacency)
            backward_adj = adjacency.transpose(-1, -2)
            backward_out = self.backward_conv(enc, backward_adj)
            return (forward_out + backward_out) / 2
        else:
            return self.graph_conv(enc, adjacency)


class GraphConvLayer(nn.Module):
    """基础图卷积层"""

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.aggregation = GraphAggregation()
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, enc, adj):
        agg = self.aggregation(enc, adj)
        agg_proj = self.proj(agg.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(agg_proj)


class SimpleSpatialBlock(nn.Module):
    """简化版空间块，不带邻接门控"""

    def __init__(self, model_dim, dropout=0.1, use_bid=True, use_sta=True, adaptive_embed_dim=64):
        super().__init__()
        self.model_dim = model_dim
        self.use_bid = use_bid
        self.use_sta = use_sta
        self.graph_conv = BidirectionalGraphConvLayer(model_dim, dropout, use_bid=use_bid)

    def forward(self, enc, A_static=None):
        batch, channels, num_nodes, time_steps = enc.shape
        device, dtype = enc.device, enc.dtype

        adjacency = expand_static_adjacency(
            A_static, batch * time_steps, device, dtype, num_nodes
        ).view(batch, time_steps, num_nodes, num_nodes)
        adjacency = row_normalize(adjacency)

        return enc + self.graph_conv(enc, adjacency)


class TemporalAttention(nn.Module):
    """时间自注意力层"""

    def __init__(self, model_dim, num_heads=4, dropout=0.1, feed_forward_dim=None):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        if feed_forward_dim is None:
            feed_forward_dim = model_dim * 4

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

        # 添加 LayerNorm 稳定训练
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        """
        Args:
            x: [B, C, N, T]
        Returns:
            [B, C, N, T]
        """
        B, C, N, T = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * N, T, C)

        # Pre-LN 架构（更稳定）
        x_flat = self.norm1(x_flat)
        residual = x_flat
        query = self.q_proj(x_flat)
        key = self.k_proj(x_flat)
        value = self.v_proj(x_flat)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)

        attn_score = query @ key / (self.head_dim ** 0.5)
        attn_score = F.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, B * N, dim=0), dim=-1)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = residual + out

        # Pre-LN for FFN
        out = self.norm2(out)
        residual = out
        out = self.ffn(out)
        out = self.dropout(out)
        out = residual + out

        return out.view(B, N, T, C).permute(0, 3, 1, 2)


class DSTGN(AbstractTrafficStateModel):
    """
    DSTGN: Dynamic Spatial-Temporal Graph Network

    适配 LibCity 框架的时空图预测模型。

    输入格式 (LibCity): [B, T, N, F]
    如果 F > 1，会使用时间特征 (time_of_day, day_of_week)
    如果 F == 1，只使用订单特征
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()

        # 从配置获取参数
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.ext_dim = self.data_feature.get('ext_dim', 0)  # 外部特征维度

        self.model_dim = config.get('model_dim', 128)
        self.hidden_dim = config.get('hidden_dim', self.model_dim)
        self.num_layers = config.get('num_layers', 2)
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.dropout = config.get('dropout', 0.1)

        # 可选参数
        self.use_bid = config.get('use_bid', True)

        # Embedding 维度分配 (根据是否有时间特征动态调整)
        if self.ext_dim >= 2:
            # 有时间特征：node_dim + order_dim + tod_dim + dow_dim
            self.node_dim = self.model_dim // 4
            self.order_dim = self.model_dim // 4
            self.tod_dim = self.model_dim // 4
            self.dow_dim = self.model_dim // 4
            self.use_time_embed = True
        else:
            # 无时间特征：node_dim + order_dim（更少的嵌入）
            self.node_dim = self.model_dim // 2
            self.order_dim = self.model_dim // 2
            self.tod_dim = 0
            self.dow_dim = 0
            self.use_time_embed = False

        assert self.model_dim == self.node_dim + self.order_dim + self.tod_dim + self.dow_dim

        # 投影层
        self.order_projection = nn.Linear(1, self.order_dim, bias=True)

        if self.use_time_embed:
            self.tod_embedding = nn.Embedding(24, self.tod_dim)
            self.dow_embedding = nn.Embedding(7, self.dow_dim)

        # 节点嵌入
        self.node_embedding = nn.Embedding(self.num_nodes, self.node_dim)

        # 时空交替层
        self.spatial_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.spatial_layers.append(
                SimpleSpatialBlock(
                    model_dim=self.model_dim,
                    dropout=self.dropout,
                    use_bid=self.use_bid,
                    use_sta=True,
                )
            )
            self.temporal_layers.append(
                TemporalAttention(
                    model_dim=self.model_dim,
                    num_heads=4,
                    dropout=self.dropout,
                )
            )

        # 输出层
        self.output_projection = nn.Linear(
            in_features=self.model_dim * self.input_window,
            out_features=self.output_window,
            bias=True
        )

        self._logger.info(f"DSTGN model built: num_nodes={self.num_nodes}, "
                         f"model_dim={self.model_dim}, num_layers={self.num_layers}, "
                         f"use_time_embed={self.use_time_embed}")

        self._init_parameters()

    def _init_parameters(self):
        """Xavier 权重初始化，防止数值爆炸"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, a=-0.1, b=0.1)

    def forward(self, x):
        """
        Args:
            x: [B, T, N, F] 其中 F = 1 (package) 或 F = 1 + time features

        Returns:
            torch.Tensor [B, T, N, 1] 其中 T = output_window
        """
        batch_size, time_steps, num_nodes, num_features = x.shape

        # 提取订单量特征
        order_feat = x[..., :1]  # [B, T, N, 1]

        # 订单量投影
        order_enc = self.order_projection(order_feat)  # [B, T, N, order_dim]

        # 节点嵌入
        node_ids = torch.arange(num_nodes, device=x.device).unsqueeze(0).unsqueeze(0)
        node_ids = node_ids.expand(batch_size, time_steps, -1)
        node_emb = self.node_embedding(node_ids)  # [B, T, N, node_dim]

        # 时间身份嵌入（如果有）
        if self.use_time_embed and num_features >= 3:
            time_feat = x[..., 1:3]  # [B, T, N, 2]
            hour_id = time_feat[..., 0].long().clamp(0, 23)
            day_id = time_feat[..., 1].long().clamp(0, 6)

            tod_emb = self.tod_embedding(hour_id)
            dow_emb = self.dow_embedding(day_id)

            # 拼接 node, order, tod, dow
            encoded = torch.cat([node_emb, order_enc, tod_emb, dow_emb], dim=-1)
        else:
            # 只使用 node 和 order
            encoded = torch.cat([node_emb, order_enc], dim=-1)

        # [B, T, N, C] -> [B, C, N, T]
        encoded = encoded.permute(0, 3, 2, 1)

        # 获取邻接矩阵
        adj_mx = self.data_feature.get('adj_mx')
        if adj_mx is not None:
            if isinstance(adj_mx, torch.Tensor):
                A_static = adj_mx.to(encoded.device)
            else:
                A_static = torch.from_numpy(adj_mx).to(encoded.device)
        else:
            A_static = None

        # 时空交替处理
        for spatial_block, temporal_block in zip(self.spatial_layers, self.temporal_layers):
            encoded = spatial_block(encoded, A_static=A_static)
            encoded = temporal_block(encoded)

        # 输出投影
        # [B, C, N, T] -> [B, N, C * T] -> [B, N, H] -> [B, H, N, 1]
        encoded = encoded.permute(0, 2, 1, 3)
        batch_size, num_nodes, model_dim, time_steps = encoded.shape
        encoded = encoded.reshape(batch_size, num_nodes, model_dim * time_steps)
        output = self.output_projection(encoded)
        output = output.unsqueeze(-1)
        output = output.permute(0, 2, 1, 3)

        return output

    def predict(self, batch):
        """
        LibCity 格式的预测

        Args:
            batch: dict with 'X' [B, T, N, F]

        Returns:
            torch.Tensor [B, T, N, output_dim]
        """
        x = batch['X']
        output = self.forward(x)
        return output

    def calculate_loss(self, batch):
        """
        计算损失函数

        Args:
            batch: dict with 'X' and 'y'

        Returns:
            torch.Tensor: loss value
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, np.nan)
