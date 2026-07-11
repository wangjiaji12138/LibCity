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


## TODO:
# 改名Spatial Pattern Module SPM
# Sinkhorn-Knopp 正则化验证效果
# 是否需要detach

class SpatialPrototypeModule(nn.Module):
    """
    空间范式原型模块

    功能：
    1. 构建M个可学习的原型（空间范式）
    2. 根据节点特征分配节点到原型（Sinkhorn-Knopp 正则化 + 地理约束）
    3. 原型和编码器联合更新（无 detach）

    Args:
        num_nodes: 节点数量
        model_dim: 模型特征维度
        num_prototypes: 原型数量
        prototype_dim: 原型维度
        temperature: Sinkhorn 温度参数
        geo_adj: 地理邻接矩阵
        sinkhorn_iterations: Sinkhorn 算法迭代次数
        geo_smooth_weight: 地理平滑权重
    """

    def __init__(self, num_nodes: int, model_dim: int,
                 num_prototypes: int = 16, prototype_dim: int = 32,
                 temperature: float = 0.1, contrastive_temperature: float = 0.1,
                 geo_adj: Tensor | None = None,
                 sinkhorn_iterations: int = 3, geo_smooth_weight: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.temperature = temperature
        self.contrastive_temperature = contrastive_temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.geo_smooth_weight = geo_smooth_weight

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)

        self.proto_proj = nn.Sequential(
            nn.Linear(model_dim, prototype_dim),
            nn.LayerNorm(prototype_dim)
        )

        # 预计算地理平滑矩阵
        self._init_geo_smooth_matrix(geo_adj)

    def _init_geo_smooth_matrix(self, geo_adj: Tensor | None) -> None:
        """预计算地理平滑矩阵（在 GPU 上执行）"""
        if geo_adj is None:
            self._geo_smooth_matrix: Tensor | None = None
            return

        # 确保在 GPU 上执行
        device = geo_adj.device
        geo_adj_tensor = geo_adj.float()
        deg = geo_adj_tensor.sum(dim=-1, keepdim=True).clamp(min=1)
        D_inv = 1.0 / deg
        geo_smooth_matrix = geo_adj_tensor * D_inv
        self.register_buffer('_geo_smooth_matrix', geo_smooth_matrix)

    def _sinkhorn_knopp(self, logits: Tensor) -> Tensor:
        """
        Sinkhorn-Knopp 算法：让分配矩阵按行和列均匀分布。

        Args:
            logits: 分配 logits, shape (BT, N, M)

        Returns:
            归一化后的分配概率, shape (BT, N, M)
        """
        Q = logits
        for i in range(self.sinkhorn_iterations):
            # 行归一化
            Q = Q - torch.logsumexp(Q, dim=-1, keepdim=True)
            # 列归一化
            Q = Q - torch.logsumexp(Q, dim=-2, keepdim=True)
            # 地理平滑（除最后一次迭代）
            if self._geo_smooth_matrix is not None and i < self.sinkhorn_iterations - 1:
                Q = self._geo_smooth(Q)
        return torch.exp(Q)

    def _geo_smooth(self, Q: Tensor) -> Tensor:
        """
        地理平滑：对 Sinkhorn 分配矩阵进行空间平滑。

        让相邻节点倾向于有相似的原型分配

        Args:
            Q: (BT, N, M) - Sinkhorn 分配
        """
        # W @ Q: (N,N) is broadcast left -> (BT,N,N) @ (BT,N,M) -> (BT,N,M)
        Q_smoothed = self._geo_smooth_matrix @ Q
        alpha = self.geo_smooth_weight
        return (1 - alpha) * Q + alpha * Q_smoothed

    def forward(self, node_features: Tensor) -> tuple[Tensor, Tensor]:
        """
        前向传播：计算原型分配

        Args:
            node_features: (B, T, N, D)

        Returns:
            prototypes: (M, proto_dim)
            proto_assign: (B, T, N, M) 原型分配概率
        """
        B, T, N, D = node_features.shape
        node_features_flat = node_features.reshape(B * T, N, D)

        node_proj = self.proto_proj(node_features_flat)
        node_proj = F.normalize(node_proj, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        logits = torch.matmul(node_proj, prototypes_norm.transpose(0, 1)) / self.temperature

        proto_assign = self._sinkhorn_knopp(logits)

        return self.prototypes, proto_assign.reshape(B, T, N, self.num_prototypes)

    def get_contrastive_loss(self, node_emb: Tensor, proto_assign: Tensor) -> Tensor:
        """
        原型引导的节点级对比损失（InfoNCE / NT-Xent）

        思想：利用原型分配作为"软标签"，对每个节点构造正负样本的对比损失：
        - Anchor: 节点 i 的表示 z_i
        - Positive: 与 i 属于同一原型的节点 j 的表示 z_j
        - Negative: 与 i 属于不同原型的节点 k 的表示 z_k

        损失：L = -log( exp(sim(i,j)/τ) / Σ_k exp(sim(i,k)/τ) )
        其中 sim(i,k) = <z_i, z_k> / (||z_i||·||z_k||)，τ 是温度系数

        Args:
            node_emb: (B, T, N, D) 节点嵌入
            proto_assign: (B, T, N, M) 原型分配概率

        Returns:
            对比损失标量
        """
        B, T, N, D = node_emb.shape

        with torch.no_grad():
            hard_assign = torch.argmax(proto_assign, dim=-1)  # (B,T,N)

        emb = node_emb.reshape(B * T, N, D)          # (BT, N, D)
        labels = hard_assign.reshape(B * T, N)       # (BT, N)

        # 归一化嵌入（余弦相似度）
        emb_norm = F.normalize(emb, p=2, dim=-1)      # (BT, N, D)
        sim_matrix = torch.bmm(emb_norm, emb_norm.transpose(1, 2))  # (BT, N, N)

        # 排除自身: sim[b, i, i] = 0
        diag_mask = 1.0 - torch.eye(N, device=emb.device).unsqueeze(0)  # (1, N, N)
        sim_matrix = sim_matrix * diag_mask

        # 标签相等矩阵
        same_label = labels.unsqueeze(2) == labels.unsqueeze(1)  # (BT, N, N)
        pos_mask = same_label.float() * diag_mask        # (BT, N, N)
        neg_mask = (1.0 - same_label.float())             # (BT, N, N)

        # 过滤无效样本（没有正样本或没有负样本的 anchor）
        pos_cnt = pos_mask.sum(-1)   # (BT, N)
        neg_cnt = neg_mask.sum(-1)   # (BT, N)
        valid = (pos_cnt > 0) & (neg_cnt > 0)

        # InfoNCE: -log( exp(sim(i,pos)/τ) / [exp(sim(i,pos)/τ) + Σ_neg exp(sim(i,k)/τ)] )
        tau = self.contrastive_temperature
        logits = sim_matrix / tau   # (BT, N, N)

        # 数值稳定化：对每行减去该行最大值
        logits_max, _ = logits.max(dim=-1, keepdim=True)  # (BT, N, 1)
        logits_stable = logits - logits_max

        exp_logits = torch.exp(logits_stable) * diag_mask

        # 从 logits 矩阵直接提取正样本 logits（与负样本分母一致的数值稳定化）
        pos_logits = (logits_stable * pos_mask).sum(-1) / pos_cnt.clamp(min=1)  # (BT, N)

        # 每个 anchor 的 log-sum-exp 分母（正样本 + 负样本）
        denom_logsumexp = torch.log(exp_logits.sum(-1).clamp(min=1e-8))  # (BT, N)

        # 每个 anchor 的 InfoNCE loss
        loss_per_anchor = -pos_logits + denom_logsumexp  # (BT, N)

        loss = loss_per_anchor[valid].mean()

        if loss.isnan().item():
            return torch.tensor(0.0, device=node_emb.device, dtype=node_emb.dtype)

        return loss


class DilatedInception(nn.Module):
    """
    多尺度膨胀卷积：捕获不同时间尺度的模式

    使用 4 种不同卷积核大小的膨胀卷积：
    kernel_set = [2, 3, 6, 7]

    Args:
        cin: 输入通道数
        cout: 输出通道数
        dilation_factor: 膨胀因子
    """

    KERNEL_SET = [2, 3, 6, 7]

    def __init__(self, cin: int, cout: int, dilation_factor: int = 2):
        super().__init__()
        cout_per_head = cout // len(self.KERNEL_SET)
        self.tconv_list = nn.ModuleList()
        for kernel_size in self.KERNEL_SET:
            pad = (kernel_size - 1) * dilation_factor // 2
            self.tconv_list.append(
                nn.Conv2d(cin, cout_per_head, (1, kernel_size),
                         padding=(0, pad), dilation=(1, dilation_factor))
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, D, N, T) - Conv2d 格式

        Returns:
            (B, D', N, T') - 通道拼接后的输出
        """
        outputs = [tconv(x) for tconv in self.tconv_list]
        return torch.cat(outputs, dim=1)

class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=4, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q,K,V all have shape (B, T, N, D) in DSTGN
        B, T, N, D = query.shape
        H, h = self.num_heads, self.head_dim

        # Linear projection -> reshape to (B, H, T, N, h) -> merge to (B*H, T, N, h)
        Q = self.FC_Q(query).view(B, H, T, N, h).permute(0, 1, 3, 2, 4).reshape(B * H, T, N, h)
        K = self.FC_K(key).view(B, H, T, N, h).permute(0, 1, 3, 2, 4).reshape(B * H, T, N, h)
        V = self.FC_V(value).view(B, H, T, N, h).permute(0, 1, 3, 2, 4).reshape(B * H, T, N, h)

        # Scaled dot-product attention: (B*H, T, N, h) @ (B*H, T, h, N) -> (B*H, T, N, N)
        attn_score = (Q @ K.transpose(-1, -2)) / (h ** 0.5)

        if self.mask:
            mask = torch.ones(N, N, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ V  # (B*H, T, N, h)

        # Restore: (B*H, T, N, h) -> (B, H, T, N, h) -> (B, T, N, D)
        out = out.reshape(B, H, T, N, h).permute(0, 2, 3, 1, 4).reshape(B, T, N, D)
        out = self.out_proj(out)

        return out
        

class SelfAttentionLayer(nn.Module):
    
    def __init__(self, model_dim: int, feed_forward_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.1, mask: bool = False):
        super().__init__()
        self.model_dim = model_dim
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

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(query = x, key = x, value = x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class ProtoAwareGraphConvLayer(nn.Module):
    """
    原型感知图卷积层
    """
    def __init__(self, model_dim: int, num_prototypes: int, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_prototypes = num_prototypes

        self.linear = nn.Linear(model_dim, model_dim)
        self.proto_linear = nn.Linear(num_prototypes, model_dim)
        self.dropout = nn.Dropout(dropout)

    def _compute_prototype_enhanced_adj(self, proto_flat: Tensor, adj_norm: Tensor) -> Tensor:
        """
        计算原型增强的归一化邻接矩阵

        Args:
            proto_flat: (BT, N, M) 原型分配概率
            adj_norm: (BT, N, N) 基础归一化邻接矩阵 (D^(-1/2) A D^(-1/2))

        Returns:
            (BT, N, N) 增强后的归一化邻接矩阵
        """
        # 原型协同性：节点i和j属于同原型的概率
        A_coop = proto_flat @ proto_flat.transpose(-2, -1)  # (BT, N, N)

        # 增强邻接矩阵：直接使用已有的归一化 adj_norm，不做二次归一化
        return adj_norm * A_coop

    def forward(self, x: Tensor, adj_norm: Tensor, proto: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (B, T, N, D) 节点特征
            adj_norm: (BT, N, N) 预归一化的邻接矩阵
            proto: (B, T, N, M) 原型分配概率

        Returns:
            (B, T, N, D) 聚合后的节点特征
        """
        B, T, N, D = x.shape
        x_flat = x.transpose(1, 2).reshape(B * T, N, D)
        residual = x 

        # 选择使用原型感知或基础图卷积
        if proto is not None:
            proto_flat = proto.transpose(1, 2).reshape(B * T, N, self.num_prototypes)
            agg = self._compute_prototype_enhanced_adj(proto_flat, adj_norm) @ x_flat
        else:
            agg = adj_norm @ x_flat

        agg = agg.reshape(B, T, N, D)
        out = self.linear(agg)

        # 原型级别的残差
        if proto is not None:
            proto_res = self.proto_linear(proto)
            out = out + proto_res

        # 残差连接 + LayerNorm + ReLU
        out = out + residual
        out = F.layer_norm(out, (D,))
        return self.dropout(F.relu(out))


class DSTGN(AbstractTrafficStateModel):
    """
    双重时空图网络

    架构：特征嵌入 -> 时间注意力 -> 原型生成 + 动态邻接矩阵 -> 图卷积 -> 输出投影

    Args:
        config: 配置字典
        data_feature: 数据特征字典
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # 基本属性
        self.num_nodes = data_feature.get('num_nodes')
        self._scaler = data_feature.get('scaler')
        self._logger = getLogger()
        self.out_dim = data_feature.get('output_dim', 1)

        # 时间窗口
        self.in_window = config.get('input_window', 24)
        self.out_window = config.get('output_window', 24)

        # 嵌入维度
        self.input_embedding_dim = config.get('input_embedding_dim', 32)
        self.spatial_embedding_dim = config.get('spatial_embedding_dim', 32)
        self.tod_embedding_dim = config.get('tod_embedding_dim', 16)
        self.dow_embedding_dim = config.get('dow_embedding_dim', 16)

        # 模型维度 = 所有嵌入维度之和
        self.model_dim = self.input_embedding_dim + self.spatial_embedding_dim + self.tod_embedding_dim + self.dow_embedding_dim

        # 注意力相关
        self.feed_forward_dim = config.get('feed_forward_dim', 256)
        self.num_heads = config.get('num_heads', 4)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)

        # 原型模块参数
        self.num_prototypes = config.get('num_prototypes', 16)
        self.prototype_dim = config.get('prototype_dim', 64)
        self.prototype_temperature = config.get('prototype_temperature', 0.3)
        self.contrastive_loss_weight = config.get('contrastive_loss_weight', 0.1)
        self.contrastive_temperature = config.get('contrastive_temperature', 0.1)
        self.geo_smooth_weight = config.get('geo_smooth_weight', 0.1)
        self.sinkhorn_iterations = config.get('sinkhorn_iterations', 2)
        self.use_proto = config.get('use_proto', True)

        # GCN 相关
        self.gcn_depth = config.get('gcn_depth', 2)
        
        # 设备
        self.device = config.get('device', torch.device('cpu'))

        # 构建组件
        self.input_proj = nn.Linear(1, self.input_embedding_dim)
        self.tod_proj = nn.Linear(1, self.tod_embedding_dim)
        self.dow_proj = nn.Linear(7, self.dow_embedding_dim)

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
        nn.init.xavier_uniform_(self.node_emb)

        self.concat_linear = nn.Linear(self.model_dim, self.model_dim)


        """构建时间处理层（注意力 + 膨胀卷积）"""
        self.time_conv = nn.ModuleList([
            DilatedInception(self.model_dim, self.model_dim, dilation_factor=2)
            for _ in range(self.num_layers)
        ])
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, self.feed_forward_dim,
                              self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        """构建图卷积层"""
        self.graph_convs = nn.ModuleList([
            ProtoAwareGraphConvLayer(self.model_dim, self.num_prototypes, self.dropout)
            for _ in range(self.gcn_depth)
        ])
        self.gcn_ln = nn.LayerNorm(self.model_dim)

        """构建并注册地理邻接矩阵及其归一化形式（在 GPU 上执行）"""
        adj_mx = data_feature.get('adj_mx')
        if adj_mx is not None:
            # 将邻接矩阵移动到模型所在设备
            adj = torch.from_numpy(adj_mx).float().to(self.device)
            self.register_buffer('_geo_adj', adj)

            # 预计算对称归一化: D^(-1/2) @ (A + I) @ D^(-1/2)，在 GPU 上执行
            A_hat = adj + torch.eye(adj.shape[0], dtype=adj.dtype, device=self.device)
            deg = A_hat.sum(dim=-1, keepdim=True).clamp(min=1)
            deg_sqrt = deg.pow(0.5)
            D_inv_sqrt = deg_sqrt.reciprocal()
            adj_norm = D_inv_sqrt * A_hat * D_inv_sqrt.transpose(-2, -1)
            self.register_buffer('_geo_adj_norm', adj_norm)
            
        if self.use_proto:
            self.prototype_module = SpatialPrototypeModule(
                num_nodes=self.num_nodes,
                model_dim=self.model_dim,
                num_prototypes=self.num_prototypes,
                prototype_dim=self.prototype_dim,
                temperature=self.prototype_temperature,
                contrastive_temperature=self.contrastive_temperature,
                geo_adj=getattr(self, '_geo_adj', None),
                geo_smooth_weight=self.geo_smooth_weight,
                sinkhorn_iterations=self.sinkhorn_iterations,
            )
        
        self.end_conv_1 = nn.Conv2d(self.model_dim, self.model_dim, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(self.model_dim, self.out_window, kernel_size=(1, 1))
        
        # 日志记录
        self._logger.info(
            f'DSTGN | nodes={self.num_nodes}, model_dim={self.model_dim}, '
            f'layers={self.num_layers}, use_proto={self.use_proto}'
        )

        self.apply(self._init_weights)
        

    def _compute_dynamic_adj(self, x: Tensor, proto: Tensor | None) -> Tensor:
        """
        计算动态邻接矩阵

        Args:
            x: (B, T, N, D) 嵌入特征
            proto: (B, T, N, M) 原型分配概率

        Returns:
            (B, N, N) 对时间平均后的动态邻接矩阵
        """
        B, T, N, _ = x.shape

        if proto is not None:
            sim = torch.matmul(proto, proto.transpose(-2, -1))
            sim = F.relu(sim)
        else:
            sim = torch.ones(B, T, N, N, device=x.device, dtype=x.dtype)

        # 结合地理邻接矩阵
        if hasattr(self, '_geo_adj') and self._geo_adj is not None:
            dynamic_adj = sim * self._geo_adj  # broadcast (B,T,N,N)*(N,N)->(B,T,N,N), already non-negative
        else:
            dynamic_adj = sim

        # 对时间窗口取平均
        return dynamic_adj.mean(dim=1)

    def _normalize_adj(self, dynamic_adj: Tensor, B: int, T: int, N: int) -> Tensor:
        """
        归一化动态邻接矩阵

        Args:
            dynamic_adj: (B, N, N) 对时间平均后的动态邻接矩阵
            B, T, N: 批次、时间、节点维度

        Returns:
            (B*T, N, N) 归一化后的邻接矩阵
        """
        if hasattr(self, '_geo_adj_norm') and self._geo_adj_norm is not None:
            geo_adj_norm = self._geo_adj_norm
            if geo_adj_norm.device != dynamic_adj.device:
                geo_adj_norm = geo_adj_norm.to(dynamic_adj.device)

            adj_norm = geo_adj_norm * dynamic_adj
            deg = adj_norm.sum(dim=-1, keepdim=True).clamp(min=1)
            deg_sqrt = deg.pow(0.5)
            adj_norm = deg_sqrt.reciprocal() * adj_norm * deg_sqrt.reciprocal().transpose(-2, -1)
        else:
            adj_norm = dynamic_adj

        return adj_norm.unsqueeze(1).expand(B, T, N, N).reshape(B * T, N, N)

    def _init_weights(self, m) -> None:
        """权重初始化"""
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, batch: dict, return_prototype_info: bool = False) -> Tensor:
        B, T, N, _ = batch['X'].shape

        # === 特征嵌入 ===
        x_val = self.input_proj(batch['X'][..., 0:1])
        x_tod = self.tod_proj(batch['X'][..., 1:2])
        x_dow = self.dow_proj(batch['X'][..., 2:9])
        spatial_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(B, T, -1, -1)
        
        x = torch.cat([x_val, x_tod, x_dow, spatial_emb], dim=-1)
        x = self.concat_linear(x)

        # === 时间处理：时间注意力 + 膨胀卷积 ===
        for i, (attn_t, tconv) in enumerate(zip(self.attn_layers_t, self.time_conv)):
            residual = x
            x = attn_t(x, dim=1)   #[B, T, N, D]
            x_conv = x.permute(0, 3, 2, 1)  #[B, D, N, T]
            x_conv = tconv(x_conv)  #[B, D, N, T]
            x = x_conv.permute(0, 3, 2, 1) + residual

        # === 原型模块 ===
        proto = None
        if self.use_proto:
            _, proto = self.prototype_module(x)
            proto_info = {
                'prototypes': None,
                'proto': proto,
                'embedded_features': x,
            }
        else:
            proto_info = {
                'prototypes': None,
                'proto': None,
                'embedded_features': x,
            }

        # === 动态邻接矩阵 ===
        dynamic_adj = self._compute_dynamic_adj(x, proto)
        adj_norm = self._normalize_adj(dynamic_adj, B, T, N)

        # === 图卷积 ===
        for gcn in self.graph_convs:
            x = gcn(x, adj_norm, proto)
        x = self.gcn_ln(x)
        x = F.relu(x)

        # === 输出映射 ===
        output = x.permute(0, 3, 2, 1) #[B, D, N, T]
        output = F.relu(self.end_conv_1(output)) 
        output = self.end_conv_2(output) #[B, 1, N, T]
        output = output.permute(0, 3, 2, 1) #[B, T, N, 1]

        if return_prototype_info:
            return output, proto_info
        return output

    def predict(self, batch: dict) -> Tensor:
        return self.forward(batch, return_prototype_info=False)

    def calculate_loss(self, batch: dict) -> Tensor:
        """
        计算总损失 = 预测损失 + 原型对比损失（可选）

        Args:
            batch: 输入数据

        Returns:
            总损失标量
        """
        y_pred, proto_info = self.forward(batch, return_prototype_info=True)

        y_true = batch['y']
        y_pred_inv = self._scaler.inverse_transform(y_pred[..., :self.out_dim])
        y_true_inv = self._scaler.inverse_transform(y_true[..., :self.out_dim])
        pred_loss = loss.masked_mae_torch(y_pred_inv, y_true_inv)

        self._last_losses = {'pred': pred_loss.item()}
        total_loss = pred_loss

        # 原型对比损失（三元组）
        if self.use_proto and proto_info['proto'] is not None:
            contrastive_loss = self.prototype_module.get_contrastive_loss(
                proto_info['embedded_features'],
                proto_info['proto']
            )

            # 原型分布统计
            hard_proto = torch.argmax(proto_info['proto'], dim=-1)
            proto_counts = torch.bincount(
                hard_proto.view(-1), minlength=self.num_prototypes
            ).float()
            proto_ratio = proto_counts / proto_counts.sum()

            self._last_losses.update({
                'contrastive': contrastive_loss.item(),
                'contrastive_loss_weight': self.contrastive_loss_weight,
                'contrastive_loss_contrib': (self.contrastive_loss_weight * contrastive_loss).item(),
                'proto_dist': proto_ratio.cpu().tolist()
            })

            total_loss = total_loss + self.contrastive_loss_weight * contrastive_loss

        return total_loss

    def get_last_losses(self) -> dict:
        return getattr(self, '_last_losses', {})
