"""
GNN 模型定义模块。

支持通过配置切换不同的图卷积层类型（GCN / SAGE / GAT），
统一了 z.py ~ z4.py 中 5 种不同的模型变体。

优化增强（基于 PyTorch Skill 最佳实践）：
- Kaiming 权重初始化
- 残差连接（Residual / Skip Connections）
- Edge Dropout（模拟 NeighborLoader 图结构扰动）
- 输入投影层（Input Projection）
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import dropout_edge

logger = logging.getLogger(__name__)

# 卷积层注册表
CONV_REGISTRY = {
    "gcn": GCNConv,
    "sage": SAGEConv,
    "gat": GATConv,
}


class AnimeGNN(nn.Module):
    """
    可配置的动漫评分预测图神经网络。

    特性：
    - 支持 GCN / GraphSAGE / GAT 卷积层
    - 流派 Embedding 聚合
    - 残差连接（Residual Connections）增强梯度流
    - Kaiming 初始化提升训练稳定性
    - Edge Dropout 模拟图采样正则化
    - Sigmoid 输出缩放至评分范围
    - min_rating / max_rating 通过构造函数传入（不依赖全局变量）

    Args:
        in_channels: 输入节点特征维度
        hidden_channels: 隐层维度
        num_genres: 流派总数（用于 Embedding 层）
        min_rating: 训练集最小评分
        max_rating: 训练集最大评分
        genre_embed_dim: 流派 Embedding 维度
        num_layers: GNN 层数
        dropout: Dropout 比率
        conv_type: 卷积层类型 ("gcn", "sage", "gat")
        gat_heads: GAT 注意力头数（仅 conv_type="gat" 时有效）
        edge_dropout: Edge Dropout 比率（训练期间随机丢弃边的概率）
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_genres: int,
        min_rating: float,
        max_rating: float,
        genre_embed_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        conv_type: str = "sage",
        gat_heads: int = 4,
        edge_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.edge_dropout = edge_dropout

        if conv_type not in CONV_REGISTRY:
            raise ValueError(
                f"Unknown conv_type '{conv_type}'. "
                f"Available: {list(CONV_REGISTRY.keys())}"
            )

        # 流派 Embedding
        self.genre_embed = nn.Embedding(num_genres, genre_embed_dim)
        total_in = in_channels + genre_embed_dim

        # 输入投影层：将拼接后的特征投影到 hidden_channels
        if conv_type == "gat":
            hidden_out = hidden_channels * gat_heads
        else:
            hidden_out = hidden_channels
        self.input_proj = nn.Linear(total_in, hidden_out)

        # 构建卷积层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if conv_type == "gat":
                in_dim = hidden_out  # 统一使用 hidden_out 维度
                self.convs.append(
                    GATConv(in_dim, hidden_channels, heads=gat_heads, dropout=dropout)
                )
                self.bns.append(nn.BatchNorm1d(hidden_out))
            else:
                conv_cls = CONV_REGISTRY[conv_type]
                in_dim = hidden_out if i == 0 else hidden_channels
                out_dim = hidden_channels
                self.convs.append(conv_cls(in_dim, out_dim))
                self.bns.append(nn.BatchNorm1d(out_dim))

        # 残差投影层（当输入输出维度不一致时）
        if conv_type != "gat" and hidden_out != hidden_channels:
            self.res_proj = nn.Linear(hidden_out, hidden_channels)
        else:
            self.res_proj = None

        # 输出头：两层 MLP 增强表达力
        final_dim = hidden_out if conv_type == "gat" else hidden_channels
        self.output_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, 1),
        )

        # 评分范围（来自训练集，非全局数据）
        self.register_buffer(
            "min_rating", torch.tensor(min_rating, dtype=torch.float32)
        )
        self.register_buffer(
            "max_rating", torch.tensor(max_rating, dtype=torch.float32)
        )

        # PyTorch Skill 最佳实践：Kaiming 权重初始化
        self._init_weights()

        logger.info(
            "AnimeGNN initialized: conv=%s, layers=%d, hidden=%d, dropout=%.2f, edge_drop=%.2f",
            conv_type, num_layers, hidden_channels, dropout, edge_dropout,
        )

    def _init_weights(self) -> None:
        """Kaiming 初始化所有线性层和 Embedding 层的权重。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        genre_ids: torch.Tensor,
        genre_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 节点特征 [N, D]
            edge_index: 边索引 [2, E]
            genre_ids: 流派索引 [N, max_genres]
            genre_mask: 流派掩码 [N, max_genres]

        Returns:
            预测评分 [N, 1]
        """
        # 流派 Embedding 聚合（均值池化，带掩码）
        genre_emb = self.genre_embed(genre_ids)  # [N, max_genres, D_genre]
        masked_emb = genre_emb * genre_mask.unsqueeze(-1).float()
        denom = genre_mask.sum(dim=1, keepdim=True).float().clamp(min=1e-8)
        genre_pooled = masked_emb.sum(dim=1) / denom  # [N, D_genre]

        # 拼接节点特征与流派 Embedding
        x = torch.cat([x, genre_pooled], dim=1)

        # 输入投影（统一维度，为残差连接做准备）
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 引入 Edge Dropout 模拟图采样带来的正则化
        current_edge_index = edge_index
        if self.training:
            current_edge_index, _ = dropout_edge(
                edge_index, p=self.edge_dropout, training=self.training
            )

        # 图卷积层 + 残差连接
        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, current_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # 残差连接：当维度匹配时直接相加，否则通过投影对齐
            if residual.size(-1) == x.size(-1):
                x = x + residual
            elif self.res_proj is not None and i == 0:
                x = x + self.res_proj(residual)

        # 输出映射到评分范围
        out = self.output_head(x)
        out = torch.sigmoid(out)
        out = self.min_rating + (self.max_rating - self.min_rating) * out
        return out
