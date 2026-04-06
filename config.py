"""
集中式配置模块：管理所有超参数、路径和随机种子。
避免魔法数字散落在各模块中。
"""

import os
from dataclasses import dataclass, field
from typing import List

# ── 项目根目录 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 路径配置 ──
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_FILE = os.path.join(DATA_DIR, "anime.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# Hugging Face 镜像源（解决国内网络问题）
HF_MIRROR = "https://hf-mirror.com"


@dataclass
class DataConfig:
    """数据预处理相关配置。"""
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42


@dataclass
class FeatureConfig:
    """特征工程相关配置。"""
    # 名称文本嵌入
    sentence_model: str = "all-MiniLM-L6-v2"
    name_pca_dim: int = 128
    tfidf_max_features: int = 384

    # 流派嵌入
    genre_embed_dim: int = 32


@dataclass
class GraphConfig:
    """图构建相关配置。"""
    # 流派边采样
    max_pairs_per_genre: int = 80
    genre_seed: int = 42

    # kNN 相似性边
    knn_k: int = 15
    knn_metric: str = "cosine"


@dataclass
class ModelConfig:
    """GNN 模型相关配置。"""
    conv_type: str = "sage"          # "gcn", "sage", "gat"
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    genre_embed_dim: int = 32

    # GAT 专用
    gat_heads: int = 4


@dataclass
class TrainConfig:
    """训练相关配置。"""
    # 优化器
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # 训练循环
    max_epochs: int = 300
    batch_size: int = 1024
    num_neighbors: List[int] = field(default_factory=lambda: [10, 10])

    # 早停
    patience: int = 20

    # 学习率调度
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10

    # 梯度裁剪
    max_grad_norm: float = 1.0

    # 模型保存
    best_model_name: str = "best_gnn.pt"


def ensure_dirs() -> None:
    """确保所有输出目录存在。"""
    for d in [MODEL_DIR, RESULT_DIR, FIGURE_DIR]:
        os.makedirs(d, exist_ok=True)
