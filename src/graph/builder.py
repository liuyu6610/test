"""
图构建模块。

职责：
- 流派共现边构建（O(1) 内存采样，杜绝 OOM）
- kNN 特征相似性边构建（Scaler 仅 train fit）
- 组装完整 PyG Data 对象
"""

import logging
import random
import math
from typing import Dict, Set, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src.data.features import FeatureResult

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    图构建器。

    将动漫节点通过流派共现关系和特征相似性连接起来，
    构建用于图神经网络的 PyG Data 对象。
    """

    def __init__(
        self,
        max_pairs_per_genre: int = 50,
        knn_k: int = 15,
        knn_metric: str = "cosine",
        seed: int = 42,
    ):
        self.max_pairs_per_genre = max_pairs_per_genre
        self.knn_k = knn_k
        self.knn_metric = knn_metric
        self.seed = seed

    def build(
        self,
        features: FeatureResult,
        genre_lists: list,
        splits: Dict[str, np.ndarray],
    ) -> Data:
        """
        构建完整的 PyG Data 对象。

        Args:
            features: 特征工程的输出结果
            genre_lists: 每部动漫的流派列表 (list of list)
            splits: 包含 'train', 'val', 'test' 索引的字典

        Returns:
            PyG Data 对象，包含 x, edge_index, y, genre_ids, genre_mask, masks
        """
        train_idx = splits["train"]
        n_nodes = len(features.target)

        # 构建边集合
        edges: Set[Tuple[int, int]] = set()

        # 1. 流派共现边
        self._add_genre_edges(edges, genre_lists)

        # 2. kNN 相似性边
        self._add_knn_edges(edges, features, train_idx)

        logger.info("Total edges: %d", len(edges))

        # 转为 edge_index 张量
        edge_index = (
            torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        )

        # 构建 Data 对象
        data = Data(
            x=torch.tensor(features.node_features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(features.target, dtype=torch.float).view(-1, 1),
            genre_ids=features.genre_indices_padded,
            genre_mask=features.genre_mask,
        )

        # 设置 masks
        for split_name, idx in splits.items():
            mask = torch.zeros(n_nodes, dtype=torch.bool)
            mask[idx] = True
            setattr(data, f"{split_name}_mask", mask)

        return data

    def _add_genre_edges(
        self, edges: Set[Tuple[int, int]], genre_lists: list
    ) -> None:
        """
        流派共现边构建。

        使用 O(1) 内存的随机索引采样，避免生成全部 combinations 导致 OOM。
        当某流派下的组合数超过阈值时，通过随机采样两个元素的方式
        逐步收集目标数量的边对。
        """
        logger.info(
            "Building genre edges (max %d pairs/genre)...",
            self.max_pairs_per_genre,
        )

        # 建立流派 → 动漫索引的映射
        genre_to_anime: Dict[str, list] = {}
        for i, genres in enumerate(genre_lists):
            for g in genres:
                if g:
                    genre_to_anime.setdefault(g, []).append(i)

        random.seed(self.seed)

        for genre, anime_list in genre_to_anime.items():
            n = len(anime_list)
            if n < 2:
                continue

            total_pairs = n * (n - 1) // 2
            sample_size = min(self.max_pairs_per_genre, int(math.sqrt(total_pairs)) * 8)

            if total_pairs <= sample_size:
                # 数量可控，直接枚举
                for i in range(n):
                    for j in range(i + 1, n):
                        u, v = anime_list[i], anime_list[j]
                        edges.add((u, v))
                        edges.add((v, u))
            else:
                # O(1) 内存随机采样
                sampled = 0
                seen = set()
                while sampled < sample_size:
                    u, v = random.sample(anime_list, 2)
                    pair = (min(u, v), max(u, v))
                    if pair not in seen:
                        seen.add(pair)
                        edges.add((u, v))
                        edges.add((v, u))
                        sampled += 1

    def _add_knn_edges(
        self,
        edges: Set[Tuple[int, int]],
        features: FeatureResult,
        train_idx: np.ndarray,
    ) -> None:
        """
        kNN 相似性边构建。

        StandardScaler 仅在训练集上 fit，防止数据泄露。
        """
        logger.info("Building kNN graph (k=%d)...", self.knn_k)

        X_sim = np.hstack([features.name_pca, features.type_dense, features.num_dense])

        scaler = StandardScaler()
        scaler.fit(X_sim[train_idx])
        X_scaled = scaler.transform(X_sim)

        knn = NearestNeighbors(
            n_neighbors=self.knn_k + 1, metric=self.knn_metric
        )
        knn.fit(X_scaled[train_idx])
        _, indices = knn.kneighbors(X_scaled)

        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # 跳过自身
                if i != j:
                    edges.add((i, j))
                    edges.add((j, i))
