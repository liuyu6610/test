"""
特征工程模块。

职责：
- 名称文本向量化（Sentence-BERT / TF-IDF + PCA）
- 流派多标签编码 + Embedding 索引准备
- 类型 One-Hot 编码
- 数值特征标准化
- 所有 fit() 仅在训练集上进行，严格防止数据泄露
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FeatureResult:
    """特征工程的输出结果。"""

    # 节点特征矩阵（用于 GNN 的 data.x）
    node_features: np.ndarray            # [N, D_fixed]

    # 流派 Embedding 索引 + mask
    genre_indices_padded: torch.Tensor   # [N, max_genres]
    genre_mask: torch.Tensor             # [N, max_genres]
    num_genres: int                      # 流派总数
    max_genres: int                      # 最大流派数

    # 稀疏特征（用于 baseline 模型的对比实验）
    name_sparse: sp.csr_matrix
    type_sparse: sp.csr_matrix
    genre_sparse: sp.csr_matrix
    num_sparse: sp.csr_matrix

    # 密集特征（用于图构建的相似度计算）
    name_pca: np.ndarray
    type_dense: np.ndarray
    num_dense: np.ndarray

    # 目标值
    target: np.ndarray


class FeatureEngineer:
    """
    特征工程器。

    所有的 Encoder / Scaler **仅在训练集上 fit**，确保零数据泄露。
    """

    def __init__(
        self,
        sentence_model: str = "all-MiniLM-L6-v2",
        name_pca_dim: int = 128,
        tfidf_max_features: int = 384,
        hf_mirror: str = "https://hf-mirror.com",
    ):
        self.sentence_model = sentence_model
        self.name_pca_dim = name_pca_dim
        self.tfidf_max_features = tfidf_max_features
        self.hf_mirror = hf_mirror

    def transform(
        self, df: pd.DataFrame, splits: Dict[str, np.ndarray]
    ) -> FeatureResult:
        """
        执行完整的特征工程管线。

        Args:
            df: 清洗后的 DataFrame
            splits: 包含 'train', 'val', 'test' 索引的字典

        Returns:
            FeatureResult 数据类
        """
        train_idx = splits["train"]

        # 1. 名称文本向量化
        name_pca, name_sparse = self._encode_names(df["name"].tolist())

        # 2. 流派处理
        genre_sparse, genre_padded, genre_mask, num_genres, max_genres = (
            self._encode_genres(df)
        )

        # 3. 类型 One-Hot（仅 train fit）
        type_dense, type_sparse = self._encode_types(df, train_idx)

        # 4. 数值特征（仅 train fit）
        num_dense, num_sparse = self._scale_numerical(df, train_idx)

        # 新增特性交叉: Type * [Episodes, Log Members]
        num_cross = np.hstack([
            type_dense * num_dense[:, 0:1],
            type_dense * num_dense[:, 1:2]
        ]).astype(np.float32)
        logger.info("Cross features shape: %s", num_cross.shape)

        # 5. 组装节点特征矩阵（不含流派 Embedding，那部分在模型内部完成）
        node_features = np.hstack([name_pca, type_dense, num_dense, num_cross]).astype(
            np.float32
        )
        logger.info("Final node features shape: %s", node_features.shape)

        target = df["rating"].values.astype(np.float32)

        return FeatureResult(
            node_features=node_features,
            genre_indices_padded=genre_padded,
            genre_mask=genre_mask,
            num_genres=num_genres,
            max_genres=max_genres,
            name_sparse=name_sparse,
            type_sparse=type_sparse,
            genre_sparse=genre_sparse,
            num_sparse=num_sparse,
            name_pca=name_pca,
            type_dense=type_dense,
            num_dense=num_dense,
            target=target,
        )

    # ── 私有方法 ──────────────────────────────────────────

    def _encode_names(self, names: list) -> Tuple[np.ndarray, sp.csr_matrix]:
        """名称文本向量化：Sentence-BERT（fallback TF-IDF）+ PCA 降维。
        
        嵌入向量会被缓存到磁盘（outputs/cache/），后续运行直接加载，
        避免因网络不稳定降级到 TF-IDF 导致预测精度下降。
        """
        import os
        os.environ["HF_ENDPOINT"] = self.hf_mirror

        # 缓存路径
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs", "cache",
        )
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"name_embeddings_pca{self.name_pca_dim}.npy")

        # 优先从缓存加载（性能最佳实践：避免重复计算）
        if os.path.exists(cache_path):
            logger.info("Loading cached name embeddings from %s", cache_path)
            name_pca = np.load(cache_path)
            name_sparse = sp.csr_matrix(name_pca.astype(np.float32))
            logger.info("Name features shape: %s", name_sparse.shape)
            return name_pca.astype(np.float32), name_sparse

        logger.info("Encoding anime names...")
        try:
            from sentence_transformers import SentenceTransformer

            # 强制使用本地缓存模型，避免 SSL 验证错误
            os.environ["HF_HUB_OFFLINE"] = "1"
            embedder = SentenceTransformer(self.sentence_model)
            raw_embeddings = embedder.encode(names, show_progress_bar=True)
            raw_embeddings = raw_embeddings.astype(np.float32)
            logger.info("Using Sentence-BERT embeddings.")

            pca = PCA(n_components=self.name_pca_dim)
            name_pca = pca.fit_transform(raw_embeddings)
        except Exception as e:
            logger.warning("Sentence-BERT failed: %s. Falling back to TF-IDF.", e)
            vectorizer = TfidfVectorizer(max_features=self.tfidf_max_features)
            tfidf_sparse = vectorizer.fit_transform(names)
            svd = TruncatedSVD(n_components=self.name_pca_dim, random_state=42)
            name_pca = svd.fit_transform(tfidf_sparse)

        # 缓存到磁盘，后续运行无需重新计算
        np.save(cache_path, name_pca)
        logger.info("Cached name embeddings to %s", cache_path)

        name_sparse = sp.csr_matrix(name_pca.astype(np.float32))
        logger.info("Name features shape: %s", name_sparse.shape)
        return name_pca.astype(np.float32), name_sparse

    def _encode_genres(
        self, df: pd.DataFrame
    ) -> Tuple[sp.csr_matrix, torch.Tensor, torch.Tensor, int, int]:
        """流派多标签编码 + Embedding 索引。"""
        genre_lists = df["genre"].fillna("").apply(
            lambda x: x.split(", ") if x else []
        )

        mlb = MultiLabelBinarizer(sparse_output=True)
        genre_sparse = mlb.fit_transform(genre_lists)
        logger.info("Genre features shape: %s", genre_sparse.shape)

        all_genres = mlb.classes_
        genre_to_idx = {g: i for i, g in enumerate(all_genres)}

        genre_indices = [
            torch.tensor([genre_to_idx[g] for g in genres if g])
            for genres in genre_lists
        ]
        max_genres = max(len(idx) for idx in genre_indices)

        padded = torch.zeros((len(df), max_genres), dtype=torch.long)
        mask = torch.zeros((len(df), max_genres), dtype=torch.bool)
        for i, indices in enumerate(genre_indices):
            length = len(indices)
            padded[i, :length] = indices
            mask[i, :length] = True

        return genre_sparse, padded, mask, len(all_genres), max_genres

    def _encode_types(
        self, df: pd.DataFrame, train_idx: np.ndarray
    ) -> Tuple[np.ndarray, sp.csr_matrix]:
        """类型 One-Hot 编码（仅在训练集上 fit）。"""
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(df.iloc[train_idx][["type"]])
        type_dense = encoder.transform(df[["type"]]).astype(np.float32)
        type_sparse = sp.csr_matrix(type_dense)
        logger.info("Type features shape: %s", type_sparse.shape)
        return type_dense, type_sparse

    def _scale_numerical(
        self, df: pd.DataFrame, train_idx: np.ndarray
    ) -> Tuple[np.ndarray, sp.csr_matrix]:
        """
        数值特征标准化（仅在训练集上 fit）。

        处理逻辑：
        1. episodes 中的 NaN 使用训练集中位数填充
        2. members 进行 log1p 变换
        3. StandardScaler 仅在训练集上 fit
        """
        # 使用训练集中位数填充 episodes
        train_episodes_median = df.iloc[train_idx]["episodes"].median()
        df["episodes"] = df["episodes"].fillna(train_episodes_median)
        df["log_members"] = np.log1p(df["members"])

        num_cols = ["episodes", "log_members"]
        scaler = StandardScaler()
        scaler.fit(df.iloc[train_idx][num_cols])
        num_dense = scaler.transform(df[num_cols]).astype(np.float32)
        num_sparse = sp.csr_matrix(num_dense)
        logger.info("Numerical features shape: %s", num_sparse.shape)
        return num_dense, num_sparse
