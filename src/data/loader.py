"""
数据加载与基础清洗模块。

职责：
- 从 CSV 文件加载原始动漫数据
- 处理缺失值与异常值
- 划分训练/验证/测试集（先切分，后处理）
"""

import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    加载并清洗动漫数据集。

    处理逻辑：
    1. 将 episodes 列中的 'Unknown' 替换为 NaN
    2. 将 episodes 转为数值类型
    3. 用中位数填充 rating 缺失值
    4. 将 members 缺失值填充为 0

    Args:
        csv_path: anime.csv 文件路径

    Returns:
        清洗后的 DataFrame
    """
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Raw data shape: %s", df.shape)

    # episodes: 'Unknown' → NaN → 数值
    df["episodes"] = df["episodes"].replace("Unknown", np.nan)
    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")

    # rating: 中位数填充
    rating_median = df["rating"].median()
    n_missing_rating = df["rating"].isna().sum()
    df["rating"] = df["rating"].fillna(rating_median)
    logger.info("Filled %d missing ratings with median %.2f", n_missing_rating, rating_median)

    # members: 0 填充
    df["members"] = df["members"].fillna(0)

    logger.info("Cleaned data shape: %s", df.shape)
    return df


def split_dataset(
    n_samples: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    划分训练/验证/测试集索引。

    **必须在任何特征工程之前调用**，以防止数据泄露。

    Args:
        n_samples: 数据集总样本数
        test_size: 测试集占总数据的比例
        val_size: 验证集占（总数据 - 测试集）的比例
        random_state: 随机种子

    Returns:
        包含 'train', 'val', 'test' 三个键的索引字典
    """
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size, random_state=random_state
    )

    logger.info(
        "Dataset split: train=%d, val=%d, test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}
