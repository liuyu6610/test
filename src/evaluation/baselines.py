"""
基线模型对比实验模块。

运行传统机器学习模型（线性回归、随机森林、XGBoost、kNN）
作为 GNN 的 Baseline 对比。
"""

import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

logger = logging.getLogger(__name__)


def run_baseline_comparison(
    name_sparse: sp.csr_matrix,
    type_sparse: sp.csr_matrix,
    genre_sparse: sp.csr_matrix,
    num_sparse: sp.csr_matrix,
    target: np.ndarray,
    splits: Dict[str, np.ndarray],
    gnn_metrics: Dict[str, float],
    save_path: str = "outputs/results/comparison_results.csv",
) -> Dict[str, Dict[str, float]]:
    """
    运行基线对比实验。

    Args:
        name_sparse: 名称特征稀疏矩阵
        type_sparse: 类型特征稀疏矩阵
        genre_sparse: 流派特征稀疏矩阵
        num_sparse: 数值特征稀疏矩阵
        target: 评分目标值
        splits: 数据集拆分索引
        gnn_metrics: GNN 模型的测试集指标
        save_path: 结果保存路径

    Returns:
        所有模型的 {model_name: {RMSE, MAE}} 字典
    """
    logger.info("Running baseline comparison experiments...")

    # 组合全部特征
    X_full = sp.hstack(
        [name_sparse, type_sparse, genre_sparse, num_sparse], format="csr"
    )

    train_idx = splits["train"]
    test_idx = splits["test"]
    X_train, y_train = X_full[train_idx], target[train_idx]
    X_test, y_test = X_full[test_idx], target[test_idx]

    # 定义基线模型
    baselines = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "kNN": KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
    }

    # 尝试导入 XGBoost
    try:
        import xgboost as xgb

        baselines["XGBoost"] = xgb.XGBRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    except ImportError:
        logger.warning("XGBoost not installed, skipping.")

    results: Dict[str, Dict[str, float]] = {}

    for name, model in baselines.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        results[name] = {"RMSE": rmse, "MAE": mae}
        logger.info("%s - RMSE: %.4f, MAE: %.4f", name, rmse, mae)

    # 加入 GNN 结果
    results["GNN"] = gnn_metrics

    # 打印表格
    logger.info("=== Comparison Results ===")
    for name, metrics in results.items():
        logger.info("  %-25s RMSE: %.4f  MAE: %.4f", name, metrics["RMSE"], metrics["MAE"])

    # 保存 CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(results).T.to_csv(save_path)
    logger.info("Results saved to %s", save_path)

    return results
