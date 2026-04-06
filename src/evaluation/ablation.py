"""
消融实验模块。

系统性地移除特定特征组件，评估其对模型性能的影响：
1. 移除名称向量 → 仅保留类型、流派、数值特征
2. 移除图结构 → 使用 XGBoost 代替 GNN（等效于 MLP）
"""

import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def run_ablation_study(
    name_sparse: sp.csr_matrix,
    type_sparse: sp.csr_matrix,
    genre_sparse: sp.csr_matrix,
    num_sparse: sp.csr_matrix,
    target: np.ndarray,
    splits: Dict[str, np.ndarray],
    save_path: str = "outputs/results/ablation_results.csv",
) -> Dict[str, Dict[str, float]]:
    """
    运行消融实验。

    Args:
        name_sparse: 名称特征稀疏矩阵
        type_sparse: 类型特征稀疏矩阵
        genre_sparse: 流派特征稀疏矩阵
        num_sparse: 数值特征稀疏矩阵
        target: 评分目标值
        splits: 数据集拆分索引
        save_path: 结果保存路径

    Returns:
        消融实验结果字典
    """
    logger.info("Running ablation study...")

    train_idx = splits["train"]
    test_idx = splits["test"]
    y_train = target[train_idx]
    y_test = target[test_idx]

    results: Dict[str, Dict[str, float]] = {}

    # ── 消融 1：移除名称向量 ──
    logger.info("Ablation: removing name features...")
    X_no_name = sp.hstack([type_sparse, genre_sparse, num_sparse], format="csr")
    rf_no_name = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_no_name.fit(X_no_name[train_idx], y_train)
    pred_no_name = rf_no_name.predict(X_no_name[test_idx])
    rmse_no_name = float(np.sqrt(mean_squared_error(y_test, pred_no_name)))
    mae_no_name = float(mean_absolute_error(y_test, pred_no_name))
    results["RF (no name)"] = {"RMSE": rmse_no_name, "MAE": mae_no_name}
    logger.info(
        "  RF (no name) - RMSE: %.4f, MAE: %.4f", rmse_no_name, mae_no_name
    )

    # ── 消融 2：移除图结构（用 XGBoost 代替 GNN） ──
    logger.info("Ablation: removing graph structure (XGBoost as proxy)...")
    X_full = sp.hstack(
        [name_sparse, type_sparse, genre_sparse, num_sparse], format="csr"
    )
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        xgb_model.fit(X_full[train_idx], y_train)
        pred_xgb = xgb_model.predict(X_full[test_idx])
        rmse_xgb = float(np.sqrt(mean_squared_error(y_test, pred_xgb)))
        mae_xgb = float(mean_absolute_error(y_test, pred_xgb))
        results["XGBoost (no graph)"] = {"RMSE": rmse_xgb, "MAE": mae_xgb}
        logger.info(
            "  XGBoost (no graph) - RMSE: %.4f, MAE: %.4f", rmse_xgb, mae_xgb
        )
    except ImportError:
        logger.warning("XGBoost not installed, skipping ablation 2.")

    # 打印表格
    logger.info("=== Ablation Results ===")
    for name, metrics in results.items():
        logger.info("  %-25s RMSE: %.4f  MAE: %.4f", name, metrics["RMSE"], metrics["MAE"])

    # 保存 CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(results).T.to_csv(save_path)
    logger.info("Results saved to %s", save_path)

    return results
