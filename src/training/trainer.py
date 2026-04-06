"""
GNN 训练器模块。

封装训练循环、验证评估、早停机制和学习率调度。
支持 NeighborLoader (Mini-Batch) 图采样训练。
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from src.models.gnn import AnimeGNN

logger = logging.getLogger(__name__)


class GNNTrainer:
    """
    GNN 训练器。

    使用 NeighborLoader 进行 Mini-Batch 图采样训练，
    提供天然的拓扑正则化效果。

    Args:
        model: AnimeGNN 模型实例
        data: PyG Data 对象
        learning_rate: 学习率
        weight_decay: L2 正则化权重衰减
        max_epochs: 最大训练轮数
        batch_size: Mini-Batch 大小
        num_neighbors: 每层采样邻居数列表
        patience: 早停耐心值
        max_grad_norm: 梯度裁剪阈值
        save_dir: 模型权重保存目录
        best_model_name: 最佳模型文件名
    """

    def __init__(
        self,
        model: AnimeGNN,
        data: Data,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        max_epochs: int = 300,
        batch_size: int = 1024,
        num_neighbors: Optional[List[int]] = None,
        patience: int = 30,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 15,
        max_grad_norm: float = 1.0,
        save_dir: str = "outputs/models",
        best_model_name: str = "best_gnn.pt",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        self.model = model.to(self.device)
        self.data = data.to(self.device)

        self.max_epochs = max_epochs
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.save_path = os.path.join(save_dir, best_model_name)

        if num_neighbors is None:
            num_neighbors = [10, 10]
        self.num_neighbors = num_neighbors

        # 构建 NeighborLoader（Mini-Batch 图采样）
        train_node_ids = torch.nonzero(data.train_mask).view(-1)
        self.train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=train_node_ids,
            shuffle=True,
        )
        logger.info(
            "NeighborLoader created: batch_size=%d, num_neighbors=%s, train_nodes=%d",
            batch_size, num_neighbors, len(train_node_ids),
        )

        # 优化器与调度器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # Huber Loss（SmoothL1）对离群评分更鲁棒
        self.criterion = nn.SmoothL1Loss()
        # ReduceLROnPlateau：在验证指标停止改善时自动降低学习率
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=scheduler_factor,
            patience=scheduler_patience, min_lr=1e-6,
        )

    def train(self) -> Dict[str, float]:
        """
        执行完整的训练流程（含早停）。

        Returns:
            测试集上的 {'RMSE': float, 'MAE': float}
        """
        best_val_rmse = float("inf")
        counter = 0

        logger.info("Starting training for up to %d epochs...", self.max_epochs)

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_one_epoch()
            val_rmse, val_mae = self._evaluate(self.data.val_mask)

            self.scheduler.step(val_rmse)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.save_path)
                counter = 0
                if epoch % 20 == 0 or epoch == 1:
                    logger.info(
                        "Epoch %03d: New best (Val RMSE: %.4f)", epoch, val_rmse
                    )
            else:
                counter += 1
                if counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            if epoch % 20 == 0:
                logger.info(
                    "Epoch %03d | Loss: %.4f | Val RMSE: %.4f | Val MAE: %.4f",
                    epoch, train_loss, val_rmse, val_mae,
                )

        # 加载最佳模型并在测试集上评估
        self.model.load_state_dict(torch.load(self.save_path, weights_only=True))
        test_rmse, test_mae = self._evaluate(self.data.test_mask)
        logger.info("Test RMSE: %.4f | Test MAE: %.4f", test_rmse, test_mae)

        return {"RMSE": test_rmse, "MAE": test_mae}

    def _train_one_epoch(self) -> float:
        """使用 NeighborLoader 进行 Mini-Batch 图采样训练。"""
        self.model.train()
        total_loss = 0.0
        total_nodes = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(
                batch.x, batch.edge_index, batch.genre_ids, batch.genre_mask
            )

            # 只对 batch 中的 seed 节点（即 batch_size 个中心节点）计算 loss
            batch_size = batch.batch_size
            loss = self.criterion(out[:batch_size], batch.y[:batch_size])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_nodes += batch_size

        return total_loss / total_nodes

    def _evaluate(self, mask: torch.Tensor) -> Tuple[float, float]:
        """在给定 mask 上评估模型（全图推理）。"""
        self.model.eval()

        with torch.no_grad():
            out = self.model(
                self.data.x, self.data.edge_index,
                self.data.genre_ids, self.data.genre_mask,
            )
            preds = out[mask].cpu().numpy()
            trues = self.data.y[mask].cpu().numpy()

        rmse = float(np.sqrt(mean_squared_error(trues, preds)))
        mae = float(mean_absolute_error(trues, preds))
        return rmse, mae
