"""
主训练入口脚本。

串联完整的 Pipeline：
数据加载 → 特征工程 → 图构建 → GNN 训练 → 基线对比 → 消融实验

用法：
    python scripts/train.py
    python scripts/train.py --conv_type gat --hidden 128 --epochs 500
"""

import argparse
import logging
import os
import sys

# 将项目根目录加入 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    DATA_FILE,
    MODEL_DIR,
    RESULT_DIR,
    DataConfig,
    FeatureConfig,
    GraphConfig,
    ModelConfig,
    TrainConfig,
    ensure_dirs,
)
from src.data.loader import load_and_clean_data, split_dataset
from src.data.features import FeatureEngineer
from src.graph.builder import GraphBuilder
from src.models.gnn import AnimeGNN
from src.training.trainer import GNNTrainer
from src.evaluation.baselines import run_baseline_comparison
from src.evaluation.ablation import run_ablation_study


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="基于图神经网络的动漫评分预测系统"
    )
    parser.add_argument(
        "--conv_type",
        type=str,
        default="sage",
        choices=["gcn", "sage", "gat"],
        help="GNN 卷积层类型 (default: sage)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="隐层维度 (default: 128)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="GNN 层数 (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout 比率 (default: 0.4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="学习率 (default: 0.001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="最大训练轮数 (default: 300)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="批大小 (default: 1024)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="早停耐心值 (default: 30)",
    )
    parser.add_argument(
        "--skip_baselines",
        action="store_true",
        help="跳过基线对比实验",
    )
    parser.add_argument(
        "--skip_ablation",
        action="store_true",
        help="跳过消融实验",
    )
    return parser.parse_args()


def setup_logging() -> None:
    """配置日志格式。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """主流程入口。"""
    args = parse_args()
    setup_logging()
    ensure_dirs()

    logger = logging.getLogger("train")
    logger.info("=" * 60)
    logger.info("基于深度学习的动漫评分数据分析研究")
    logger.info("=" * 60)

    # ── 1. 数据加载与清洗 ──
    df = load_and_clean_data(DATA_FILE)

    # ── 2. 数据集划分（先切分，防止泄露） ──
    data_cfg = DataConfig()
    splits = split_dataset(
        n_samples=len(df),
        test_size=data_cfg.test_size,
        val_size=data_cfg.val_size,
        random_state=data_cfg.random_state,
    )

    # 计算训练集评分范围（用于模型输出层）
    target = df["rating"].values.astype("float32")
    train_min_rating = float(target[splits["train"]].min())
    train_max_rating = float(target[splits["train"]].max())
    logger.info(
        "Train rating range: [%.2f, %.2f]", train_min_rating, train_max_rating
    )

    # ── 3. 特征工程 ──
    feat_cfg = FeatureConfig()
    engineer = FeatureEngineer(
        sentence_model=feat_cfg.sentence_model,
        name_pca_dim=feat_cfg.name_pca_dim,
        tfidf_max_features=feat_cfg.tfidf_max_features,
    )
    features = engineer.transform(df, splits)

    # ── 4. 图构建 ──
    graph_cfg = GraphConfig()
    builder = GraphBuilder(
        max_pairs_per_genre=graph_cfg.max_pairs_per_genre,
        knn_k=graph_cfg.knn_k,
        knn_metric=graph_cfg.knn_metric,
        seed=graph_cfg.genre_seed,
    )

    genre_lists = df["genre"].fillna("").apply(
        lambda x: x.split(", ") if x else []
    ).tolist()

    data = builder.build(features, genre_lists, splits)

    # ── 5. GNN 训练 ──
    num_neighbors = [10] * args.layers
    model = AnimeGNN(
        in_channels=data.x.size(1),
        hidden_channels=args.hidden,
        num_genres=features.num_genres,
        min_rating=train_min_rating,
        max_rating=train_max_rating,
        genre_embed_dim=feat_cfg.genre_embed_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        conv_type=args.conv_type,
    )

    trainer = GNNTrainer(
        model=model,
        data=data,
        learning_rate=args.lr,
        weight_decay=1e-4,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        num_neighbors=num_neighbors,
        patience=args.patience,
        save_dir=MODEL_DIR,
    )

    gnn_metrics = trainer.train()

    # ── 6. 基线对比实验 ──
    if not args.skip_baselines:
        run_baseline_comparison(
            name_sparse=features.name_sparse,
            type_sparse=features.type_sparse,
            genre_sparse=features.genre_sparse,
            num_sparse=features.num_sparse,
            target=features.target,
            splits=splits,
            gnn_metrics=gnn_metrics,
            save_path=os.path.join(RESULT_DIR, "comparison_results.csv"),
        )

    # ── 7. 消融实验 ──
    if not args.skip_ablation:
        run_ablation_study(
            name_sparse=features.name_sparse,
            type_sparse=features.type_sparse,
            genre_sparse=features.genre_sparse,
            num_sparse=features.num_sparse,
            target=features.target,
            splits=splits,
            save_path=os.path.join(RESULT_DIR, "ablation_results.csv"),
        )

    logger.info("=" * 60)
    logger.info("All experiments completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
