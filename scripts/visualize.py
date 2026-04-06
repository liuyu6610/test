"""
数据可视化脚本。

生成三种核心可视化图表：
1. 评分分布图（直方图 + KDE）
2. 流派词云
3. 不同制作类型的平均评分对比条形图

用法：
    python scripts/visualize.py
"""

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")  # 服务器端无头渲染

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_FILE, FIGURE_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def setup_plot_style() -> None:
    """配置全局绘图样式。"""
    sns.set_style("whitegrid")
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_rating_distribution(df: pd.DataFrame, save_dir: str) -> None:
    """绘制评分分布图（直方图 + 核密度估计）。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["rating"].dropna(), bins=30, kde=True, color="skyblue", ax=ax)
    ax.set_title("评分分布", fontsize=16)
    ax.set_xlabel("评分")
    ax.set_ylabel("频数")
    fig.tight_layout()

    path = os.path.join(save_dir, "rating_distribution.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_genre_wordcloud(df: pd.DataFrame, save_dir: str) -> None:
    """生成流派词云图。"""
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud package not installed, skipping.")
        return

    all_genres = ",".join(df["genre"].dropna().astype(str)).split(",")
    all_genres = [g.strip() for g in all_genres if g.strip()]
    genre_freq = pd.Series(all_genres).value_counts().to_dict()

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
    )
    wc.generate_from_frequencies(genre_freq)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("动漫类型词云", fontsize=16)
    fig.tight_layout()

    path = os.path.join(save_dir, "genre_wordcloud.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_avg_rating_by_type(df: pd.DataFrame, save_dir: str) -> None:
    """绘制不同制作类型的平均评分对比条形图。"""
    type_rating = df.dropna(subset=["type", "rating"])
    avg_by_type = type_rating.groupby("type")["rating"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    avg_by_type.plot(kind="barh", color="lightcoral", ax=ax)
    ax.set_xlabel("平均评分")
    ax.set_ylabel("制作类型")
    ax.set_title("不同制作类型的平均评分对比")

    for index, value in enumerate(avg_by_type):
        ax.text(value, index, f"{value:.2f}", va="center")

    fig.tight_layout()

    path = os.path.join(save_dir, "avg_rating_by_type.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved: %s", path)


def main() -> None:
    """生成所有可视化图表。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_dirs()

    logger.info("Loading data...")
    df = pd.read_csv(DATA_FILE)

    setup_plot_style()

    plot_rating_distribution(df, FIGURE_DIR)
    plot_genre_wordcloud(df, FIGURE_DIR)
    plot_avg_rating_by_type(df, FIGURE_DIR)

    logger.info("All visualizations generated in %s", FIGURE_DIR)


if __name__ == "__main__":
    main()
