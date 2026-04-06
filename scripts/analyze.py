"""
受众累积分布分析脚本。

分析 members 字段的帕累托效应（二八定律），
输出头部动漫的受众集中度统计与累积分布曲线。

用法：
    python scripts/analyze.py
"""

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_FILE, FIGURE_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def analyze_members_distribution(df: pd.DataFrame, save_dir: str) -> None:
    """
    分析 members 的累积分布并生成曲线图。

    计算头部动漫贡献的受众占比，绘制帕累托曲线。
    """
    df_clean = df.dropna(subset=["members"])
    df_sorted = df_clean.sort_values("members", ascending=False).reset_index(drop=True)

    logger.info("Top 10 anime by members:")
    for _, row in df_sorted.head(10).iterrows():
        logger.info("  %s: %d", row.get("name", "N/A"), int(row["members"]))

    total_members = df_sorted["members"].sum()
    df_sorted["cumsum"] = df_sorted["members"].cumsum()
    df_sorted["cum_percent"] = df_sorted["cumsum"] / total_members * 100

    # 绘制累积分布曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_sorted.index, df_sorted["cum_percent"])
    ax.set_xlabel("动漫数量（按 members 降序）")
    ax.set_ylabel("累积 members 占比 (%)")
    ax.set_title("members 的累积分布曲线")
    ax.grid(True)
    ax.axhline(y=80, color="r", linestyle="--", label="80% 线")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(save_dir, "members_cumulative_distribution.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved: %s", path)

    # 计算 80% 阈值
    threshold_idx = (df_sorted["cum_percent"] >= 80).idxmax()
    num_anime_80 = threshold_idx + 1
    pct_anime_80 = num_anime_80 / len(df_sorted) * 100
    logger.info(
        "达到 80%% 总 members 需要前 %d 部动漫（占总数的 %.2f%%）",
        num_anime_80,
        pct_anime_80,
    )


def main() -> None:
    """执行受众分布分析。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_dirs()

    logger.info("Loading data...")
    df = pd.read_csv(DATA_FILE)

    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    analyze_members_distribution(df, FIGURE_DIR)
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
