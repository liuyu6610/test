# 基于深度学习的动漫评分数据分析研究

> 毕业设计项目 — 使用图神经网络 (GNN) 预测动漫评分

## 项目简介

本项目基于 [MyAnimeList](https://myanimelist.net/) 的 12,294 部动漫数据，构建图神经网络模型来预测动漫评分。系统采用 Sentence-BERT 语义嵌入 + 多维特征交叉 + GraphSAGE 图卷积的技术方案，在测试集上取得了 **RMSE 0.6519** 的预测精度，优于所有传统机器学习基线。

## 目录结构

```
毕业设计/
├── data/                       # 原始数据
│   └── anime.csv               # MyAnimeList 数据集 (12,294 条)
│
├── src/                        # 核心源码（模块化架构）
│   ├── data/                   # 数据处理层
│   │   ├── loader.py           #   数据加载与清洗
│   │   └── features.py         #   特征工程（SBERT/TF-IDF、特征交叉）
│   ├── graph/                  # 图构建层
│   │   └── builder.py          #   流派共现边 + kNN 相似性边
│   ├── models/                 # 模型定义层
│   │   └── gnn.py              #   AnimeGNN（支持 GCN/SAGE/GAT）
│   ├── training/               # 训练层
│   │   └── trainer.py          #   NeighborLoader 图采样训练器
│   └── evaluation/             # 评估层
│       ├── baselines.py        #   基线对比实验
│       └── ablation.py         #   消融实验
│
├── scripts/                    # 可执行脚本
│   ├── train.py                # 主训练入口（含 CLI 参数配置）
│   ├── visualize.py            # 数据可视化（分布图、词云）
│   └── analyze.py              # 受众分析（帕累托分布）
│
├── config.py                   # 集中式超参数配置
│
├── outputs/                    # 运行输出（自动生成）
│   ├── results/                #   实验结果 CSV
│   ├── figures/                #   可视化图表
│   ├── models/                 #   模型权重 (.pt)
│   └── cache/                  #   SBERT 嵌入缓存 (.npy)
│
├── legacy/                     # 历史参考代码
│   └── z4.py                   # 早期单文件版本（NeighborLoader 原型）
│
├── docs/                       # 论文相关文档
│
├── 一键运行.bat                 # Windows 一键运行脚本
├── requirements.txt            # Python 依赖
├── .gitignore                  # Git 忽略规则
└── README.md                   # 本文件
```

## 快速开始

### 环境要求

- Python 3.12+
- PyTorch 2.11+
- torch-geometric, pyg-lib, torch-sparse

### 安装依赖

```bash
pip install -r requirements.txt
pip install pyg-lib torch-sparse -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

### 一键运行

```bash
# Windows
一键运行.bat

# 或手动执行
python scripts/visualize.py      # 数据可视化
python scripts/analyze.py        # 受众分析
python scripts/train.py          # GNN 训练 + 基线对比 + 消融实验
```

### 命令行参数

```bash
python scripts/train.py --conv_type sage --hidden 128 --dropout 0.4 --epochs 300 --patience 30
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--conv_type` | `sage` | 图卷积类型 (gcn / sage / gat) |
| `--hidden` | `128` | 隐层维度 |
| `--dropout` | `0.4` | Dropout 比率 |
| `--lr` | `0.001` | 学习率 |
| `--epochs` | `300` | 最大训练轮数 |
| `--patience` | `30` | 早停耐心值 |
| `--skip_baselines` | `false` | 跳过基线对比实验 |
| `--skip_ablation` | `false` | 跳过消融实验 |

## 实验结果

### 基线对比

| 模型 | RMSE ↓ | MAE ↓ |
|---|---|---|
| Linear Regression | 0.7030 | 0.5192 |
| XGBoost | 0.6879 | 0.5041 |
| kNN | 0.6754 | 0.4993 |
| Random Forest | 0.6647 | 0.4810 |
| **GNN (Ours)** | **0.6519** | **0.4776** |

### 消融实验

| 实验设置 | RMSE | MAE |
|---|---|---|
| 移除名称文本特征 (RF) | 0.6827 | 0.4862 |
| 移除图结构 (XGBoost) | 0.6879 | 0.5041 |

## 技术栈

- **图神经网络**: PyTorch Geometric (GraphSAGE + NeighborLoader)
- **文本嵌入**: Sentence-BERT (all-MiniLM-L6-v2) + PCA 降维
- **特征工程**: 类型×数值 特征交叉、自适应流派图采样
- **训练优化**: AdamW、SmoothL1Loss、Edge Dropout、Kaiming 初始化、残差连接
- **基线模型**: scikit-learn (LR, RF, kNN) + XGBoost

## 核心功能特性

### 🎯 数据处理
- ✅ 自动化数据清洗与缺失值填充
- ✅ 多层次数据集划分 (训练/验证/测试 = 64%/16%/20%)
- ✅ Sentence-BERT 语义嵌入 + TF-IDF 文本特征
- ✅ 数值型特征归一化处理

### 🕸️ 图构建
- ✅ **流派共现边**: 基于动漫流派相似性构建异构图
- ✅ **kNN 相似性边**: 基于多维特征向量的近邻连接
- ✅ 自适应图采样策略，支持大规模图训练

### 🧠 模型架构
- ✅ 支持多种图卷积层 (GCN / GraphSAGE / GAT)
- ✅ 残差连接 + 多层堆叠
- ✅  Genre 嵌入模块，捕捉流派语义
- ✅ 输出范围约束，匹配训练集评分分布

### ⚡ 训练优化
- ✅ NeighborLoader 图采样，降低显存占用
- ✅ AdamW 优化器 + 权重衰减
- ✅ ReduceLROnPlateau 学习率调度
- ✅ 早停机制 + 梯度裁剪
- ✅ 自动保存最优模型权重

## 详细使用指南

### 数据可视化

生成动漫数据分布图表：

```bash
python scripts/visualize.py
```

**输出内容：**
- `rating_distribution.png` - 评分分布直方图
- `avg_rating_by_type.png` - 不同类型平均评分对比
- `genre_wordcloud.png` - 流派词云图
- `members_cumulative_distribution.png` - 观看人数累积分布

### 受众分析

执行帕累托分布分析：

```bash
python scripts/analyze.py
```

**分析内容：**
- TOP 流派集中度分析
- 长尾分布可视化
- 受众偏好统计

### 模型训练

**基础用法：**
```bash
python scripts/train.py
```

**自定义配置：**
```bash
# 使用 GAT 模型，隐层维度 256，训练 500 轮
python scripts/train.py --conv_type gat --hidden 256 --epochs 500

# 跳过基线和消融实验（仅训练 GNN）
python scripts/train.py --skip_baselines --skip_ablation

# 调整批大小和邻居采样
python scripts/train.py --batch_size 512 --layers 3
```

### 查看实验结果

训练完成后，在 `outputs/` 目录下查看：

```
outputs/
├── results/
│   ├── comparison_results.csv    # GNN vs 基线模型对比
│   └── ablation_results.csv      # 消融实验结果
├── models/
│   └── best_gnn.pt               # 最优模型权重
└── figures/                      # 可视化图表
```

## 模型性能详情

### 不同卷积类型对比

| 模型 | RMSE ↓ | MAE ↓ | 参数量 |
|------|--------|-------|--------|
| GCN | 0.6589 | 0.4801 | 中等 |
| **GraphSAGE** | **0.6519** | **0.4776** | 较小 |
| GAT | 0.6545 | 0.4789 | 较大 |

### 推荐超参数配置

**最佳配置（已设为默认）：**
```python
{
    "conv_type": "sage",
    "hidden_channels": 128,
    "num_layers": 2,
    "dropout": 0.4,
    "learning_rate": 0.001,
    "batch_size": 1024,
    "patience": 30
}
```

## 常见问题 FAQ

### Q: 国内无法下载 Sentence-BERT 模型？
A: 项目已配置 Hugging Face 镜像源 (`hf-mirror.com`)，如仍有问题可手动设置环境变量：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: CUDA 版本不兼容？
A: 当前默认安装 CPU 版本。如需 GPU 版本，请参考 PyTorch 官网安装对应 CUDA 版本的 torch。

### Q: 如何修改数据集划分比例？
A: 编辑 `config.py` 中的 `DataConfig` 类：
```python
@dataclass
class DataConfig:
    test_size: float = 0.2  # 测试集比例
    val_size: float = 0.2   # 验证集比例
```

### Q: 训练速度太慢？
A: 可尝试以下优化：
1. 减小 `batch_size`（如改为 512）
2. 减少 `num_neighbors` 采样数量
3. 使用 GPU 加速
4. 减少 `max_epochs` 或增大 `patience`

## 项目架构说明

```
src/
├── data/              # 数据处理层
│   ├── loader.py      # 数据加载、清洗、划分
│   └── features.py    # 特征工程（SBERT、TF-IDF、特征交叉）
├── graph/             # 图构建层
│   └── builder.py     # 流派边 + kNN 边构建
├── models/            # 模型定义层
│   └── gnn.py         # AnimeGNN 模型（多卷积层支持）
├── training/          # 训练层
│   └── trainer.py     # 训练器（采样、早停、LR 调度）
└── evaluation/        # 评估层
    ├── baselines.py   # 基线模型实现
    └── ablation.py    # 消融实验逻辑
```

## 依赖说明

### 必需依赖
详见 [`requirements.txt`](requirements.txt)，主要包含：
- PyTorch 生态（torch, torch-geometric）
- 机器学习库（scikit-learn, xgboost）
- 数据处理（pandas, numpy, scipy）
- 文本处理（sentence-transformers）
- 可视化（matplotlib, seaborn, wordcloud）

### PyG 特殊依赖
需单独安装 PyTorch Geometric 的编译扩展：
```bash
pip install pyg-lib torch-sparse -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

**GPU 版本替换：**
将 URL 中的 `cpu` 改为对应的 CUDA 版本（如 `cu118`）。

## 许可证与引用

本项目为毕业设计作品，仅供学术交流使用。

**论文题目：** 基于深度学习的动漫评分数据分析研究  
**数据来源：** MyAnimeList (https://myanimelist.net/)

## 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 📧 Email: [你的邮箱]
- 💬 微信：[你的微信号]
- 🐛 Issues: 提交 GitHub Issue

---

*Last Updated: 2026-04-02*
