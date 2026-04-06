import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import warnings
from itertools import combinations
import random
import math

warnings.filterwarnings('ignore')

# ------------------------------
# 1. 数据加载与预处理
# ------------------------------
df = pd.read_csv('./data/anime.csv')
df['episodes'] = df['episodes'].replace('Unknown', np.nan)
df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
df['rating'] = df['rating'].fillna(df['rating'].median())
df['members'] = df['members'].fillna(0)

print(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")

# ------------------------------
# 1.1 名称文本向量化
# ------------------------------
print("Encoding anime names...")
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    name_embeddings = embedder.encode(df['name'].tolist(), show_progress_bar=True)
    name_embeddings = name_embeddings.astype(np.float32)
    print("Using Sentence-BERT embeddings.")
    pca = PCA(n_components=128)
    name_embeddings_pca = pca.fit_transform(name_embeddings)
    name_embeddings_sparse = sp.csr_matrix(name_embeddings_pca)
except Exception as e:
    print(f"Sentence-BERT failed: {e}")
    print("Falling back to TF-IDF + TruncatedSVD.")
    vectorizer = TfidfVectorizer(max_features=384)
    name_embeddings_sparse = vectorizer.fit_transform(df['name'].tolist())
    svd = TruncatedSVD(n_components=128, random_state=42)
    name_embeddings_pca = svd.fit_transform(name_embeddings_sparse)
    name_embeddings_sparse = sp.csr_matrix(name_embeddings_pca)

print(f"Name features shape: {name_embeddings_sparse.shape}")

# ------------------------------
# 1.2 流派处理
# ------------------------------
df['genre'] = df['genre'].fillna('').apply(lambda x: x.split(', ') if x else '')
mlb = MultiLabelBinarizer(sparse_output=True)
genre_sparse = mlb.fit_transform(df['genre'])
print(f"Genre features shape: {genre_sparse.shape}")

all_genres = mlb.classes_
genre_to_idx = {g: i for i, g in enumerate(all_genres)}
genre_indices = [torch.tensor([genre_to_idx[g] for g in genres if g]) for genres in df['genre']]
max_genres = max(len(indices) for indices in genre_indices)
genre_indices_padded = torch.zeros((len(df), max_genres), dtype=torch.long)
genre_mask = torch.zeros((len(df), max_genres), dtype=torch.bool)
for i, indices in enumerate(genre_indices):
    length = len(indices)
    genre_indices_padded[i, :length] = indices
    genre_mask[i, :length] = True

# ------------------------------
# 1.3 其他特征及特征交叉
# ------------------------------
type_encoder = OneHotEncoder(sparse_output=True)
type_sparse = type_encoder.fit_transform(df[['type']])
print(f"Type features shape: {type_sparse.shape}")

type_dense = type_sparse.toarray().astype(np.float32)

episodes_median = df['episodes'].median()
df['episodes'] = df['episodes'].fillna(episodes_median)
df['log_members'] = np.log1p(df['members'])

num_features = ['episodes', 'log_members']
scaler = StandardScaler()
num_dense = scaler.fit_transform(df[num_features])
num_sparse = sp.csr_matrix(num_dense)
print(f"Numerical features shape: {num_sparse.shape}")

# 特征交叉
type_onehot = type_dense
num_cross = np.hstack([
    type_onehot * num_dense[:, 0:1],
    type_onehot * num_dense[:, 1:2]
])
print(f"Cross features shape: {num_cross.shape}")

target = df['rating'].values.astype(np.float32)

node_indices = np.arange(len(df))
train_idx, test_idx = train_test_split(node_indices, test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

# ------------------------------
# 2. 构建无权图（移除权重相关代码）
# ------------------------------
# 2.1 流派边：动态采样
print("Building genre edges with adaptive sampling...")
genre_to_anime = {}
for i, genres in enumerate(df['genre']):
    for g in genres:
        if g not in genre_to_anime:
            genre_to_anime[g] = []
        genre_to_anime[g].append(i)

edges = set()  # 使用 set 自动去重
max_pairs_per_genre = 80
random.seed(42)
for genre, anime_list in genre_to_anime.items():
    if len(anime_list) < 2:
        continue
    total_pairs = len(anime_list) * (len(anime_list) - 1) // 2
    sample_size = min(max_pairs_per_genre, int(math.sqrt(total_pairs)) * 8)
    if total_pairs <= sample_size:
        sampled_pairs = combinations(anime_list, 2)
    else:
        sampled_pairs = random.sample(list(combinations(anime_list, 2)), sample_size)
    for u, v in sampled_pairs:
        edges.add((u, v))
        edges.add((v, u))

# 2.2 kNN 边（无权）
print("Building similarity-based kNN graph...")
X_sim = np.hstack([name_embeddings_pca, type_dense, num_dense])
scaler_sim = StandardScaler()
X_sim_scaled = scaler_sim.fit_transform(X_sim)

k = 15
knn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine')
knn_model.fit(X_sim_scaled[train_idx])
distances, indices = knn_model.kneighbors(X_sim_scaled)

for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:
        if i != j:
            edges.add((i, j))
            edges.add((j, i))

print(f"Total edges after augmentation: {len(edges)}")
edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
print(f"Edge index shape: {edge_index.shape}")

# ------------------------------
# 3. 创建 PyG Data 对象（无 edge_attr）
# ------------------------------
name_dense = name_embeddings_sparse.toarray().astype(np.float32)
fixed_features = np.hstack([name_dense, type_dense, num_dense, num_cross])
print(f"Final node features shape: {fixed_features.shape}")

data = Data(
    x=torch.tensor(fixed_features, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(target, dtype=torch.float).view(-1, 1),
    genre_ids=genre_indices_padded,
    genre_mask=genre_mask
)

train_mask = torch.zeros(len(df), dtype=torch.bool)
val_mask = torch.zeros(len(df), dtype=torch.bool)
test_mask = torch.zeros(len(df), dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# ------------------------------
# 4. GNN 模型（去掉权重参数）
# ------------------------------
class ImprovedGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_genres, genre_embed_dim=32, max_genres=10,
                 num_layers=2, dropout=0.4):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_genres = max_genres
        self.genre_embed = nn.Embedding(num_genres, genre_embed_dim)

        final_in_channels = in_channels + genre_embed_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = final_in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels
            self.convs.append(SAGEConv(in_dim, out_dim))
            self.bns.append(nn.BatchNorm1d(out_dim))

        self.lin = nn.Linear(hidden_channels, 1)
        self.register_buffer('min_rating', torch.tensor(df['rating'].min()))
        self.register_buffer('max_rating', torch.tensor(df['rating'].max()))

    def forward(self, x, edge_index, genre_ids, genre_mask):
        # 流派嵌入
        genre_emb = self.genre_embed(genre_ids)
        masked_emb = genre_emb * genre_mask.unsqueeze(-1).float()
        genre_pooled = masked_emb.sum(dim=1) / (genre_mask.sum(dim=1, keepdim=True).float() + 1e-8)
        genre_pooled = torch.where(genre_mask.sum(dim=1, keepdim=True) > 0,
                                   genre_pooled,
                                   torch.zeros_like(genre_pooled))

        x = torch.cat([x, genre_pooled], dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.lin(x)
        out = torch.sigmoid(out)
        out = self.min_rating + (self.max_rating - self.min_rating) * out
        return out

# ------------------------------
# 5. 训练 GNN
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=1024,
    shuffle=True,
    input_nodes=data.train_mask,
)

val_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=1024,
    shuffle=False,
    input_nodes=data.val_mask,
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=1024,
    shuffle=False,
    input_nodes=data.test_mask,
)

model = ImprovedGNN(
    in_channels=data.x.size(1),
    hidden_channels=128,
    out_channels=1,
    num_genres=len(all_genres),
    genre_embed_dim=32,
    max_genres=max_genres,
    num_layers=2,
    dropout=0.4
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

def train_one_epoch():
    model.train()
    total_loss = 0
    total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.genre_ids, batch.genre_mask)
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        total_nodes += batch.batch_size
    return total_loss / total_nodes

def evaluate(loader, return_pred=False):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.genre_ids, batch.genre_mask)
            preds.append(out[:batch.batch_size].cpu().numpy())
            trues.append(batch.y[:batch.batch_size].cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    if return_pred:
        return rmse, mae, pred
    return rmse, mae

best_val_rmse = float('inf')
patience = 20
counter = 0

for epoch in range(1, 301):
    loss = train_one_epoch()
    train_rmse, _ = evaluate(train_loader)
    val_rmse, _ = evaluate(val_loader)

    scheduler.step(val_rmse)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), 'best_gnn.pt')
        counter = 0
        print(f'Epoch {epoch:03d}: New best model saved (Val RMSE: {val_rmse:.4f})')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}')

model.load_state_dict(torch.load('best_gnn.pt'))
test_rmse_gnn, test_mae_gnn, gnn_pred = evaluate(test_loader, return_pred=True)
print(f'Improved GNN Test RMSE: {test_rmse_gnn:.4f}, MAE: {test_mae_gnn:.4f}')

# ------------------------------
# 6. 对比实验（与之前相同）
# ------------------------------
print("\n--- 对比实验（基于稀疏特征）---")
X_advanced_sparse = sp.hstack([name_embeddings_sparse, type_sparse, genre_sparse, num_sparse], format='csr')
y = target

X_train_adv = X_advanced_sparse[train_idx]
y_train_adv = y[train_idx]
X_val_adv = X_advanced_sparse[val_idx]
y_val_adv = y[val_idx]
X_test_adv = X_advanced_sparse[test_idx]
y_test_adv = y[test_idx]

results = {}

def train_sklearn_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    results[name] = {'RMSE': rmse, 'MAE': mae}
    print(f'{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}')

train_sklearn_model(LinearRegression(), X_train_adv, y_train_adv, X_test_adv, y_test_adv, 'Linear Regression')
train_sklearn_model(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    X_train_adv, y_train_adv, X_test_adv, y_test_adv, 'Random Forest')
train_sklearn_model(xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    X_train_adv, y_train_adv, X_test_adv, y_test_adv, 'XGBoost')
train_sklearn_model(KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
                    X_train_adv, y_train_adv, X_test_adv, y_test_adv, 'kNN')

results['Improved GNN'] = {'RMSE': test_rmse_gnn, 'MAE': test_mae_gnn}

print("\n=== 对比实验结果 ===")
for name, metrics in results.items():
    print(f"{name:30} RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")

# ------------------------------
# 7. 消融实验
# ------------------------------
print("\n--- 消融实验：移除名称向量 ---")
X_basic_sparse = sp.hstack([type_sparse, genre_sparse, num_sparse], format='csr')
X_train_basic = X_basic_sparse[train_idx]
y_train_basic = y[train_idx]
X_test_basic = X_basic_sparse[test_idx]
y_test_basic = y[test_idx]

rf_basic = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_basic.fit(X_train_basic, y_train_basic)
pred_basic = rf_basic.predict(X_test_basic)
rmse_basic = np.sqrt(mean_squared_error(y_test_basic, pred_basic))
mae_basic = mean_absolute_error(y_test_basic, pred_basic)
print(f'Basic features (no name) - RMSE: {rmse_basic:.4f}, MAE: {mae_basic:.4f}')

print("\n--- 消融实验：移除图结构 ---")
xgb_adv = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_adv.fit(X_train_adv, y_train_adv)
pred_xgb = xgb_adv.predict(X_test_adv)
rmse_xgb = np.sqrt(mean_squared_error(y_test_adv, pred_xgb))
mae_xgb = mean_absolute_error(y_test_adv, pred_xgb)
print(f'XGBoost (adv, no graph) - RMSE: {rmse_xgb:.4f}, MAE: {mae_xgb:.4f}')

# ------------------------------
# 8. 集成学习
# ------------------------------
print("\n--- 集成学习：GNN + Random Forest ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_adv, y_train_adv)
rf_pred = rf_model.predict(X_test_adv)

for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
    ensemble_pred = alpha * gnn_pred.flatten() + (1 - alpha) * rf_pred
    rmse = np.sqrt(mean_squared_error(y_test_adv, ensemble_pred))
    print(f"Alpha={alpha}: RMSE={rmse:.4f}")

# ------------------------------
# 9. 保存结果
# ------------------------------
pd.DataFrame(results).T.to_csv('comparison_results_final.csv')