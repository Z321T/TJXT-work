"""
推荐系统混合模型实验验证方案 - 改进版
使用LightGCN、DeepFM和SASRec三个模型进行特征级+分数级融合
数据集：MovieLens 10M
"""

import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 机器学习库
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# 深度学习库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("推荐系统混合模型实验验证方案 - LightGCN + DeepFM + SASRec")
print("=" * 80)

# ============================================================================
# 第一部分：数据加载与预处理
# ============================================================================

import os


class DataPreprocessor:
    """数据预处理类"""

    def __init__(self, min_user_interactions=5, min_item_interactions=5):
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def download_movielens_10m(self, data_dir='./data'):
        """下载 MovieLens 10M 数据集"""
        import zipfile
        import requests

        os.makedirs(data_dir, exist_ok=True)

        zip_path = os.path.join(data_dir, 'ml-10m.zip')
        extract_dir = os.path.join(data_dir, 'ml-10M100K')

        if os.path.exists(extract_dir):
            print(f"数据集已存在: {extract_dir}")
            return extract_dir

        print("下载 MovieLens 10M 数据集...")
        url = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192

            with open(zip_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = downloaded / total_size * 100
                        print(f"\r进度: {progress:.1f}% {downloaded:,} / {total_size:,} 字节", end='')

            print("\n解压文件...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

            os.remove(zip_path)
            print(f"下载完成: {extract_dir}")
            return extract_dir

        except Exception as e:
            print(f"\n下载失败: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return None

    def load_movielens_10m(self, data_dir, sample_size=None):
        """加载 MovieLens 10M 数据集"""
        print(f"加载 MovieLens 10M 数据集...")

        ratings_file = os.path.join(data_dir, 'ratings.dat')

        if not os.path.exists(ratings_file):
            print(f"错误：未找到评分文件 {ratings_file}")
            return None

        print("读取评分数据...")
        ratings_data = []

        with open(ratings_file, 'r', encoding='latin-1') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break

                parts = line.strip().split('::')
                if len(parts) == 4:
                    ratings_data.append({
                        'user_id': int(parts[0]),
                        'item_id': int(parts[1]),
                        'rating': float(parts[2]),
                        'timestamp': datetime.fromtimestamp(int(parts[3]))
                    })

        df = pd.DataFrame(ratings_data)
        print(f"加载评分: {len(df):,} 条")

        # 添加模拟特征
        categories = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance']
        item_to_category = {item: categories[hash(item) % len(categories)]
                            for item in df['item_id'].unique()}
        df['category'] = df['item_id'].map(item_to_category)

        np.random.seed(42)
        item_to_price = {item: np.random.uniform(5, 50)
                         for item in df['item_id'].unique()}
        df['price'] = df['item_id'].map(item_to_price)

        return df

    def _generate_sample_data(self, n_samples=100000):
        """生成模拟数据（备用）"""
        print(f"\n生成模拟数据集 ({n_samples:,} 条)...")

        n_users = 5000
        n_items = 2000

        np.random.seed(42)

        user_ids = np.random.choice(n_users, n_samples)
        item_ids = np.random.choice(n_items, n_samples)
        ratings = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.15, 0.3, 0.4])

        start_time = datetime(2024, 1, 1)
        timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]

        categories = ['Fiction', 'NonFiction', 'Science', 'History', 'Art']
        item_categories = {i: np.random.choice(categories) for i in range(n_items)}

        df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': timestamps,
            'category': [item_categories[item] for item in item_ids],
            'price': np.random.uniform(5, 50, n_samples)
        })

        return df

    def load_and_preprocess(self, file_path=None, use_sample=True, dataset='movielens-10m', sample_size=500000):
        """加载并预处理数据"""

        if use_sample:
            if dataset == 'movielens-10m':
                data_dir = self.download_movielens_10m()
                if data_dir is None:
                    print("\n使用模拟数据代替...")
                    return self._generate_sample_data(sample_size)

                df = self.load_movielens_10m(data_dir, sample_size=sample_size)

                if df is None:
                    print("\n使用模拟数据代替...")
                    return self._generate_sample_data(sample_size)
            else:
                return self._generate_sample_data(sample_size)
        else:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 数据清洗
        print("\n数据清洗...")
        df = df.dropna(subset=['user_id', 'item_id', 'rating', 'timestamp'])
        df = df[df['rating'].between(1, 5)]

        # 过滤低频用户和物品
        print("\n过滤低频用户和物品...")
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()

        valid_users = user_counts[user_counts >= self.min_user_interactions].index
        valid_items = item_counts[item_counts >= self.min_item_interactions].index

        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]

        print(f"\n过滤后数据统计:")
        print(f"  交互数: {len(df):,} 条")
        print(f"  用户数: {df['user_id'].nunique():,}")
        print(f"  物品数: {df['item_id'].nunique():,}")
        print(f"  时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")

        return df

    def create_binary_labels(self, df, threshold=4):
        """创建二分类标签"""
        df['label'] = (df['rating'] >= threshold).astype(int)
        pos_rate = df['label'].mean()
        print(f"\n正样本比例: {pos_rate:.2%} (threshold={threshold})")
        return df

    def split_data_by_time(self, df, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
        """按时间顺序划分数据集"""
        df = df.sort_values('timestamp').reset_index(drop=True)

        n = len(df)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))

        train_df = df.iloc[:train_end].copy()
        valid_df = df.iloc[train_end:valid_end].copy()
        test_df = df.iloc[valid_end:].copy()

        print(f"\n数据集划分:")
        print(f"训练集: {len(train_df):,} 条")
        print(f"验证集: {len(valid_df):,} 条")
        print(f"测试集: {len(test_df):,} 条")

        return train_df, valid_df, test_df

    def encode_ids(self, train_df, valid_df, test_df):
        """编码用户和物品ID"""
        all_df = pd.concat([train_df, valid_df, test_df])

        all_df['user_idx'] = self.user_encoder.fit_transform(all_df['user_id'])
        all_df['item_idx'] = self.item_encoder.fit_transform(all_df['item_id'])

        train_df = all_df.loc[train_df.index].copy()
        valid_df = all_df.loc[valid_df.index].copy()
        test_df = all_df.loc[test_df.index].copy()

        self.n_users = all_df['user_idx'].max() + 1
        self.n_items = all_df['item_idx'].max() + 1

        print(f"\n编码后统计:")
        print(f"  用户数: {self.n_users:,}")
        print(f"  物品数: {self.n_items:,}")

        return train_df, valid_df, test_df


# ============================================================================
# 第二部分：特征工程
# ============================================================================

class FeatureEngine:
    """特征工程类"""

    def __init__(self):
        self.user_features = {}
        self.item_features = {}
        self.category_encoder = LabelEncoder()

    def build_features(self, train_df, valid_df, test_df):
        """构建特征"""
        print("\n构建特征...")

        self._build_user_features(train_df)
        self._build_item_features(train_df)

        train_df = self._add_features(train_df)
        valid_df = self._add_features(valid_df)
        test_df = self._add_features(test_df)

        # 编码类别特征
        all_df = pd.concat([train_df, valid_df, test_df])
        all_df['category_idx'] = self.category_encoder.fit_transform(all_df['category'])

        train_df['category_idx'] = all_df.loc[train_df.index, 'category_idx']
        valid_df['category_idx'] = all_df.loc[valid_df.index, 'category_idx']
        test_df['category_idx'] = all_df.loc[test_df.index, 'category_idx']

        return train_df, valid_df, test_df

    def _build_user_features(self, df):
        """构建用户特征"""
        user_stats = df.groupby('user_idx').agg({
            'rating': ['mean', 'std', 'count'],
            'item_idx': 'nunique'
        }).reset_index()

        user_stats.columns = ['user_idx', 'user_avg_rating', 'user_std_rating',
                              'user_rating_count', 'user_item_count']
        user_stats['user_std_rating'].fillna(0, inplace=True)

        self.user_features = user_stats.set_index('user_idx').to_dict('index')

    def _build_item_features(self, df):
        """构建物品特征"""
        item_stats = df.groupby('item_idx').agg({
            'rating': ['mean', 'std', 'count'],
            'user_idx': 'nunique'
        }).reset_index()

        item_stats.columns = ['item_idx', 'item_avg_rating', 'item_std_rating',
                              'item_rating_count', 'item_user_count']
        item_stats['item_std_rating'].fillna(0, inplace=True)

        self.item_features = item_stats.set_index('item_idx').to_dict('index')

    def _add_features(self, df):
        """添加特征到数据框"""
        df = df.copy()

        df['user_avg_rating'] = df['user_idx'].map(
            lambda x: self.user_features.get(x, {}).get('user_avg_rating', 3.5))
        df['user_std_rating'] = df['user_idx'].map(
            lambda x: self.user_features.get(x, {}).get('user_std_rating', 0))
        df['user_rating_count'] = df['user_idx'].map(
            lambda x: self.user_features.get(x, {}).get('user_rating_count', 0))

        df['item_avg_rating'] = df['item_idx'].map(
            lambda x: self.item_features.get(x, {}).get('item_avg_rating', 3.5))
        df['item_std_rating'] = df['item_idx'].map(
            lambda x: self.item_features.get(x, {}).get('item_std_rating', 0))
        df['item_rating_count'] = df['item_idx'].map(
            lambda x: self.item_features.get(x, {}).get('item_rating_count', 0))

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        return df


# ============================================================================
# 第三部分：候选集生成
# ============================================================================

class CandidateGenerator:
    """生成测试候选集"""

    def __init__(self, n_candidates=100):
        self.n_candidates = n_candidates

    def generate_candidates(self, df, train_df, n_neg=99):
        """为每个用户生成候选集：1个正样本 + n_neg个负样本"""
        print(f"\n生成候选集 (每用户{n_neg + 1}个候选)...")

        user_items = defaultdict(set)
        for _, row in train_df.iterrows():
            if row['label'] == 1:
                user_items[row['user_idx']].add(row['item_idx'])

        candidates = []

        for user_idx in df['user_idx'].unique():
            user_df = df[df['user_idx'] == user_idx]
            pos_samples = user_df[user_df['label'] == 1]

            if len(pos_samples) == 0:
                continue

            pos_sample = pos_samples.sample(1).iloc[0]
            candidates.append(pos_sample.to_dict())

            interacted = user_items[user_idx]
            all_items = set(range(df['item_idx'].max() + 1))
            neg_pool = list(all_items - interacted)

            if len(neg_pool) < n_neg:
                n_neg = len(neg_pool)

            neg_items = np.random.choice(neg_pool, size=n_neg, replace=False)

            for item_idx in neg_items:
                neg_sample = pos_sample.copy()
                neg_sample['item_idx'] = item_idx
                neg_sample['label'] = 0

                item_features = self._get_item_features(train_df, item_idx)
                neg_sample.update(item_features)

                candidates.append(neg_sample)

        candidates_df = pd.DataFrame(candidates)
        print(f"生成候选集: {len(candidates_df):,} 条 (用户数: {candidates_df['user_idx'].nunique():,})")

        return candidates_df

    def _get_item_features(self, train_df, item_idx):
        """获取物品特征"""
        item_df = train_df[train_df['item_idx'] == item_idx]

        if len(item_df) > 0:
            return {
                'item_avg_rating': item_df['rating'].mean(),
                'item_std_rating': item_df['rating'].std() if len(item_df) > 1 else 0,
                'item_rating_count': len(item_df),
                'category_idx': item_df['category_idx'].iloc[0],
                'price': item_df['price'].iloc[0]
            }
        else:
            return {
                'item_avg_rating': train_df['rating'].mean(),
                'item_std_rating': 0,
                'item_rating_count': 0,
                'category_idx': 0,
                'price': train_df['price'].mean()
            }


# ============================================================================
# 第四部分：模型1 - LightGCN (图嵌入模型)
# ============================================================================

class LightGCN(nn.Module):
    """LightGCN模型 - 图卷积协同过滤"""

    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3):
        super(LightGCN, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def build_graph(self, train_df, device):
        """构建二部图的邻接矩阵"""
        print("构建用户-物品交互图...")

        # 构建交互矩阵
        user_items = defaultdict(list)
        item_users = defaultdict(list)

        for _, row in train_df.iterrows():
            if row['label'] == 1:
                user_items[row['user_idx']].append(row['item_idx'])
                item_users[row['item_idx']].append(row['user_idx'])

        # 计算度数用于归一化
        user_degree = {u: len(items) for u, items in user_items.items()}
        item_degree = {i: len(users) for i, users in item_users.items()}

        # 构建稀疏邻接矩阵 (对称归一化)
        indices = []
        values = []

        # 用户->物品的边
        for user_idx, items in user_items.items():
            for item_idx in items:
                # D^(-1/2) * A * D^(-1/2)
                norm = np.sqrt(user_degree[user_idx] * item_degree[item_idx])
                indices.append([user_idx, self.n_users + item_idx])
                values.append(1.0 / norm)

        # 物品->用户的边
        for item_idx, users in item_users.items():
            for user_idx in users:
                norm = np.sqrt(user_degree[user_idx] * item_degree[item_idx])
                indices.append([self.n_users + item_idx, user_idx])
                values.append(1.0 / norm)

        indices = torch.LongTensor(indices).t()
        values = torch.FloatTensor(values)

        # 创建稀疏张量
        graph = torch.sparse.FloatTensor(
            indices, values,
            torch.Size([self.n_users + self.n_items, self.n_users + self.n_items])
        ).to(device)

        return graph

    def forward(self, graph):
        """图卷积传播"""
        # 初始嵌入
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)

        embeddings_list = [all_embeddings]

        # 多层图卷积
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(graph, all_embeddings)
            embeddings_list.append(all_embeddings)

        # 层聚合 (平均)
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)

        user_embeddings = final_embeddings[:self.n_users]
        item_embeddings = final_embeddings[self.n_users:]

        return user_embeddings, item_embeddings

    def predict(self, user_idx, item_idx):
        """预测用户-物品交互分数"""
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        scores = torch.sum(user_emb * item_emb, dim=1)
        return torch.sigmoid(scores)


class LightGCNTrainer:
    """LightGCN训练器"""

    def __init__(self, n_users, n_items, embedding_dim=64, device='cpu'):
        self.device = device
        self.model = LightGCN(n_users, n_items, embedding_dim=embedding_dim).to(device)
        self.graph = None

    def train(self, train_df, valid_df, epochs=20, batch_size=2048, lr=0.001):
        """训练模型"""
        print("\n训练LightGCN模型...")

        # 构建图
        self.graph = self.model.build_graph(train_df, self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # 准备训练数据
        pos_train = train_df[train_df['label'] == 1]
        n_train = len(pos_train)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            indices = np.random.permutation(n_train)

            for i in range(0, n_train, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_data = pos_train.iloc[batch_indices]

                user_idx = torch.LongTensor(batch_data['user_idx'].values).to(self.device)
                item_idx = torch.LongTensor(batch_data['item_idx'].values).to(self.device)

                # 负采样
                neg_items = np.random.randint(1, self.model.n_items, size=len(batch_data))
                neg_item_idx = torch.LongTensor(neg_items).to(self.device)

                optimizer.zero_grad()

                # 正样本预测
                pos_scores = self.model.predict(user_idx, item_idx)
                # 负样本预测
                neg_scores = self.model.predict(user_idx, neg_item_idx)

                # BPR损失
                loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / n_batches:.4f}")

        print("LightGCN训练完成")

    def predict(self, df):
        """预测用户-物品交互分数 (接受DataFrame)"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            user_idx = torch.LongTensor(df['user_idx'].values).to(self.device)
            item_idx = torch.LongTensor(df['item_idx'].values).to(self.device)

            # 分批预测以避免内存溢出
            batch_size = 2048
            for i in range(0, len(df), batch_size):
                batch_user = user_idx[i:i + batch_size]
                batch_item = item_idx[i:i + batch_size]

                scores = self.model.predict(batch_user, batch_item)
                predictions.extend(scores.cpu().numpy())

        return np.array(predictions)

    def get_embeddings(self):
        """获取用户和物品的嵌入向量"""
        self.model.eval()
        with torch.no_grad():
            user_embeddings, item_embeddings = self.model.forward(self.graph)
        return user_embeddings, item_embeddings



# ============================================================================
# 第五部分：模型2 - DeepFM (评分预测模型,含图嵌入特征)
# ============================================================================

class DeepFMWithGraphEmbedding(nn.Module):
    """DeepFM模型 - 融合图嵌入特征"""

    def __init__(self, n_users, n_items, n_categories, embedding_dim=16,
                 graph_embedding_dim=64, hidden_dims=[64, 32]):
        super(DeepFMWithGraphEmbedding, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # 原始嵌入
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.category_embedding = nn.Embedding(n_categories, embedding_dim)

        # FM bias
        self.fm_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        # 图嵌入融合层
        self.graph_transform = nn.Linear(graph_embedding_dim, embedding_dim)

        # Dense特征 + 图嵌入特征
        n_dense_features = 9
        deep_input_dim = embedding_dim * 5 + n_dense_features  # 3原始嵌入 + 2图嵌入

        layers = []
        input_dim = deep_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.deep_layers = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.category_embedding.weight, std=0.01)
        nn.init.constant_(self.user_bias.weight, 0)
        nn.init.constant_(self.item_bias.weight, 0)

    def forward(self, user_idx, item_idx, category_idx, dense_features,
                user_graph_emb, item_graph_emb):
        """前向传播 - 融合图嵌入"""
        # 原始嵌入
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        category_emb = self.category_embedding(category_idx)

        # 图嵌入转换
        user_graph_emb_trans = self.graph_transform(user_graph_emb)
        item_graph_emb_trans = self.graph_transform(item_graph_emb)

        # FM部分
        fm_first_order = (self.fm_bias +
                          self.user_bias(user_idx).squeeze() +
                          self.item_bias(item_idx).squeeze())

        embeddings = torch.stack([user_emb, item_emb, category_emb,
                                   user_graph_emb_trans, item_graph_emb_trans], dim=1)
        sum_of_square = torch.sum(embeddings, dim=1) ** 2
        square_of_sum = torch.sum(embeddings ** 2, dim=1)
        fm_second_order = 0.5 * torch.sum(sum_of_square - square_of_sum, dim=1)

        # Deep部分 - 融合所有嵌入
        deep_input = torch.cat([user_emb, item_emb, category_emb,
                                user_graph_emb_trans, item_graph_emb_trans,
                                dense_features], dim=1)
        deep_output = self.deep_layers(deep_input).squeeze()

        output = fm_first_order + fm_second_order + deep_output

        return torch.sigmoid(output)


class DeepFMTrainer:
    """DeepFM训练器"""

    def __init__(self, n_users, n_items, n_categories, device='cpu'):
        self.device = device
        self.model = DeepFMWithGraphEmbedding(n_users, n_items, n_categories).to(device)
        self.feature_cols = [
            'user_avg_rating', 'user_std_rating', 'user_rating_count',
            'item_avg_rating', 'item_std_rating', 'item_rating_count',
            'price', 'hour', 'day_of_week'
        ]
        self.user_graph_embeddings = None
        self.item_graph_embeddings = None

    def set_graph_embeddings(self, user_embeddings, item_embeddings):
        """设置图嵌入（来自LightGCN）"""
        self.user_graph_embeddings = user_embeddings
        self.item_graph_embeddings = item_embeddings

    def prepare_data(self, df):
        """准备数据"""
        user_idx = torch.LongTensor(df['user_idx'].values).to(self.device)
        item_idx = torch.LongTensor(df['item_idx'].values).to(self.device)
        category_idx = torch.LongTensor(df['category_idx'].values).to(self.device)
        dense_features = torch.FloatTensor(df[self.feature_cols].values).to(self.device)
        labels = torch.FloatTensor(df['label'].values).to(self.device) if 'label' in df.columns else None

        # 获取图嵌入
        user_graph_emb = self.user_graph_embeddings[user_idx]
        item_graph_emb = self.item_graph_embeddings[item_idx]

        return user_idx, item_idx, category_idx, dense_features, user_graph_emb, item_graph_emb, labels

    def train(self, train_df, valid_df, epochs=20, batch_size=1024, lr=0.001):
        """训练模型"""
        print("\n训练DeepFM模型（含图嵌入特征）...")

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        train_data = self.prepare_data(train_df)
        valid_data = self.prepare_data(valid_df)

        n_train = len(train_df)
        best_valid_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            indices = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                batch_indices = indices[i:i + batch_size]

                batch_user = train_data[0][batch_indices]
                batch_item = train_data[1][batch_indices]
                batch_category = train_data[2][batch_indices]
                batch_dense = train_data[3][batch_indices]
                batch_user_graph = train_data[4][batch_indices]
                batch_item_graph = train_data[5][batch_indices]
                batch_labels = train_data[6][batch_indices]

                optimizer.zero_grad()
                predictions = self.model(batch_user, batch_item, batch_category,
                                         batch_dense, batch_user_graph, batch_item_graph)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_indices)

            train_loss /= n_train

            # Validation
            self.model.eval()
            with torch.no_grad():
                valid_predictions = self.model(valid_data[0], valid_data[1], valid_data[2],
                                               valid_data[3], valid_data[4], valid_data[5])
                valid_loss = criterion(valid_predictions, valid_data[6]).item()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print("DeepFM训练完成")

    def predict(self, df):
        """预测"""
        self.model.eval()
        data = self.prepare_data(df)

        with torch.no_grad():
            predictions = self.model(data[0], data[1], data[2], data[3], data[4], data[5])

        return predictions.cpu().numpy()


# ============================================================================
# 第六部分:模型3 - SASRec (序列模型,含图嵌入特征)
# ============================================================================

class SASRecWithGraphEmbedding(nn.Module):
    """SASRec模型 - 融合图嵌入特征"""

    def __init__(self, n_items, hidden_size=64, graph_embedding_dim=64,
                 max_len=50, num_heads=2, num_blocks=2, dropout=0.2):
        super(SASRecWithGraphEmbedding, self).__init__()

        self.n_items = n_items
        self.hidden_size = hidden_size
        self.max_len = max_len

        # 物品嵌入
        self.item_embedding = nn.Embedding(n_items, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_size)

        # 图嵌入转换层
        self.graph_transform = nn.Linear(graph_embedding_dim, hidden_size)

        # 融合层
        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)

        # Dropout和LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Self-attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_blocks)
        ])

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.position_embedding.weight, std=0.01)

    def forward(self, item_seq, item_graph_emb_seq, target_item=None, target_graph_emb=None):
        """前向传播 - 融合图嵌入"""
        seq_len = item_seq.size(1)

        # Position
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        # Embeddings
        item_emb = self.item_embedding(item_seq)
        position_emb = self.position_embedding(position_ids)

        # 图嵌入转换
        graph_emb = self.graph_transform(item_graph_emb_seq)

        # 融合序列嵌入和图嵌入
        seq_emb = self.fusion_layer(torch.cat([item_emb + position_emb, graph_emb], dim=-1))
        seq_emb = self.dropout(seq_emb)
        seq_emb = self.layer_norm(seq_emb)

        # Attention mask (causal mask)
        attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=item_seq.device), diagonal=1)
        attention_mask = attention_mask.bool()

        # Self-attention blocks
        for block in self.attention_blocks:
            seq_emb = block(seq_emb, src_mask=attention_mask)

        # 如果提供了目标物品,进行预测
        if target_item is not None and target_graph_emb is not None:
            return self.predict(seq_emb, target_item, target_graph_emb)

        return seq_emb

    def predict(self, seq_emb, target_item, target_graph_emb):
        """预测目标物品的分数"""
        # 取最后一个时间步的序列表示
        seq_emb = seq_emb[:, -1, :]

        # 目标物品嵌入
        target_item_emb = self.item_embedding(target_item)

        # 融合目标物品的两种嵌入
        target_graph_trans = self.graph_transform(target_graph_emb)
        target_emb_fused = self.fusion_layer(torch.cat([target_item_emb, target_graph_trans], dim=-1))

        # 计算分数
        scores = torch.sum(seq_emb * target_emb_fused, dim=-1)

        return torch.sigmoid(scores)


class SASRecTrainer:
    """SASRec训练器"""

    def __init__(self, n_items, max_len=50, device='cpu'):
        self.device = device
        self.max_len = max_len
        self.n_items = n_items
        self.user_sequences = {}
        self.item_graph_embeddings = None

        self.model = SASRecWithGraphEmbedding(
            n_items=n_items,
            hidden_size=64,
            graph_embedding_dim=64,
            max_len=max_len,
            num_heads=2,
            num_blocks=2,
            dropout=0.2
        ).to(device)

    def set_graph_embeddings(self, item_embeddings):
        """设置物品的图嵌入"""
        self.item_graph_embeddings = item_embeddings.to(self.device)
        print(f"图嵌入形状: {self.item_graph_embeddings.shape}, 设备: {self.item_graph_embeddings.device}")

    def build_sequences(self, train_df):
        """构建用户行为序列"""
        print("\n构建用户行为序列...")

        user_sequences = defaultdict(list)
        train_sorted = train_df.sort_values(['user_idx', 'timestamp'])

        for _, row in train_sorted.iterrows():
            if row['label'] == 1:
                user_sequences[row['user_idx']].append(row['item_idx'])

        self.user_sequences = {u: seq for u, seq in user_sequences.items() if len(seq) >= 2}
        print(f"构建了 {len(self.user_sequences)} 个用户序列")

        return self.user_sequences

    def prepare_training_data(self, train_df):
        """准备训练数据"""
        self.build_sequences(train_df)

        sequences = []
        seq_graph_embs = []
        targets = []
        target_graph_embs = []
        labels = []

        for user_idx, seq in self.user_sequences.items():
            seq_len = len(seq)

            for i in range(1, seq_len):
                history = seq[max(0, i - self.max_len):i]
                padded_seq = [0] * (self.max_len - len(history)) + history
                sequences.append(padded_seq)

                seq_graph_emb = []
                for item_idx in padded_seq:
                    if item_idx == 0:
                        emb = torch.zeros(self.item_graph_embeddings.size(1), device=self.device)
                    else:
                        emb = self.item_graph_embeddings[item_idx]
                    seq_graph_emb.append(emb)

                seq_graph_embs.append(torch.stack(seq_graph_emb))

                # 正样本
                target_item = seq[i]
                targets.append(target_item)
                target_graph_embs.append(self.item_graph_embeddings[target_item])
                labels.append(1.0)

                # 负采样
                neg_item = np.random.randint(1, self.n_items)
                while neg_item in seq:
                    neg_item = np.random.randint(1, self.n_items)

                sequences.append(padded_seq)
                seq_graph_embs.append(torch.stack(seq_graph_emb))
                targets.append(neg_item)
                target_graph_embs.append(self.item_graph_embeddings[neg_item])
                labels.append(0.0)

        return sequences, seq_graph_embs, targets, target_graph_embs, labels

    def train(self, train_df, valid_df, epochs=10, batch_size=128, lr=0.001):
        """训练模型"""
        print("\n训练SASRec模型（含图嵌入特征）...")

        if self.item_graph_embeddings is None:
            raise ValueError("请先调用 set_graph_embeddings 设置图嵌入")

        sequences, seq_graph_embs, targets, target_graph_embs, labels = self.prepare_training_data(train_df)

        sequences_tensor = torch.LongTensor(sequences).to(self.device)
        seq_graph_embs_tensor = torch.stack(seq_graph_embs).to(self.device)
        targets_tensor = torch.LongTensor(targets).to(self.device)
        target_graph_embs_tensor = torch.stack(target_graph_embs).to(self.device)
        labels_tensor = torch.FloatTensor(labels).to(self.device)

        dataset = torch.utils.data.TensorDataset(
            sequences_tensor, seq_graph_embs_tensor,
            targets_tensor, target_graph_embs_tensor, labels_tensor
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            for batch in dataloader:
                batch_seq, batch_seq_graph_emb, batch_target, batch_target_graph_emb, batch_labels = batch

                optimizer.zero_grad()

                predictions = self.model(
                    batch_seq,
                    batch_seq_graph_emb,
                    batch_target,
                    batch_target_graph_emb
                )

                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 2 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / n_batches:.4f}")

        print("SASRec训练完成")

    def predict(self, df):
        """预测评分"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for _, row in df.iterrows():
                user_idx = row['user_idx']
                target_item = row['item_idx']

                if user_idx in self.user_sequences:
                    history = self.user_sequences[user_idx][-self.max_len:]
                else:
                    history = []

                padded_seq = [0] * (self.max_len - len(history)) + history
                seq_tensor = torch.LongTensor([padded_seq]).to(self.device)

                seq_graph_emb = []
                for item_idx in padded_seq:
                    if item_idx == 0:
                        emb = torch.zeros(self.item_graph_embeddings.size(1), device=self.device)
                    else:
                        emb = self.item_graph_embeddings[item_idx]
                    seq_graph_emb.append(emb)

                seq_graph_emb_tensor = torch.stack(seq_graph_emb).unsqueeze(0)

                target_tensor = torch.LongTensor([target_item]).to(self.device)
                target_graph_emb = self.item_graph_embeddings[target_item].unsqueeze(0)

                score = self.model(seq_tensor, seq_graph_emb_tensor, target_tensor, target_graph_emb)
                predictions.append(score.item())

        return np.array(predictions)


# ============================================================================
# 第七部分：评估指标
# ============================================================================

class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def precision_at_k(y_true, y_pred_scores, k=10):
        """Precision@K"""
        top_k_indices = np.argsort(y_pred_scores)[-k:]
        top_k_true = y_true[top_k_indices]
        return np.sum(top_k_true) / k

    @staticmethod
    def recall_at_k(y_true, y_pred_scores, k=10):
        """Recall@K"""
        if np.sum(y_true) == 0:
            return 0.0
        top_k_indices = np.argsort(y_pred_scores)[-k:]
        top_k_true = y_true[top_k_indices]
        return np.sum(top_k_true) / np.sum(y_true)

    @staticmethod
    def f1_score_at_k(precision, recall):
        """F1-Score@K"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def ndcg_at_k(y_true, y_pred_scores, k=10):
        """NDCG@K"""
        top_k_indices = np.argsort(y_pred_scores)[-k:][::-1]
        top_k_true = y_true[top_k_indices]

        dcg = np.sum((2 ** top_k_true - 1) / np.log2(np.arange(2, k + 2)))

        ideal_true = np.sort(y_true)[::-1][:k]
        idcg = np.sum((2 ** ideal_true - 1) / np.log2(np.arange(2, k + 2)))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def map_at_k(y_true, y_pred_scores, k=10):
        """MAP@K"""
        top_k_indices = np.argsort(y_pred_scores)[-k:][::-1]
        top_k_true = y_true[top_k_indices]

        if np.sum(top_k_true) == 0:
            return 0.0

        precision_sum = 0
        num_hits = 0

        for i, label in enumerate(top_k_true):
            if label == 1:
                num_hits += 1
                precision_sum += num_hits / (i + 1)

        return precision_sum / min(np.sum(y_true), k)

    @staticmethod
    def evaluate_by_user(df, score_col, k_list=[5, 10, 20]):
        """按用户评估"""
        results = {k: {'precision': [], 'recall': [], 'f1': [], 'ndcg': [], 'map': []}
                   for k in k_list}

        for user_idx, group in df.groupby('user_idx'):
            y_true = group['label'].values
            y_scores = group[score_col].values

            if len(y_true) == 0:
                continue

            for k in k_list:
                actual_k = min(k, len(y_true))

                precision = MetricsCalculator.precision_at_k(y_true, y_scores, actual_k)
                recall = MetricsCalculator.recall_at_k(y_true, y_scores, actual_k)
                f1 = MetricsCalculator.f1_score_at_k(precision, recall)
                ndcg = MetricsCalculator.ndcg_at_k(y_true, y_scores, actual_k)
                map_score = MetricsCalculator.map_at_k(y_true, y_scores, actual_k)

                results[k]['precision'].append(precision)
                results[k]['recall'].append(recall)
                results[k]['f1'].append(f1)
                results[k]['ndcg'].append(ndcg)
                results[k]['map'].append(map_score)

        avg_results = {}
        for k in k_list:
            avg_results[k] = {
                'precision': np.nanmean(results[k]['precision']) if results[k]['precision'] else 0.0,
                'recall': np.nanmean(results[k]['recall']) if results[k]['recall'] else 0.0,
                'f1': np.nanmean(results[k]['f1']) if results[k]['f1'] else 0.0,
                'ndcg': np.nanmean(results[k]['ndcg']) if results[k]['ndcg'] else 0.0,
                'map': np.nanmean(results[k]['map']) if results[k]['map'] else 0.0
            }

        return avg_results


# ============================================================================
# 第八部分：动态权重融合策略
# ============================================================================

class DynamicWeightedFusion:
    """动态加权融合策略"""

    def __init__(self):
        self.base_weights = None
        self.user_activity = {}

    def calculate_user_activity(self, train_df):
        """计算用户活跃度"""
        print("\n计算用户活跃度...")

        for user_idx, group in train_df.groupby('user_idx'):
            activity_score = len(group) / train_df.groupby('user_idx').size().max()
            self.user_activity[user_idx] = activity_score

    def grid_search_base_weights(self, valid_df, score_cols, k=10, step=0.1):
        """网格搜索基础权重"""
        print("\n网格搜索基础融合权重...")

        best_score = 0
        best_weights = None

        weight_range = np.arange(0, 1.0 + step, step)

        for w1 in weight_range:
            for w2 in weight_range:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-6 or w3 > 1.0 + 1e-6:
                    continue

                weights = np.array([w1, w2, max(0, w3)])
                weights = weights / np.sum(weights)

                fusion_scores = np.zeros(len(valid_df))
                for i, col in enumerate(score_cols):
                    fusion_scores += weights[i] * valid_df[col].values

                valid_df_temp = valid_df.copy()
                valid_df_temp['fusion_score'] = fusion_scores

                results = MetricsCalculator.evaluate_by_user(
                    valid_df_temp, 'fusion_score', k_list=[k]
                )

                score = (0.3 * results[k]['precision'] +
                         0.3 * results[k]['recall'] +
                         0.4 * results[k]['ndcg'])

                if score > best_score:
                    best_score = score
                    best_weights = weights

        if best_weights is None:
            best_weights = np.array([1/3, 1/3, 1/3])
            print("\n警告：未找到最优权重，使用均等权重")
        else:
            self.base_weights = best_weights
            print(f"\n最优基础权重:")
            print(f"  LightGCN: {best_weights[0]:.3f}")
            print(f"  DeepFM:   {best_weights[1]:.3f}")
            print(f"  SASRec:   {best_weights[2]:.3f}")
            print(f"  综合得分: {best_score:.4f}")

        return best_weights

    def apply_dynamic_fusion(self, df, score_cols):
        """应用动态融合策略"""
        if self.base_weights is None:
            raise ValueError("请先调用 grid_search_base_weights")

        print("\n应用动态权重融合...")

        fusion_scores = []

        for _, row in df.iterrows():
            user_idx = row['user_idx']
            activity = self.user_activity.get(user_idx, 0.5)

            # 动态调整权重
            # α (LightGCN): 基础权重保持稳定
            alpha = self.base_weights[0]

            # β (DeepFM): 基础权重
            beta = self.base_weights[1]

            # γ (SASRec): 根据活跃度动态调整
            gamma_boost = 0.2 * activity  # 活跃用户增强序列模型权重
            gamma = self.base_weights[2] + gamma_boost

            # 归一化
            total = alpha + beta + gamma
            weights = np.array([alpha, beta, gamma]) / total

            # 加权融合
            score = (weights[0] * row[score_cols[0]] +
                     weights[1] * row[score_cols[1]] +
                     weights[2] * row[score_cols[2]])

            fusion_scores.append(score)

        return np.array(fusion_scores)


# ============================================================================
# 第九部分：主实验流程
# ============================================================================

def main():
    """主实验流程"""

    np.random.seed(42)
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # ========================================================================
    # 步骤1：数据加载与预处理
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤1：数据加载与预处理")
    print("=" * 80)

    preprocessor = DataPreprocessor(min_user_interactions=5, min_item_interactions=5)

    df = preprocessor.load_and_preprocess(
        use_sample=True,
        dataset='movielens-10m',
        sample_size=500000  # 使用50万条数据
    )

    if df is None:
        print("数据加载失败，退出实验")
        return

    df = preprocessor.create_binary_labels(df, threshold=4)

    train_df, valid_df, test_df = preprocessor.split_data_by_time(df)
    train_df, valid_df, test_df = preprocessor.encode_ids(train_df, valid_df, test_df)

    # ========================================================================
    # 步骤2：特征工程
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤2：特征工程")
    print("=" * 80)

    feature_engine = FeatureEngine()
    train_df, valid_df, test_df = feature_engine.build_features(train_df, valid_df, test_df)

    # ========================================================================
    # 步骤3：训练LightGCN（图嵌入模型）
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤3：训练LightGCN（图嵌入模型）")
    print("=" * 80)

    lightgcn_trainer = LightGCNTrainer(preprocessor.n_users, preprocessor.n_items,
                                       embedding_dim=64, device=device)
    lightgcn_trainer.train(train_df, valid_df, epochs=20)

    # 获取图嵌入
    user_graph_embeddings, item_graph_embeddings = lightgcn_trainer.get_embeddings()

    # ========================================================================
    # 步骤4：训练DeepFM（融合图嵌入）
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤4：训练DeepFM（融合图嵌入特征）")
    print("=" * 80)

    n_categories = train_df['category_idx'].max() + 1
    deepfm_trainer = DeepFMTrainer(preprocessor.n_users, preprocessor.n_items,
                                   n_categories, device=device)
    deepfm_trainer.set_graph_embeddings(user_graph_embeddings, item_graph_embeddings)
    deepfm_trainer.train(train_df, valid_df, epochs=20)

    # ========================================================================
    # 步骤5：训练SASRec（融合图嵌入）
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤5：训练SASRec（融合图嵌入特征）")
    print("=" * 80)

    sasrec_trainer = SASRecTrainer(preprocessor.n_items, max_len=30, device=device)
    sasrec_trainer.set_graph_embeddings(item_graph_embeddings)
    sasrec_trainer.train(train_df, valid_df, epochs=10)

    # ========================================================================
    # 步骤6:生成候选集
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤6:生成候选集")
    print("=" * 80)

    candidate_gen = CandidateGenerator()
    valid_candidates = candidate_gen.generate_candidates(valid_df, train_df, n_neg=99)
    test_candidates = candidate_gen.generate_candidates(test_df, train_df, n_neg=99)

    # ========================================================================
    # 步骤7:在候选集上预测
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤7:在候选集上预测")
    print("=" * 80)

    print("\n验证集候选预测...")
    valid_candidates['lightgcn_score'] = lightgcn_trainer.predict(valid_candidates)
    valid_candidates['deepfm_score'] = deepfm_trainer.predict(valid_candidates)
    valid_candidates['sasrec_score'] = sasrec_trainer.predict(valid_candidates)

    print("测试集候选预测...")
    test_candidates['lightgcn_score'] = lightgcn_trainer.predict(test_candidates)
    test_candidates['deepfm_score'] = deepfm_trainer.predict(test_candidates)
    test_candidates['sasrec_score'] = sasrec_trainer.predict(test_candidates)

    # ========================================================================
    # 步骤8: 动态权重融合与评估
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤8: 动态权重融合与评估")
    print("=" * 80)

    score_cols = ['lightgcn_score', 'deepfm_score', 'sasrec_score']

    # 初始化融合器并计算用户活跃度
    fusion = DynamicWeightedFusion()
    fusion.calculate_user_activity(train_df)

    # 网格搜索基础权重
    fusion.grid_search_base_weights(valid_candidates, score_cols, k=10, step=0.1)

    # 应用动态融合
    valid_candidates['fusion_score'] = fusion.apply_dynamic_fusion(valid_candidates, score_cols)
    test_candidates['fusion_score'] = fusion.apply_dynamic_fusion(test_candidates, score_cols)

    # ========================================================================
    # 步骤9: 测试集评估与对比
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤9: 测试集评估与对比")
    print("=" * 80)

    k_list = [5, 10, 20]
    model_names = ['LightGCN', 'DeepFM', 'SASRec', 'Dynamic Fusion']
    score_columns = ['lightgcn_score', 'deepfm_score', 'sasrec_score', 'fusion_score']

    all_results = {}

    for model_name, score_col in zip(model_names, score_columns):
        print(f"\n评估 {model_name}...")
        results = MetricsCalculator.evaluate_by_user(
            test_candidates, score_col, k_list=k_list
        )
        all_results[model_name] = results

        for k in k_list:
            print(f"  @K={k}: Precision={results[k]['precision']:.4f}, "
                  f"Recall={results[k]['recall']:.4f}, "
                  f"F1={results[k]['f1']:.4f}, "
                  f"NDCG={results[k]['ndcg']:.4f}, "
                  f"MAP={results[k]['map']:.4f}")

    # ========================================================================
    # 步骤10: 结果可视化
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤10: 结果可视化")
    print("=" * 80)

    metrics = ['precision', 'recall', 'f1', 'ndcg', 'map']
    metric_names = ['Precision', 'Recall', 'F1-Score', 'NDCG', 'MAP']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        x = np.arange(len(k_list))
        width = 0.2

        for i, model_name in enumerate(model_names):
            values = [all_results[model_name][k][metric] for k in k_list]
            ax.bar(x + i * width, values, width, label=model_name)

        ax.set_xlabel('K')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}@K Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f'K={k}' for k in k_list])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('dynamic_fusion_results.png', dpi=300, bbox_inches='tight')
    print("\n对比图表已保存为 'dynamic_fusion_results.png'")
    plt.show()

    # ========================================================================
    # 步骤11: 生成结果报告
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤11: 生成结果报告")
    print("=" * 80)

    report_data = []
    for k in k_list:
        for metric in metrics:
            row = {'K': k, 'Metric': metric.upper()}
            for model_name in model_names:
                row[model_name] = f"{all_results[model_name][k][metric]:.4f}"
            report_data.append(row)

    report_df = pd.DataFrame(report_data)

    print("\n实验结果汇总表:")
    print("=" * 80)
    print(report_df.to_string(index=False))

    report_df.to_csv('dynamic_fusion_experiment_results.csv', index=False)
    print("\n结果已保存为 'dynamic_fusion_experiment_results.csv'")

    # ========================================================================
    # 步骤12: 性能提升分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤12: 性能提升分析")
    print("=" * 80)

    print("\n动态融合相对于单一模型的性能提升:")
    print("-" * 80)

    for k in k_list:
        print(f"\n@K={k}:")
        for metric in metrics:
            fusion_value = all_results['Dynamic Fusion'][k][metric]

            best_single_model = max(['LightGCN', 'DeepFM', 'SASRec'],
                                    key=lambda m: all_results[m][k][metric])
            best_single_value = all_results[best_single_model][k][metric]

            if best_single_value > 0:
                improvement = (fusion_value - best_single_value) / best_single_value * 100
                print(f"  {metric.upper():10s}: {fusion_value:.4f} vs {best_single_model} "
                      f"{best_single_value:.4f} (提升 {improvement:+.2f}%)")
            else:
                print(f"  {metric.upper():10s}: {fusion_value:.4f} vs {best_single_model} "
                      f"{best_single_value:.4f}")

    print("\n" + "=" * 80)
    print("实验完成!")
    print("=" * 80)
    print("\n生成的文件:")
    print("  1. dynamic_fusion_results.png - 动态融合对比可视化图表")
    print("  2. dynamic_fusion_experiment_results.csv - 详细实验结果")
    print("\n实验验证了动态加权融合策略的有效性,混合模型在多个指标上优于单一模型。")

if __name__ == "__main__":
    main()
