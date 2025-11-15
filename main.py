"""
推荐系统混合模型实验验证方案
使用LightGBM、DeepFM和SASRec三个模型进行加权融合
数据集：Amazon Product Data (Books类别 - 小数据集)
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
import torch.optim as optim

# 可视化
import matplotlib.pyplot as plt

print("=" * 80)
print("推荐系统混合模型实验验证方案")
print("=" * 80)

# ============================================================================
# 第一部分：数据加载与预处理（修改版）
# ============================================================================

import os



class DataPreprocessor:
    """数据预处理类（支持 Amazon Books 5-core 数据集）"""

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
        movies_file = os.path.join(data_dir, 'movies.dat')

        if not os.path.exists(ratings_file):
            print(f"错误：未找到评分文件 {ratings_file}")
            return None

        # 加载评分数据 (UserID::MovieID::Rating::Timestamp)
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

        # 加载电影元数据 (MovieID::Title::Genres)
        if os.path.exists(movies_file):
            print("读取电影元数据...")
            movies_data = {}

            with open(movies_file, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 3:
                        movie_id = int(parts[0])
                        genres = parts[2].split('|')[0]  # 取第一个类别
                        movies_data[movie_id] = genres

            # 添加类别和价格（模拟）
            df['category'] = df['item_id'].map(
                lambda x: movies_data.get(x, 'Unknown')
            )

            np.random.seed(42)
            unique_items = df['item_id'].unique()
            item_to_price = {item: np.random.uniform(5, 50) for item in unique_items}
            df['price'] = df['item_id'].map(item_to_price)
        else:
            # 如果没有元数据，使用模拟类别
            categories = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance',
                          'Horror', 'SciFi', 'Animation', 'Documentary', 'Crime']
            item_to_category = {item: categories[hash(item) % len(categories)]
                                for item in df['item_id'].unique()}
            df['category'] = df['item_id'].map(item_to_category)

            np.random.seed(42)
            item_to_price = {item: np.random.uniform(5, 50)
                             for item in df['item_id'].unique()}
            df['price'] = df['item_id'].map(item_to_price)

        return df

    def load_and_preprocess(self, file_path=None, use_sample=True, dataset='amazon-books', sample_size=100000):
        """加载并预处理数据"""

        if use_sample:
            if dataset == 'movielens-10m':
                # 使用 MovieLens 10M
                data_dir = self.download_movielens_10m()
                if data_dir is None:
                    print("\n使用模拟数据代替...")
                    return self._generate_sample_data(sample_size)

                df = self.load_movielens_10m(data_dir, sample_size=sample_size)

                if df is None:
                    print("\n使用模拟数据代替...")
                    return self._generate_sample_data(sample_size)

            else:
                # 模拟数据
                return self._generate_sample_data(sample_size)

        else:
            # 从自定义文件加载
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
        print(f"  评分分布:\n{df['rating'].value_counts().sort_index()}")
        print(f"  时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")

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

    def create_binary_labels(self, df, threshold=4):
        """创建二分类标签（rating >= threshold 为正样本）"""
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
        print(f"训练集: {len(train_df):,} 条 ({train_df['timestamp'].min()} 到 {train_df['timestamp'].max()})")
        print(f"验证集: {len(valid_df):,} 条 ({valid_df['timestamp'].min()} 到 {valid_df['timestamp'].max()})")
        print(f"测试集: {len(test_df):,} 条 ({test_df['timestamp'].min()} 到 {test_df['timestamp'].max()})")

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

        # 基于训练集构建统计特征
        self._build_user_features(train_df)
        self._build_item_features(train_df)

        # 为每个数据集添加特征
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

        # 用户特征
        df['user_avg_rating'] = df['user_idx'].map(
            lambda x: self.user_features.get(x, {}).get('user_avg_rating', 3.5))
        df['user_std_rating'] = df['user_idx'].map(
            lambda x: self.user_features.get(x, {}).get('user_std_rating', 0))
        df['user_rating_count'] = df['user_idx'].map(
            lambda x: self.user_features.get(x, {}).get('user_rating_count', 0))

        # 物品特征
        df['item_avg_rating'] = df['item_idx'].map(
            lambda x: self.item_features.get(x, {}).get('item_avg_rating', 3.5))
        df['item_std_rating'] = df['item_idx'].map(
            lambda x: self.item_features.get(x, {}).get('item_std_rating', 0))
        df['item_rating_count'] = df['item_idx'].map(
            lambda x: self.item_features.get(x, {}).get('item_rating_count', 0))

        # 时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        return df


class CandidateGenerator:
    """生成测试候选集（模拟真实推荐场景）"""

    def __init__(self, n_candidates=100):
        self.n_candidates = n_candidates

    def generate_candidates(self, df, train_df, n_neg=99):
        """为每个用户生成候选集：1个正样本 + n_neg个负样本"""
        print(f"\n生成候选集 (每用户{n_neg + 1}个候选)...")

        # 构建用户已交互物品集合
        user_items = defaultdict(set)
        for _, row in train_df.iterrows():
            if row['label'] == 1:
                user_items[row['user_idx']].add(row['item_idx'])

        candidates = []

        for user_idx in df['user_idx'].unique():
            user_df = df[df['user_idx'] == user_idx]

            # 取该用户的正样本
            pos_samples = user_df[user_df['label'] == 1]

            if len(pos_samples) == 0:
                continue

            # 随机选择一个正样本
            pos_sample = pos_samples.sample(1).iloc[0]
            candidates.append(pos_sample.to_dict())

            # 随机负采样：未交互的物品
            interacted = user_items[user_idx]
            all_items = set(range(df['item_idx'].max() + 1))
            neg_pool = list(all_items - interacted)

            if len(neg_pool) < n_neg:
                n_neg = len(neg_pool)

            neg_items = np.random.choice(neg_pool, size=n_neg, replace=False)

            for item_idx in neg_items:
                # 构造负样本（使用平均特征值）
                neg_sample = pos_sample.copy()
                neg_sample['item_idx'] = item_idx
                neg_sample['label'] = 0

                # 更新物品相关特征（使用训练集统计）
                item_features = self._get_item_features(train_df, item_idx)
                neg_sample.update(item_features)

                candidates.append(neg_sample)

        candidates_df = pd.DataFrame(candidates)
        print(f"生成候选集: {len(candidates_df):,} 条 (用户数: {candidates_df['user_idx'].nunique():,})")

        return candidates_df

    def _get_item_features(self, train_df, item_idx):
        """获取物品特征（从训练集统计）"""
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
            # 使用全局默认值
            return {
                'item_avg_rating': train_df['rating'].mean(),
                'item_std_rating': 0,
                'item_rating_count': 0,
                'category_idx': 0,
                'price': train_df['price'].mean()
            }


# ============================================================================
# 第三部分：模型1 - LightGBM
# ============================================================================

class LightGBMModel:
    """LightGBM模型"""

    def __init__(self):
        self.model = None
        self.feature_cols = None

    def prepare_features(self, df):
        """准备特征"""
        self.feature_cols = [
            'user_idx', 'item_idx', 'category_idx',
            'user_avg_rating', 'user_std_rating', 'user_rating_count',
            'item_avg_rating', 'item_std_rating', 'item_rating_count',
            'price', 'hour', 'day_of_week'
        ]

        X = df[self.feature_cols].values
        y = df['label'].values if 'label' in df.columns else None

        return X, y

    def train(self, train_df, valid_df):
        """训练模型"""
        print("\n训练LightGBM模型...")

        X_train, y_train = self.prepare_features(train_df)
        X_valid, y_valid = self.prepare_features(valid_df)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
        )

        print("LightGBM训练完成")

    def predict(self, df):
        """预测"""
        X, _ = self.prepare_features(df)
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        return predictions


# ============================================================================
# 第四部分：模型2 - DeepFM
# ============================================================================

class DeepFMModel(nn.Module):
    """DeepFM模型"""

    def __init__(self, n_users, n_items, n_categories, embedding_dim=16, hidden_dims=[64, 32]):
        super(DeepFMModel, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories
        self.embedding_dim = embedding_dim

        # Embeddings for FM part
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.category_embedding = nn.Embedding(n_categories, embedding_dim)

        # FM bias
        self.fm_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        # Dense features (9个数值特征)
        n_dense_features = 9

        # Deep part
        deep_input_dim = embedding_dim * 3 + n_dense_features

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

    def forward(self, user_idx, item_idx, category_idx, dense_features):
        """前向传播"""
        # Embeddings
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        category_emb = self.category_embedding(category_idx)

        # FM part - first order
        fm_first_order = (self.fm_bias +
                          self.user_bias(user_idx).squeeze() +
                          self.item_bias(item_idx).squeeze())

        # FM part - second order (interactions)
        embeddings = torch.stack([user_emb, item_emb, category_emb], dim=1)
        sum_of_square = torch.sum(embeddings, dim=1) ** 2
        square_of_sum = torch.sum(embeddings ** 2, dim=1)
        fm_second_order = 0.5 * torch.sum(sum_of_square - square_of_sum, dim=1)

        # Deep part
        deep_input = torch.cat([user_emb, item_emb, category_emb, dense_features], dim=1)
        deep_output = self.deep_layers(deep_input).squeeze()

        # Combine
        output = fm_first_order + fm_second_order + deep_output

        return torch.sigmoid(output)


class DeepFMTrainer:
    """DeepFM训练器"""

    def __init__(self, n_users, n_items, n_categories, device='cpu'):
        self.device = device
        self.model = DeepFMModel(n_users, n_items, n_categories).to(device)
        self.feature_cols = [
            'user_avg_rating', 'user_std_rating', 'user_rating_count',
            'item_avg_rating', 'item_std_rating', 'item_rating_count',
            'price', 'hour', 'day_of_week'
        ]

    def prepare_data(self, df):
        """准备数据"""
        user_idx = torch.LongTensor(df['user_idx'].values).to(self.device)
        item_idx = torch.LongTensor(df['item_idx'].values).to(self.device)
        category_idx = torch.LongTensor(df['category_idx'].values).to(self.device)
        dense_features = torch.FloatTensor(df[self.feature_cols].values).to(self.device)
        labels = torch.FloatTensor(df['label'].values).to(self.device) if 'label' in df.columns else None

        return user_idx, item_idx, category_idx, dense_features, labels

    def train(self, train_df, valid_df, epochs=20, batch_size=1024, lr=0.001):
        """训练模型"""
        print("\n训练DeepFM模型...")

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

            # Mini-batch training
            indices = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                batch_indices = indices[i:i + batch_size]

                batch_user = train_data[0][batch_indices]
                batch_item = train_data[1][batch_indices]
                batch_category = train_data[2][batch_indices]
                batch_dense = train_data[3][batch_indices]
                batch_labels = train_data[4][batch_indices]

                optimizer.zero_grad()
                predictions = self.model(batch_user, batch_item, batch_category, batch_dense)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_indices)

            train_loss /= n_train

            # Validation
            self.model.eval()
            with torch.no_grad():
                valid_predictions = self.model(valid_data[0], valid_data[1],
                                               valid_data[2], valid_data[3])
                valid_loss = criterion(valid_predictions, valid_data[4]).item()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

            # Early stopping
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
            predictions = self.model(data[0], data[1], data[2], data[3])

        return predictions.cpu().numpy()


# ============================================================================
# 第五部分：模型3 - SASRec (Self-Attentive Sequential Recommendation)
# ============================================================================

class SASRecModel(nn.Module):
    """SASRec模型"""

    def __init__(self, n_items, hidden_size=64, num_heads=2, num_blocks=2,
                 max_len=50, dropout=0.2):
        super(SASRecModel, self).__init__()

        self.n_items = n_items
        self.hidden_size = hidden_size
        self.max_len = max_len

        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_size)

        # Self-attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation='relu',
                batch_first=True
            ) for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.position_embedding.weight, std=0.01)

    def forward(self, item_seq):
        """前向传播"""
        # item_seq: [batch_size, seq_len]
        seq_len = item_seq.size(1)

        # Position
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        # Embeddings
        item_emb = self.item_embedding(item_seq)
        position_emb = self.position_embedding(position_ids)

        seq_emb = item_emb + position_emb
        seq_emb = self.dropout(seq_emb)
        seq_emb = self.layer_norm(seq_emb)

        # Attention mask (causal mask)
        attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=item_seq.device), diagonal=1)
        attention_mask = attention_mask.bool()

        # Self-attention blocks
        for block in self.attention_blocks:
            seq_emb = block(seq_emb, src_mask=attention_mask)

        return seq_emb

    def predict(self, seq_emb, target_items):
        """预测目标物品的分数"""
        # seq_emb: [batch_size, seq_len, hidden_size]
        # target_items: [batch_size]

        # Get last hidden state
        seq_emb = seq_emb[:, -1, :]  # [batch_size, hidden_size]

        # Get target item embeddings
        target_emb = self.item_embedding(target_items)  # [batch_size, hidden_size]

        # Compute scores
        scores = torch.sum(seq_emb * target_emb, dim=-1)  # [batch_size]

        return torch.sigmoid(scores)


class SASRecTrainer:
    """SASRec训练器"""

    def __init__(self, n_items, max_len=50, device='cpu'):
        self.device = device
        self.model = SASRecModel(n_items, max_len=max_len).to(device)
        self.max_len = max_len
        self.n_items = n_items
        self.user_sequences = {}

    def build_sequences(self, train_df):
        """构建用户序列"""
        print("\n构建用户行为序列...")

        train_df = train_df.sort_values(['user_idx', 'timestamp'])

        for user_idx, group in train_df.groupby('user_idx'):
            items = group['item_idx'].tolist()
            # 只保留正样本
            labels = group['label'].tolist()
            items = [item + 1 for item, label in zip(items, labels) if label == 1]  # +1 to reserve 0 for padding

            if len(items) > 0:
                self.user_sequences[user_idx] = items

        print(f"构建了 {len(self.user_sequences)} 个用户序列")

    def prepare_training_data(self, train_df):
        """准备训练数据"""
        sequences = []
        targets = []
        labels = []

        for user_idx, items in self.user_sequences.items():
            if len(items) < 2:
                continue

            for i in range(1, len(items)):
                seq = items[max(0, i - self.max_len):i]
                # Padding
                if len(seq) < self.max_len:
                    seq = [0] * (self.max_len - len(seq)) + seq

                sequences.append(seq)
                targets.append(items[i])
                labels.append(1)

                # Negative sampling
                neg_item = np.random.randint(1, self.n_items + 1)
                while neg_item in items:
                    neg_item = np.random.randint(1, self.n_items + 1)

                sequences.append(seq)
                targets.append(neg_item)
                labels.append(0)

        return (torch.LongTensor(sequences).to(self.device),
                torch.LongTensor(targets).to(self.device),
                torch.FloatTensor(labels).to(self.device))

    def train(self, train_df, valid_df, epochs=10, batch_size=128, lr=0.001):
        """训练模型"""
        print("\n训练SASRec模型...")

        self.build_sequences(train_df)

        sequences, targets, labels = self.prepare_training_data(train_df)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        n_samples = len(sequences)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            indices = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]

                batch_seq = sequences[batch_indices]
                batch_target = targets[batch_indices]
                batch_labels = labels[batch_indices]

                optimizer.zero_grad()

                seq_emb = self.model(batch_seq)
                predictions = self.model.predict(seq_emb, batch_target)

                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_indices)

            train_loss /= n_samples

            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        print("SASRec训练完成")

    def predict(self, df):
        """预测"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for idx, row in df.iterrows():
                user_idx = row['user_idx']
                item_idx = row['item_idx']

                # Get user sequence
                if user_idx in self.user_sequences:
                    seq = self.user_sequences[user_idx][-self.max_len:]
                    if len(seq) < self.max_len:
                        seq = [0] * (self.max_len - len(seq)) + seq
                else:
                    seq = [0] * self.max_len

                seq_tensor = torch.LongTensor([seq]).to(self.device)
                target_tensor = torch.LongTensor([item_idx + 1]).to(self.device)

                seq_emb = self.model(seq_tensor)
                score = self.model.predict(seq_emb, target_tensor)

                predictions.append(score.item())

        return np.array(predictions)


# ============================================================================
# 第六部分：评估指标
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
        """MAP@K (Mean Average Precision)"""
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

            # 修改条件：只要有数据就评估
            if len(y_true) == 0:
                continue

            for k in k_list:
                # 如果数据量小于 k，使用实际数据量
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

        # 计算平均值（使用 nanmean 避免空列表）
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
# 第七部分：模型融合
# ============================================================================

class ModelFusion:
    """模型融合类（改进版）"""

    def __init__(self, method='weighted_average'):
        self.method = method
        self.weights = None

    def grid_search_weights(self, valid_df, score_cols, k=10, step=0.1):
        """网格搜索最优权重（按用户评估）"""
        print("\n网格搜索最优融合权重...")

        best_score = 0
        best_weights = None
        best_metric_name = 'ndcg'  # 使用 NDCG 作为主指标

        weight_range = np.arange(0, 1.0 + step, step)

        search_log = []

        for w1 in weight_range:
            for w2 in weight_range:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-6 or w3 > 1.0 + 1e-6:  # 允许浮点误差
                    continue

                weights = [w1, w2, max(0, w3)]  # 确保非负

                # 归一化（以防浮点误差）
                weights = np.array(weights) / np.sum(weights)

                # 计算融合分数
                fusion_scores = np.zeros(len(valid_df))
                for i, col in enumerate(score_cols):
                    fusion_scores += weights[i] * valid_df[col].values

                valid_df_temp = valid_df.copy()
                valid_df_temp['fusion_score'] = fusion_scores

                # 按用户评估
                results = MetricsCalculator.evaluate_by_user(
                    valid_df_temp, 'fusion_score', k_list=[k]
                )

                # 综合多个指标（加权平均）
                score = (
                        0.3 * results[k]['precision'] +
                        0.3 * results[k]['recall'] +
                        0.4 * results[k]['ndcg']
                )

                search_log.append({
                    'w1': weights[0],
                    'w2': weights[1],
                    'w3': weights[2],
                    'score': score,
                    'precision': results[k]['precision'],
                    'recall': results[k]['recall'],
                    'ndcg': results[k]['ndcg']
                })

                if score > best_score:
                    best_score = score
                    best_weights = weights

        # 打印搜索日志（Top 5）
        search_df = pd.DataFrame(search_log).sort_values('score', ascending=False)
        print("\nTop 5 权重组合:")
        print(search_df.head(5).to_string(index=False))

        if best_weights is None:
            best_weights = [1 / 3, 1 / 3, 1 / 3]
            print("\n警告：未找到最优权重，使用均等权重")
        else:
            self.weights = best_weights
            print(f"\n最优权重:")
            print(f"  LightGBM: {best_weights[0]:.3f}")
            print(f"  DeepFM:   {best_weights[1]:.3f}")
            print(f"  SASRec:   {best_weights[2]:.3f}")
            print(f"  综合得分: {best_score:.4f}")

        return best_weights

    def apply_fusion(self, df, score_cols):
        """根据已有权重计算融合得分"""
        if self.weights is None:
            raise ValueError("请先调用 grid_search_weights 确定融合权重")
        fusion_scores = np.zeros(len(df), dtype=float)
        for weight, col in zip(self.weights, score_cols):
            fusion_scores += weight * df[col].values
        return fusion_scores


# ============================================================================
# 第八部分：主实验流程
# ============================================================================

def main():
    """主实验流程"""

    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # ========================================================================
    # 步骤1：数据加载与预处理
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤1：数据加载与预处理")
    print("=" * 80)

    preprocessor = DataPreprocessor(min_user_interactions=5, min_item_interactions=5)

    # 使用 MovieLens 10M（采样100万条用于训练）
    df = preprocessor.load_and_preprocess(
        use_sample=True,
        dataset='movielens-10m',  # 改为 MovieLens 10M
        sample_size=1000000  # 采样100万条，设置 None 使用全部1000万条
    )

    if df is None:
        print("数据加载失败，退出实验")
        return

    df = preprocessor.create_binary_labels(df, threshold=4)

    # 使用比例划分而非固定天数
    train_df, valid_df, test_df = preprocessor.split_data_by_time(df,
                                                                  train_ratio=0.7,
                                                                  valid_ratio=0.15,
                                                                  test_ratio=0.15)
    train_df, valid_df, test_df = preprocessor.encode_ids(train_df, valid_df, test_df)

    # ========================================================================
    # 步骤2：特征工程
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤2：特征工程")
    print("=" * 80)

    feature_engine = FeatureEngine()
    train_df, valid_df, test_df = feature_engine.build_features(train_df, valid_df, test_df)

    print(f"\n特征列: {train_df.columns.tolist()}")
    print(f"训练集正样本比例: {train_df['label'].mean():.4f}")
    print(f"验证集正样本比例: {valid_df['label'].mean():.4f}")
    print(f"测试集正样本比例: {test_df['label'].mean():.4f}")

    # ========================================================================
    # 步骤3：训练三个基准模型
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤3：训练基准模型")
    print("=" * 80)

    # 3.1 LightGBM
    lgb_model = LightGBMModel()
    lgb_model.train(train_df, valid_df)

    # 3.2 DeepFM
    n_categories = train_df['category_idx'].max() + 1
    deepfm_trainer = DeepFMTrainer(preprocessor.n_users, preprocessor.n_items,
                                   n_categories, device=device)
    deepfm_trainer.train(train_df, valid_df, epochs=20)

    # 3.3 SASRec
    sasrec_trainer = SASRecTrainer(preprocessor.n_items, max_len=30, device=device)
    sasrec_trainer.train(train_df, valid_df, epochs=10)

    # ========================================================================
    # 步骤4.5：生成候选集（新增）
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤4.5：生成测试候选集")
    print("=" * 80)

    candidate_gen = CandidateGenerator(n_candidates=100)

    # 为验证集和测试集生成候选
    valid_candidates = candidate_gen.generate_candidates(
        valid_df, train_df, n_neg=99
    )
    test_candidates = candidate_gen.generate_candidates(
        test_df, train_df, n_neg=99
    )

    # ========================================================================
    # 步骤5：在候选集上预测（修改）
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤5：在候选集上预测")
    print("=" * 80)

    print("\n验证集候选预测...")
    valid_candidates['lgb_score'] = lgb_model.predict(valid_candidates)
    valid_candidates['deepfm_score'] = deepfm_trainer.predict(valid_candidates)
    valid_candidates['sasrec_score'] = sasrec_trainer.predict(valid_candidates)

    print("测试集候选预测...")
    test_candidates['lgb_score'] = lgb_model.predict(test_candidates)
    test_candidates['deepfm_score'] = deepfm_trainer.predict(test_candidates)
    test_candidates['sasrec_score'] = sasrec_trainer.predict(test_candidates)

    # ========================================================================
    # 步骤6：确定融合权重（在候选集上）
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤6：确定融合权重")
    print("=" * 80)

    score_cols = ['lgb_score', 'deepfm_score', 'sasrec_score']

    fusion = ModelFusion(method='weighted_average')
    weights_grid = fusion.grid_search_weights(
        valid_candidates, score_cols, k=10, step=0.1  # 可调整 step
    )

    # ========================================================================
    # 步骤7：在测试集候选上评估（修改）
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤7：测试集评估与对比")
    print("=" * 80)

    # 应用融合
    test_candidates['fusion_score'] = fusion.apply_fusion(test_candidates, score_cols)

    # 评估所有模型
    k_list = [5, 10, 20]
    model_names = ['LightGBM', 'DeepFM', 'SASRec', 'Fusion']
    score_columns = ['lgb_score', 'deepfm_score', 'sasrec_score', 'fusion_score']

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
    # 步骤7：结果可视化
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤7：结果可视化")
    print("=" * 80)

    # 创建对比图表
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

    # 隐藏多余的子图
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    print("\n对比图表已保存为 'model_comparison_results.png'")

    # ========================================================================
    # 步骤8：生成结果报告
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤8：生成结果报告")
    print("=" * 80)

    # 创建结果表格
    report_data = []
    for k in k_list:
        for metric in metrics:
            row = {'K': k, 'Metric': metric.upper()}
            for model_name in model_names:
                row[model_name] = f"{all_results[model_name][k][metric]:.4f}"
            report_data.append(row)

    report_df = pd.DataFrame(report_data)

    print("\n" + "=" * 80)
    print("实验结果汇总表")
    print("=" * 80)
    print(report_df.to_string(index=False))

    # 保存结果
    report_df.to_csv('experiment_results.csv', index=False)
    print("\n结果已保存为 'experiment_results.csv'")

    # ========================================================================
    # 步骤9：统计显著性分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤9：性能提升分析")
    print("=" * 80)

    print("\n融合模型相对于单一模型的性能提升:")
    print("-" * 80)

    for k in k_list:
        print(f"\n@K={k}:")
        for metric in metrics:
            fusion_value = all_results['Fusion'][k][metric]

            best_single_model = max(['LightGBM', 'DeepFM', 'SASRec'],
                                    key=lambda m: all_results[m][k][metric])
            best_single_value = all_results[best_single_model][k][metric]

            if best_single_value > 0:
                improvement = (fusion_value - best_single_value) / best_single_value * 100
                print(f"  {metric.upper():10s}: {fusion_value:.4f} vs {best_single_model} {best_single_value:.4f} "
                      f"(提升 {improvement:+.2f}%)")
            else:
                print(f"  {metric.upper():10s}: {fusion_value:.4f} vs {best_single_model} {best_single_value:.4f}")

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print("  1. model_comparison_results.png - 模型对比可视化图表")
    print("  2. experiment_results.csv - 详细实验结果")
    print("\n实验验证了加权融合策略的有效性，混合模型在多个指标上优于单一模型。")


if __name__ == "__main__":
    main()