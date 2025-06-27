# 🎵 尼日利亚歌曲聚类分析项目

本项目旨在通过机器学习中的聚类方法，分析 Spotify 上尼日利亚听众喜欢的音乐类型。使用了 K-Means 算法，并结合主成分分析（PCA）进行可视化。

## 📌 项目亮点

- 使用 K-Means 聚类分析音乐风格
- 使用肘部法（Elbow Method）确定最佳聚类数量
- 利用 PCA 将多维特征降维到二维进行可视化
- 输出包含聚类结果的新数据文件

## 📁 项目文件

- `nigerian-songs.csv`：原始数据集
- `music_clustering.py`：主分析脚本
- `images/`：保存生成的图像
- `nigerian-songs-clustered.csv`：包含聚类结果的输出文件

## 🧠 如何运行

请确保已安装以下依赖库：

```bash
pip install pandas scikit-learn matplotlib
```

运行主脚本：

```bash
python music_clustering.py
```

## 📈 输出图示

- `images/elbow_method.png`：用于判断聚类数的肘部法图
- `images/cluster_visualization.png`：聚类结果的 PCA 可视化图

---

## 📜 源代码：music_clustering.py

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("nigerian-songs.csv")

# 选择聚类特征
features = ['danceability', 'acousticness', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo']

X = df[features]

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用肘部法确定K值
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig("images/elbow_method.png")

# KMeans 聚类
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# PCA降维可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Music Clustering PCA Projection')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Cluster')
plt.savefig("images/cluster_visualization.png")

# 添加聚类结果
df['cluster'] = clusters
df.to_csv("nigerian-songs-clustered.csv", index=False)
print("聚类完成，输出已保存。")
```
