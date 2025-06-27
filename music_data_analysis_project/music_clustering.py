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
