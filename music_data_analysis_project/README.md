# Music Data Analysis Project

This project analyzes the musical taste of Nigerian audiences using clustering techniques on Spotify data.

## 📊 Features

- K-Means clustering on musical features
- Elbow method to determine optimal number of clusters
- PCA-based 2D visualization
- Dataset: Nigerian songs from Spotify (530 entries)

## 📁 Files

- `nigerian-songs.csv`: Raw dataset
- `music_clustering.py`: Python script for clustering and visualization
- `images/`: Output visualizations (elbow method, clusters)
- `nigerian-songs-clustered.csv`: Dataset with added cluster column

## 🔧 How to Run

```bash
pip install pandas scikit-learn matplotlib
python music_clustering.py
```

## 📈 Visualizations

- `elbow_method.png`: Shows WCSS to determine optimal `k`
- `cluster_visualization.png`: PCA projection of clusters

