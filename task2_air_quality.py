# Task 2 - Unsupervised Learning: Air Quality Dataset
# BS Artificial Intelligence - 4th Semester - Machine Learning Assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv('AirQualityUCI.csv', sep=';')
print("Dataset shape:", df.shape)

# Replace missing values (-200 sentinel) with NaN
df = df.replace(-200, np.nan)
df = df.dropna(axis=1, how='all')

# Fill remaining NaN with column median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Select features for clustering
features = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S1(CO)']
available = [f for f in features if f in df.columns]
print("Features used:", available)

X = df[available].copy()
X = X.dropna()
for col in X.columns:
    X[col] = X[col].astype(str).str.replace(',', '.').astype(float)

# Standardize features (required for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method + Silhouette to find optimal k
inertias, silhouettes = [], []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

print("\nSilhouette scores:")
for k, s in zip(k_range, silhouettes):
    print(f"  k={k}: {s:.4f}")

# Plot Elbow and Silhouette
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-', linewidth=2)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouettes, 'ro-', linewidth=2)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()

# Final model with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, df['Cluster'])
print(f"\nFinal Silhouette Score (k=3): {sil:.4f}")
print("\nCluster sizes:")
print(df['Cluster'].value_counts().sort_index())
print("\nCluster means:")
print(df.groupby('Cluster')[available].mean().round(2))

# Scatter plot: CO vs NO2
colors = ['#1D9E75', '#854F0B', '#A32D2D']
cluster_names = ['Cluster 0 - Clean Air',
                 'Cluster 1 - Moderate Pollution',
                 'Cluster 2 - High Pollution']

if 'CO(GT)' in available and 'NO2(GT)' in available:
    plt.figure(figsize=(9, 6))
    for i in range(3):
        mask = df['Cluster'] == i
        plt.scatter(df.loc[mask, 'CO(GT)'],
                    df.loc[mask, 'NO2(GT)'],
                    c=colors[i], label=cluster_names[i],
                    alpha=0.55, s=20)

    # Plot centroids
    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    co_idx = available.index('CO(GT)')
    no2_idx = available.index('NO2(GT)')
    for i, c in enumerate(centroids_orig):
        plt.scatter(c[co_idx], c[no2_idx],
                    c=colors[i], marker='+',
                    s=400, linewidths=3, zorder=5)

    plt.xlabel('CO concentration (mg/m³)', fontsize=12)
    plt.ylabel('NO2 concentration (ug/m3)', fontsize=12)
    plt.title('K-Means Clustering (k=3) - CO vs NO2\nAir Quality Dataset', fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('cluster_scatter.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Scatter plot saved as cluster_scatter.png")
