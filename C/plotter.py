import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from multi_eval import FPABClustering, cluster_accuracy

os.makedirs('Ex3', exist_ok=True)

iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_norm)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, name in enumerate(target_names):
    axes[0].scatter(X[y == i, 0], X[y == i, 1], label=name, color=colors[i], alpha=0.8, edgecolors='k')
axes[0].set_title("Iris: Sepal Length vs Sepal Width", fontsize=12)
axes[0].set_xlabel(feature_names[0])
axes[0].set_ylabel(feature_names[1])
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

for i, name in enumerate(target_names):
    axes[1].scatter(X[y == i, 2], X[y == i, 3], label=name, color=colors[i], alpha=0.8, edgecolors='k')
axes[1].set_title("Iris: Petal Length vs Petal Width", fontsize=12)
axes[1].set_xlabel(feature_names[2])
axes[1].set_ylabel(feature_names[3])
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('Ex3/iris_features.png', dpi=300)
plt.close()
print("Saved Ex2/iris_features.png")

km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_norm)
km_labels = km.labels_

_, u_fcm, _, _, _, _, _ = fuzz.cluster.cmeans(X_norm.T, 3, 2, error=0.005, maxiter=1000, seed=42)
fcm_labels = np.argmax(u_fcm, axis=0)

fpab = FPABClustering(n_clusters=3, max_iter=50, seed=42).fit(X_norm)
pure_labels = fpab.pure_swarm_labels_
hybrid_labels = fpab.hybrid_labels_

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

titles = [
    f"Iris Classes",
    f"K-Means (Acc: {cluster_accuracy(y, km_labels):.1%})",
    f"Fuzzy C-Means (Acc: {cluster_accuracy(y, fcm_labels):.1%})",
    f"FPAB + FCM (Acc: {cluster_accuracy(y, hybrid_labels):.1%})"
]
pred_list = [y, km_labels, fcm_labels, hybrid_labels]

for idx, ax in enumerate(axes.flat):
    pred = pred_list[idx]
    for c_idx in range(3):
        ax.scatter(X_pca[pred == c_idx, 0], X_pca[pred == c_idx, 1], label=f"Cluster {c_idx+1}" if idx > 0 else target_names[c_idx], alpha=0.8, edgecolors='k')
    ax.set_title(titles[idx], fontsize=12, fontweight='bold')
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('Ex3/clustering_comparison.png', dpi=300)
plt.close()
print("Saved Ex2/clustering_comparison.png")

grid_counts = np.zeros((fpab.m, fpab.m))
for i in range(fpab.m):
    for j in range(fpab.m):
        grid_counts[i, j] = len(fpab.grid[i][j])

plt.figure(figsize=(7, 6))
plt.imshow(grid_counts, cmap='YlGnBu', interpolation='nearest')
plt.colorbar(label='Number of Flowers / Pollens per Cell')
plt.title(f"FPAB Swarm Grid ({fpab.m}x{fpab.m}) at Iteration 50\nEmergence of Clustered Gardens (size >= 2)", fontsize=12)
plt.xlabel("Grid X")
plt.ylabel("Grid Y")
plt.tight_layout()
plt.savefig('Ex3/swarm_gardens_grid.png', dpi=300)
plt.close()
print("Saved Ex2/swarm_gardens_grid.png")

metrics_names = ['Silhouette\n(Cohesion/Sep)', 'Davies-Bouldin\n(Lower better)', 'ARI\n(Chance-adj)', 'NMI\n(Information)', 'Accuracy\n(Hungarian)']
km_vals = [0.496, 0.771, 0.710, 0.731, 0.884]
fcm_vals = [0.495, 0.776, 0.720, 0.736, 0.889]
pure_vals = [0.464, 0.742, 0.662, 0.730, 0.851]
hybrid_vals = [0.494, 0.776, 0.718, 0.733, 0.888]

x = np.arange(len(metrics_names))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - 1.5*width, km_vals, width, label='K-Means', color='#3498db')
plt.bar(x - 0.5*width, fcm_vals, width, label='Fuzzy C-Means', color='#e67e22')
plt.bar(x + 0.5*width, pure_vals, width, label='FPAB', color='#9b59b6')
plt.bar(x + 1.5*width, hybrid_vals, width, label='FPAB + FCM', color='#2ecc71')

plt.xticks(x, metrics_names, fontsize=10)
plt.ylabel('Score', fontsize=11)
plt.title('Benchmark Clustering Metrics across Models on Iris Dataset', fontsize=13, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Ex3/metrics_comparison.png', dpi=300)
plt.close()
print("Saved Ex2/metrics_comparison.png")
