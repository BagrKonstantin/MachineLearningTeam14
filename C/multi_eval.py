import numpy as np
import random
import math
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from scipy.optimize import linear_sum_assignment
import skfuzzy as fuzz

def cluster_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[row, col] for row, col in zip(row_ind, col_ind)]) / y_pred.size

class Flower:
    def __init__(self, vec, growth, n, art=False):
        self.vec = vec
        self.growth = growth
        self.n = n
        self.art = art

class Pollen:
    def __init__(self, vec, growth, n):
        self.vec = vec
        self.growth = growth
        self.n = n

class Bee:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pollen = None

    def pick_up(self, obj):
        self.pollen = Pollen(obj.vec, obj.growth, obj.n)

    def clear(self):
        self.pollen = None

    def has_pollen(self):
        return self.pollen is not None

class FPABClustering:
    def __init__(self, n_clusters=3, max_iter=60, P_pickup=0.5, P_dist=0.5, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.P_pickup = P_pickup
        self.P_dist = P_dist
        self.seed = seed

    def fit(self, X):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        N, d = X.shape
        self.N = N
        self.d = d
        
        diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        self.D_max = np.max(dists)
        if self.D_max == 0: self.D_max = 1.0

        def compat(v1, v2):
            return 1.0 - (np.linalg.norm(v1 - v2) / self.D_max)

        self.compat = compat
        self.m = int(math.sqrt(4 * N))
        self.grid = [[[] for _ in range(self.m)] for _ in range(self.m)]
        
        for i in range(N):
            while True:
                rx, ry = random.randint(0, self.m - 1), random.randint(0, self.m - 1)
                if len(self.grid[rx][ry]) == 0:
                    self.grid[rx][ry].append(Flower(X[i], 0.0, i))
                    break

        n_bees = N // 3
        self.bees = [Bee(random.randint(0, self.m - 1), random.randint(0, self.m - 1)) for _ in range(n_bees)]

        def cell_center(cell):
            if len(cell) == 0: return np.zeros(self.d)
            return np.mean([obj.vec for obj in cell], axis=0)

        def update_cell_growth(cell):
            if len(cell) == 0: return
            c = cell_center(cell)
            for obj in cell:
                obj.growth = compat(obj.vec, c)

        for it in range(self.max_iter):
            for bee in self.bees:
                dx, dy = random.choice([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)])
                bee.x = (bee.x + dx) % self.m
                bee.y = (bee.y + dy) % self.m
                cell = self.grid[bee.x][bee.y]
                
                if len(cell) == 0:
                    continue

                if not bee.has_pollen():
                    if len(cell) == 1:
                        if random.random() > self.P_pickup:
                            bee.pick_up(cell[0])
                    elif len(cell) == 2:
                        if random.random() > self.P_dist:
                            bee.pick_up(random.choice(cell))
                    else:
                        min_obj = min(cell, key=lambda o: o.growth)
                        bee.pick_up(min_obj)
                else:
                    pollen = bee.pollen
                    c = cell_center(cell)
                    cand_growth = compat(pollen.vec, c)
                    min_obj = min(cell, key=lambda o: o.growth)
                    
                    if min_obj.growth < pollen.growth and len(cell) > 1:
                        bee.pick_up(min_obj)
                        min_obj.vec = pollen.vec
                        min_obj.growth = cand_growth
                        min_obj.n = pollen.n
                        update_cell_growth(cell)
                    elif cand_growth > pollen.growth and not any(o.n == pollen.n for o in cell):
                        cell.append(Pollen(pollen.vec, cand_growth, pollen.n))
                        update_cell_growth(cell)
                        bee.clear()

            max_growth = {}
            for i in range(self.m):
                for j in range(self.m):
                    for obj in self.grid[i][j]:
                        if obj.n not in max_growth or obj.growth > max_growth[obj.n]:
                            max_growth[obj.n] = obj.growth

            for i in range(self.m):
                for j in range(self.m):
                    cell = self.grid[i][j]
                    if len(cell) >= 2:
                        survivors = [obj for obj in cell if obj.growth >= max_growth[obj.n]]
                        if len(survivors) < len(cell):
                            self.grid[i][j] = survivors
                            update_cell_growth(self.grid[i][j])

        gardens = []
        for i in range(self.m):
            for j in range(self.m):
                cell = self.grid[i][j]
                if len(cell) >= 2:
                    c = cell_center(cell)
                    gardens.append((c, len(cell)))

        gardens.sort(key=lambda g: g[1], reverse=True)
        garden_vecs = np.array([g[0] for g in gardens])
        km_merge = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10).fit(garden_vecs)
        self.swarm_centroids_ = km_merge.cluster_centers_

        dists_to_swarm = np.linalg.norm(X[:, np.newaxis, :] - self.swarm_centroids_[np.newaxis, :, :], axis=-1)
        self.pure_swarm_labels_ = np.argmin(dists_to_swarm, axis=-1)

        dists_to_centers = np.maximum(dists_to_swarm, 1e-8)
        inv_dists = 1.0 / dists_to_centers
        u0 = (inv_dists / np.sum(inv_dists, axis=-1, keepdims=True)).T

        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X.T, self.n_clusters, 2, error=0.005, maxiter=1000, init=u0
        )
        self.hybrid_centroids_ = cntr
        self.hybrid_labels_ = np.argmax(u, axis=0)
        return self

iris = datasets.load_iris()
X = iris.data
y = iris.target
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

metrics_summary = []
for s in [10, 42, 99, 123, 2024]:
    km = KMeans(n_clusters=3, random_state=s, n_init=1).fit(X_norm)
    _, u_fcm, _, _, _, fcm_iters, _ = fuzz.cluster.cmeans(X_norm.T, 3, 2, error=0.005, maxiter=1000, seed=s)
    fcm_labels = np.argmax(u_fcm, axis=0)
    fpab = FPABClustering(n_clusters=3, max_iter=50, seed=s).fit(X_norm)
    
    metrics_summary.append({
        'seed': s,
        'KM_Sil': silhouette_score(X_norm, km.labels_),
        'KM_DBI': davies_bouldin_score(X_norm, km.labels_),
        'KM_ARI': adjusted_rand_score(y, km.labels_),
        'KM_NMI': normalized_mutual_info_score(y, km.labels_),
        'KM_Acc': cluster_accuracy(y, km.labels_),
        
        'FCM_Sil': silhouette_score(X_norm, fcm_labels),
        'FCM_DBI': davies_bouldin_score(X_norm, fcm_labels),
        'FCM_ARI': adjusted_rand_score(y, fcm_labels),
        'FCM_NMI': normalized_mutual_info_score(y, fcm_labels),
        'FCM_Acc': cluster_accuracy(y, fcm_labels),
        'FCM_Iters': fcm_iters,
        
        'PureFPAB_Sil': silhouette_score(X_norm, fpab.pure_swarm_labels_),
        'PureFPAB_DBI': davies_bouldin_score(X_norm, fpab.pure_swarm_labels_),
        'PureFPAB_ARI': adjusted_rand_score(y, fpab.pure_swarm_labels_),
        'PureFPAB_NMI': normalized_mutual_info_score(y, fpab.pure_swarm_labels_),
        'PureFPAB_Acc': cluster_accuracy(y, fpab.pure_swarm_labels_),
        
        'Hybrid_Sil': silhouette_score(X_norm, fpab.hybrid_labels_),
        'Hybrid_DBI': davies_bouldin_score(X_norm, fpab.hybrid_labels_),
        'Hybrid_ARI': adjusted_rand_score(y, fpab.hybrid_labels_),
        'Hybrid_NMI': normalized_mutual_info_score(y, fpab.hybrid_labels_),
        'Hybrid_Acc': cluster_accuracy(y, fpab.hybrid_labels_),
    })

import pandas as pd
df = pd.DataFrame(metrics_summary)
print(df.mean())
