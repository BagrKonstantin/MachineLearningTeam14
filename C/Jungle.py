from numpy import ndarray
from math import sqrt
from random import randint, choice
import os

import skfuzzy as fuzz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from scipy.optimize import linear_sum_assignment

from Entities.Flower import Flower
from Cell import Cell
from Entities.Bee import Bee


def cluster_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[row, col] for row, col in zip(row_ind, col_ind)]) / y_pred.size


def plot_clusters(data, labels, cntr, filename="Ex2/fpab_fcm_clusters.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    num_clusters = len(cntr)
    for j in range(num_clusters):
        ax.scatter(data[labels == j, 0],
                   data[labels == j, 1],
                   label=f'Cluster {j + 1}',
                   alpha=0.7, edgecolors='k')
    for pt in cntr:
        ax.scatter(pt[0], pt[1], marker='X', s=150, c='red', edgecolors='black', linewidths=1.5)
    ax.set_title("FPAB + FCM Hybrid Clustering Result", fontsize=12)
    ax.set_xlabel("Feature 1 (Sepal Length / Scaled)")
    ax.set_ylabel("Feature 2 (Sepal Width / Scaled)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


class Jungle:
    def __init__(self, data: ndarray, seed: int = 42):
        np.random.seed(seed)
        self.raw_data = np.copy(data)
        self.n: int = len(data)
        self.d: int = data.shape[1]
        self.m: int = int(sqrt(4 * self.n))
        
        diffs = data[:, np.newaxis, :] - data[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        self.d_max = float(np.max(dists))
        if self.d_max <= 0:
            self.d_max = 1.0

        self.grid: list[list[Cell]] = [[Cell(self.d_max) for _ in range(self.m)] for _ in range(self.m)]
        self.bees: list[Bee] = list()
        self.init_cells(data)
        self.init_bees()

    def init_cells(self, data: ndarray):
        for i, vec in enumerate(data):
            x, y = self.get_rand_point()
            while not self.grid[x][y].is_empty():
                x, y = self.get_rand_point()
            self.grid[x][y].put_flower(Flower(vec, 0.0, i))

    def init_bees(self):
        for i in range(self.n // 3):
            self.bees.append(Bee(*self.get_rand_point()))

    def get_rand_point(self):
        return randint(0, self.m - 1), randint(0, self.m - 1)

    def natural_selection(self):
        growth_of_flowers = dict()
        for i in range(self.m):
            for j in range(self.m):
                if self.grid[i][j].is_empty():
                    continue
                for obj in self.grid[i][j].objects:
                    if obj.n not in growth_of_flowers or obj.growth > growth_of_flowers[obj.n]:
                        growth_of_flowers[obj.n] = obj.growth

        for i in range(self.m):
            for j in range(self.m):
                cell: Cell = self.grid[i][j]
                if not cell.is_garden():
                    continue
                survivors = [obj for obj in cell.objects if obj.growth >= growth_of_flowers[obj.n]]
                if len(survivors) < cell.size():
                    cell.objects = survivors
                    cell.update_all_growths()

    def fly_bee(self, bee: Bee):
        dx, dy = choice([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)])
        bee.x = (bee.x + dx) % self.m
        bee.y = (bee.y + dy) % self.m

        target_cell: Cell = self.grid[bee.x][bee.y]
        if target_cell.is_empty():
            return

        if bee.has_pollen():
            target_cell.pollinate(bee)
        else:
            target_cell.pick_up_pollen(bee)

    def count_clusters(self) -> int:
        clusters = 0
        for row in self.grid:
            for cell in row:
                if cell.is_garden():
                    clusters += 1
        return clusters

    def get_candidate_gardens(self) -> list[tuple[np.ndarray, int]]:
        gardens = []
        for i in range(self.m):
            for j in range(self.m):
                cell: Cell = self.grid[i][j]
                if cell.is_garden():
                    gardens.append((cell.center(), cell.size()))
        return gardens

    def run(self, target_clusters: int = 3, max_iter: int = 60, ground_truth=None):
        for it in range(max_iter):
            for bee in self.bees:
                self.fly_bee(bee)
            self.natural_selection()
            if (it + 1) % 15 == 0 or it == max_iter - 1:
                n_gardens = self.count_clusters()
                print(f"  Iteration {it + 1:2d}/{max_iter}: {n_gardens} active gardens.")

        gardens = self.get_candidate_gardens()

        if len(gardens) < target_clusters:
            cents = [g[0] for g in gardens]
            while len(cents) < target_clusters:
                cents.append(self.raw_data[randint(0, self.n - 1)])
            swarm_seeds = np.array(cents[:target_clusters])
        elif len(gardens) == target_clusters:
            swarm_seeds = np.array([g[0] for g in gardens])
        else:
            gardens.sort(key=lambda g: g[1], reverse=True)
            garden_vecs = np.array([g[0] for g in gardens])
            km_merge = KMeans(n_clusters=target_clusters, random_state=42, n_init=10).fit(garden_vecs)
            swarm_seeds = km_merge.cluster_centers_

        dists = np.linalg.norm(self.raw_data[:, np.newaxis, :] - swarm_seeds[np.newaxis, :, :], axis=-1)
        dists = np.maximum(dists, 1e-8)
        inv_dists = 1.0 / dists
        u0 = (inv_dists / np.sum(inv_dists, axis=-1, keepdims=True)).T

        cntr, u, _, _, _, fcm_iters, _ = fuzz.cluster.cmeans(
            self.raw_data.T, target_clusters, 2, error=0.005, maxiter=1000, init=u0
        )
        labels = np.argmax(u, axis=0)

        plot_clusters(self.raw_data, labels, cntr)

        sil = silhouette_score(self.raw_data, labels)
        dbi = davies_bouldin_score(self.raw_data, labels)
        ch = calinski_harabasz_score(self.raw_data, labels)

        print(f"Silhouette Score: {sil:.4f}")
        print(f"Davies-Bouldin Index: {dbi:.4f}")
        print(f"Calinski-Harabasz Index: {ch:.4f}")
        print(f"FCM Iterations to Converge: {fcm_iters}")

        if ground_truth is not None:
            ari = adjusted_rand_score(ground_truth, labels)
            nmi = normalized_mutual_info_score(ground_truth, labels)
            acc = cluster_accuracy(ground_truth, labels)
            print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
            print(f"  Normalized Mutual Info: {nmi:.4f}")
            print(f"  Clustering Accuracy: {acc:.2%}")

        return cntr, labels
