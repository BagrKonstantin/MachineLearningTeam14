import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from scipy.optimize import linear_sum_assignment
import skfuzzy as fuzz
import random
import math

def cluster_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[row, col] for row, col in zip(row_ind, col_ind)]) / y_pred.size

print("Testing packages and metrics...")
iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_norm)
km_labels = km.labels_
print("K-Means:")
print("Silhouette:", silhouette_score(X_norm, km_labels))
print("Davies-Bouldin:", davies_bouldin_score(X_norm, km_labels))
print("Calinski-Harabasz:", calinski_harabasz_score(X_norm, km_labels))
print("ARI:", adjusted_rand_score(y, km_labels))
print("NMI:", normalized_mutual_info_score(y, km_labels))
print("Accuracy:", cluster_accuracy(y, km_labels))

cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_norm.T, 3, 2, error=0.005, maxiter=1000, seed=42)
fcm_labels = np.argmax(u, axis=0)
print("\nFCM:")
print("Silhouette:", silhouette_score(X_norm, fcm_labels))
print("Davies-Bouldin:", davies_bouldin_score(X_norm, fcm_labels))
print("Calinski-Harabasz:", calinski_harabasz_score(X_norm, fcm_labels))
print("ARI:", adjusted_rand_score(y, fcm_labels))
print("NMI:", normalized_mutual_info_score(y, fcm_labels))
print("Accuracy:", cluster_accuracy(y, fcm_labels))
