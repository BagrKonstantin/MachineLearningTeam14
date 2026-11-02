from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from Jungle import Jungle, cluster_accuracy

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)


    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_norm)
    km_labels = km.labels_
    print(f"Silhouette Score: {silhouette_score(X_norm, km_labels):.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score(X_norm, km_labels):.4f}")
    print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(y, km_labels):.4f}")
    print(f"Normalized Mutual Info: {normalized_mutual_info_score(y, km_labels):.4f}")
    print(f"Clustering Accuracy: {cluster_accuracy(y, km_labels):.2%}")

    jungle = Jungle(X_norm, seed=42)
    cntr, labels = jungle.run(target_clusters=3, max_iter=50, ground_truth=y)
