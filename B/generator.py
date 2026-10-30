import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

np.random.seed(42)

X1, y1 = make_blobs(
    n_samples=80,
    centers=[[-1.5, -1.5], [1.5, 1.5]],
    cluster_std=1.0,
    random_state=42
)

df1 = pd.DataFrame(X1, columns=['Feature_1', 'Feature_2'])
df1['Class'] = y1
print("Dataset 1 head:")
print(df1.head())