# D — Model evaluation and validation strategies

## D.1 — Single split and training accuracy

The Dataset 1 from B.I.1 was generated with `make_blobs` (seed 42). We have 80 points with two continuous features and 40 points per class. No additional normalization was applied because the features already have a similar scale (standard deviation $\approx 1.8$).

The data is split with `train_test_split` into 80% training data (64 points) and 20% test data (16 points). We used a **stratified** split to make sure that both classes are represented equally.

We fit three `SVC` models with the RBF kernel and $C\in\{0.01,\,1,\,100\}$. The regularization factor $C$ is the upper bound of the Lagrange multipliers in the dual problem:

$$
0\le\alpha_i\le C,\qquad \sum_i\alpha_i y_i=0 \tag{1}
$$

A small $C$ tolerates more misclassifications and keeps the margin wide while a large $C$ punishes every error and makes the margin narrow. The accuracy is computed as:

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN} \tag{2}
$$

| $C$ | Training accuracy |
| ---: | :---: |
| 0.01 | 1.000 |
| 1 | 1.000 |
| 100 | 1.000 |

*Table 1: Training accuracy of the SVC models with 80/20 split*

All three models classified the training data perfectly. This was expected because the classes of Dataset 1 are linearly separable. However because the training accuracy is maesured on the same points the models were fitted on, the result is optimistic and doesn't tell us anything about unseen data. Here the three models cannot be distinguished.

## D.2 — 5-fold cross-validation

We take the same 64 training points and split them with `StratifiedKFold` into 5 folds. Every model is trained 5 times on 4 folds ($\approx 51$ points) and validated on the remaining fold ($\approx 13$ points). For each $C$ we report the mean and the standard deviation.

$$
\bar a=\frac{1}{5}\sum_{k=1}^{5}a_k,\qquad
\sigma_a=\sqrt{\frac{1}{5}\sum_{k=1}^{5}(a_k-\bar a)^2} \tag{3}
$$

| $C$ | Validation accuracy (mean $\pm$ std) | Validation accuracy per fold |
| ---: | :---: | :---: |
| 0.01 | $0.569\pm0.215$ | 0.462, 0.462, 0.462, 0.462, 1.000 |
| 1 | $1.000\pm0.000$ | 1.000 on all folds |
| 100 | $1.000\pm0.000$ | 1.000 on all folds |

*Table 2: 5-fold cross-validation accuracy on the validation folds*

$C=1$ and $C=100$ are perfect on every fold. For $C=0.01$ the mean validation accuracy drops to $0.569$. On four of the five folds it is $0.462$ which is exactly $6/13$. The model predicted the same class for all 13 validation points. Every $\alpha_i$ is limited to $0.01$. This means that no training point has much influence on the decision function. The model is too strongly regularized and underfits. In those four folds the training data contains one more point of one class (26/25 or 25/26) and the model falls back to the class with the majority. On the fifth fold the training data is balanced (26/26) and therefore classifies all points correctly. The same happened in D.1 where the training set was exactly balanced (32/32) which is why the model looked perfect there.

The cross-validation provides a more reliable estimate. The single split measures accuracy on data already seen during training and rewards memorization. Cross-validation evaluates on unknown data and averages over 5 different partitions instead of one particular split. The standard deviation also shows the stability of a model which a single number cannot do. The downside is that it needs more computations because each model has to be trained 5 times instead of once. (Negligeable for our dataset)

## D.3 — Model selection and test evaluation

We pick the best $C$ by the highest mean validation accuracy from D.2 not by the training accuracy from D.1 or the test set. $C=0.01$ is excluded because it underfits and $C=1$ and $C=100$ are tied at $1.000$. We prefer the smaller $C$ because it gives a wider margin and a simpler model which is less likely to overfit. The chosen model is retrained on all 64 training points and evaluated once on the 16 test points.

$$
\boxed{C=1,\quad \text{test accuracy}=1.000\ (16/16).}
$$

|  | Predicted 0 | Predicted 1 |
| :--- | :---: | :---: |
| Actual 0 | 8 (TN) | 0 (FP) |
| Actual 1 | 0 (FN) | 8 (TP) |

*Table 3: Confusion matrix of $C=1$ on the test set*

| Estimate ($C=1$) | Accuracy |
| :--- | :---: |
| D.1 training accuracy (single split) | 1.000 |
| D.2 validation accuracy (mean $\pm$ std) | $1.000\pm0.000$ |
| D.3 test accuracy | 1.000 |

*Table 4: Comparison of the accuracy estimates*

All estimates for $C=1$ are 100%. There is no gap between training and test performance and no sign of over or underfitting. This is expected for a linearly separable dataset. The test accuracy is the only unbiased estimate because the test set was not used for training and also not for choosing $C$. Here the cross-validation predicted it correctly while the training accuracy from D.1 would have been "perfect" for $C=0.01$ as well.

To conclude with only 16 test points the accuracy can only change in steps of $6.25\%$ so the test estimate is rather imprecise. On a more challenging dataset the difference between $C=1$ and $C=100$ would probably become visible.
