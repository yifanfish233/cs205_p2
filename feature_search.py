from copy import deepcopy

import data_utils as du
import numpy as np
import self_knn
from sklearn.model_selection import KFold
def forward_selection(X, y, n_splits=5, random_state=42):
    #modify the code to use the KNN classifier and score to analysis the feature.
    print('Beginning search.')

    # Initialize with all features
    q = []
    for i in range(X.shape[1]):
        q.append([i])
    
    max_accuracy = 0
    best_features = q[0]
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    
    while len(q[0]) < X.shape[1]:
        cur_max_accuracy = 0
        cur_best_features = None
        # print("q: " , q)
        for feature_subset in q:
            accuracies = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index][:, feature_subset], X[test_index][:, feature_subset]
                y_train, y_test = y[train_index], y[test_index]

                knn = self_knn.MyKNNClassifier(n_neighbors=2)
                knn.fit(X_train, y_train)
                accuracy = knn.get_score(X_test, y_test)
                accuracies.append(accuracy)

            avg_accuracy = np.mean(accuracies)
            if avg_accuracy > cur_max_accuracy:
                cur_max_accuracy = avg_accuracy
                cur_best_features = feature_subset

        # Check if we found a new best
        if cur_max_accuracy > max_accuracy:
            max_accuracy = cur_max_accuracy
            best_features = cur_best_features
            print(f'Current best is using features {[f + 1 for f in best_features]} with accuracy {round(max_accuracy * 100, 2)}%')
        else:
            # break
            print(f'Current best is using features {[f + 1 for f in cur_best_features]} with accuracy {round(cur_max_accuracy * 100, 2)}%')

        # Generate next level feature subsets
        q = []
        for i in range(X.shape[1]):
            if i not in cur_best_features:
                q.append(cur_best_features + [i])


    print(f'Final result: The best feature set is {[f + 1 for f in best_features]}, with accuracy {round(max_accuracy * 100, 2)}%')
    return best_features, max_accuracy


def backward_elimination(X, y, n_splits=5, random_state=42):
    print('Beginning search.')

    # Initialize with all features
    idx_list = list(range(X.shape[1]))
    max_accuracy = 0
    best_features = idx_list
    feature_max_accuracy = [0]*len(idx_list)  # Keep track of max accuracy for each feature
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    q = [idx_list]
    while len(q) > 0:
        cur_max_accuracy = 0
        cur_best_features = None

        for feature_subset in q:
            accuracies = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index][:, feature_subset], X[test_index][:, feature_subset]
                y_train, y_test = y[train_index], y[test_index]

                knn = self_knn.MyKNNClassifier(n_neighbors=2)
                knn.fit(X_train, y_train)
                accuracy = knn.get_score(X_test, y_test)
                accuracies.append(accuracy)

            avg_accuracy = np.mean(accuracies)
            if avg_accuracy > cur_max_accuracy:
                cur_max_accuracy = avg_accuracy
                cur_best_features = feature_subset
                for f in feature_subset:  # Update max accuracy for each feature in the current subset
                    feature_max_accuracy[f] = max(feature_max_accuracy[f], cur_max_accuracy)

        # Check if we found a new best
        if cur_max_accuracy > max_accuracy:
            max_accuracy = cur_max_accuracy
            best_features = cur_best_features
            print(f'Current best is using features {[f + 1 for f in best_features]} with accuracy {round(max_accuracy * 100, 2)}%')
        else:
            break

        # Generate next level feature subsets with early purning.
        q = []
        for i in range(len(best_features)):
            if cur_max_accuracy >= feature_max_accuracy[best_features[i]]:
                subset = deepcopy(best_features)
                subset.pop(i)
                q.append(subset)

    print(f'Final result: The best feature set is {[f + 1 for f in best_features]}, with accuracy {round(max_accuracy * 100, 2)}%')
    return best_features, max_accuracy







