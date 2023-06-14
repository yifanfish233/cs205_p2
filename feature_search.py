import time
from copy import deepcopy

import data_utils as du
import numpy as np
import self_nn
from sklearn.model_selection import KFold
import random


def forward_selection(X, y, n_splits=10, random_state=80):
    #modify the code to use the KNN classifier and score to analysis the feature.
    print('Beginning search.')
    start = time.time()
    # Initialize with all features
    q = []
    for i in range(X.shape[1]):
        q.append([i])
    
    max_accuracy = 0
    best_features = q[0]
    # print(X.shape[1],len(X[0]),X.shape[0]) # 10, 10, 1000
    # print(len(y),y[0]) # 1000, 2.0
    fold_size = X.shape[0] // n_splits # 1000 // 10
    print(fold_size)

    kf = du.k_fold_cross_validation(X, n_splits, random_state)
    # kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    print(q)
    while len(q[0]) < X.shape[1]:
        cur_best_features = None
        cur_max_acc = 0

        for feature_subset in q: # [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
            acc_cnt = 0
            test_class = [0] * X.shape[0]

            for i in range(n_splits): # 0,1,2,3,4,5,6,7,8,9
                for test_num in range(fold_size*i, fold_size*i + fold_size): # 0 - 99
                    min_distance = float("inf")

                    for train_num in range(X.shape[0]): # 0 - 999
                        distance = 0
                        if train_num >= fold_size*i + fold_size or train_num < fold_size*i: # 100-999

                            for k in feature_subset: # [0]
                                distance += (X[train_num][k] - X[test_num][k]) ** 2

                            if distance < min_distance:
                                min_distance = distance 
                                test_class[test_num] = y[train_num]
                                                
            for i in range(1000):
                if y[i] == test_class[i]:
                    acc_cnt += 1
            # print(acc_cnt)
            if acc_cnt > cur_max_acc:
                cur_max_acc = acc_cnt
                cur_best_features = feature_subset
        # Check if we found a new best
        if cur_max_acc > max_accuracy:
            max_accuracy = cur_max_acc
            best_features = cur_best_features
            print(f'Current best is using features {[f + 1 for f in best_features]} with accuracy {round(max_accuracy / X.shape[0] * 100, 2)}%')
        else:
            print(f'Current best is the last one using features {[f + 1 for f in cur_best_features]} with accuracy {round(cur_max_acc / X.shape[0] * 100, 2)}%')

        # Generate next level feature subsets
        q = []
        for i in range(X.shape[1]):
            if i not in cur_best_features:
                q.append(cur_best_features + [i])
        print(q)
    end = time.time()
    print("time cost: ", end - start)
    print(f'Final result: The best feature set is {[f + 1 for f in best_features]}, with accuracy {round(max_accuracy / X.shape[0]* 100, 2)}%')
    return best_features, max_accuracy

def backward_elimination(X, y, threshold=0.70, test_size=0.2, n_splits=5, speed_priority=False, random_state=100):
    X_train, X_test, y_train, y_test = du.self_train_test_split(X, y, test_size=test_size, random_state=random_state)

    idx_list = list(range(X_train.shape[1]))
    max_accuracy = 0
    best_subset = idx_list  # Initialize best_subset
    best_features = idx_list

    accuracies = []
    times = []

    while len(best_features) > 1:
        cur_max_accuracy = 0
        cur_best_features = None

        subsets = []
        for i in range(len(best_features)):
            subset = deepcopy(best_features)
            subset.pop(i)
            subsets.append(subset)

        start_time = time.time()

        for feature_subset in subsets:
            kf = du.k_fold_cross_validation(X_train[:, feature_subset], n_splits, random_state)
            accuracy_list = []

            if speed_priority:
                random_split = random.choice(kf)  # randomly select one split from kf
                train_index = random_split[0]
                val_index = random_split[1]
                X_train_fold, X_val_fold = X_train[train_index][:, feature_subset], X_train[val_index][:, feature_subset]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                knn = self_nn.MyNNClassifier()
                knn.fit(X_train_fold, y_train_fold)
                accuracy = knn.get_score(X_val_fold, y_val_fold)

                accuracy_list.append(accuracy)
            else:
                for train_index, val_index in kf:
                    X_train_fold, X_val_fold = X_train[train_index][:, feature_subset], X_train[val_index][:, feature_subset]
                    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                    knn = self_nn.MyNNClassifier()
                    knn.fit(X_train_fold, y_train_fold)
                    accuracy = knn.get_score(X_val_fold, y_val_fold)

                    accuracy_list.append(accuracy)

            avg_accuracy = np.mean(accuracy_list)

            print(f"Testing subset: {[f + 1 for f in feature_subset]}, accuracy: {round(avg_accuracy * 100, 2)}%")
            if avg_accuracy > cur_max_accuracy:
                cur_max_accuracy = avg_accuracy
                cur_best_features = feature_subset

        end_time = time.time()

        accuracies.append(round(cur_max_accuracy, 3))
        times.append(round(end_time - start_time, 3))

        if cur_max_accuracy > max_accuracy:
            max_accuracy = cur_max_accuracy
            best_subset = cur_best_features
        elif cur_max_accuracy < max_accuracy and cur_max_accuracy >= threshold:
            print(f"Even if accuracy didn't increase, but still more than {threshold * 100.0}%, continue the search with backward elimination")

        best_features = cur_best_features

        print(f'Iteration results: The best feature set is {[f + 1 for f in best_features]}, with accuracy {round(cur_max_accuracy * 100, 2)}%')

    knn = self_nn.MyNNClassifier()
    knn.fit(X_train[:, best_subset], y_train)
    test_accuracy = knn.get_score(X_test[:, best_subset], y_test)

    print(f'Final result: The best feature set is {[f + 1 for f in best_subset]}, with accuracy {round(test_accuracy * 100, 2)}% on test set')
    return best_subset, test_accuracy, accuracies, times







