import time
from copy import deepcopy

import data_utils as du
import numpy as np
import self_nn
from sklearn.model_selection import KFold
import random


def forward_selection(X, y, n_splits=5, random_state=42):
    #modify the code to use the KNN classifier and score to analysis the feature.
    print('Beginning search.')
    # Initialize with all features
    q = []
    for i in range(X.shape[1]):
        q.append([i])
    accuracies = []
    max_accuracy = 0
    best_features = q[0]
    worse_acc_cnt = 0
    diff_cnt = 0
    fold_size = X.shape[0] // n_splits # 1000 // 10
    times = []
    # print(q)
    while len(q[0]) < X.shape[1]:
        cur_best_features = None
        cur_max_acc = 0
        start = time.time()
        diff_cnt = 0
        flag = 0
        
        for feature_subset in q: # [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
            acc_cnt = 0
            test_class = [0] * X.shape[0]
            cur_diff_cnt = 0

            for i in range(n_splits): # 0,1,2,3,4,5,6,7,8,9
                if cur_diff_cnt > diff_cnt and diff_cnt > 0:
                    # print(f'Current is using features {[f + 1 for f in feature_subset]} with accuracy less than previous one')
                    break
                for test_num in range(fold_size*i, fold_size*i + fold_size): # 0 - 99
                    min_distance = float("inf")
                    if cur_diff_cnt > diff_cnt and diff_cnt > 0:
                        break
                    
                    for train_num in range(X.shape[0]): # 0 - 999
                        distance = 0
                        if train_num >= fold_size*i + fold_size or train_num < fold_size*i: # 100-999

                            for k in feature_subset: # [0]
                                distance += (X[train_num][k] - X[test_num][k]) ** 2

                            if distance < min_distance:
                                min_distance = distance 
                                test_class[test_num] = y[train_num]

                    if y[test_num] != test_class[test_num]:
                        cur_diff_cnt += 1
            diff_cnt = 0
            for i in range(X.shape[0]):
                if y[i] == test_class[i]:
                    acc_cnt += 1
                else:
                    diff_cnt += 1
            if acc_cnt > cur_max_acc:
                cur_max_acc = acc_cnt
                cur_best_features = feature_subset
            print(f'Current is using features {[f + 1 for f in feature_subset]} with accuracy {round(acc_cnt / X.shape[0] * 100, 2)}%')
        # Check if we found a new best
        accuracies.append(cur_max_acc / X.shape[0] * 100)
        if cur_max_acc > max_accuracy:
            worse_acc_cnt = 0
            max_accuracy = cur_max_acc
            best_features = cur_best_features
            print(f'Current best is using features {[f + 1 for f in best_features]} with accuracy {round(max_accuracy / X.shape[0] * 100, 2)}%')
        else:
            worse_acc_cnt += 1
            print(f'Current best is the last one using features {[f + 1 for f in cur_best_features]} with accuracy {round(cur_max_acc / X.shape[0] * 100, 2)}%')
        end = time.time()
        times.append(round(end - start, 3))
        # if worse_acc_cnt >= 5:
        #     break
        # Generate next level feature subsets
        q = []
        for i in range(X.shape[1]):
            if i not in cur_best_features:
                q.append(cur_best_features + [i])
    print(f'Final result: The best feature set is {[f + 1 for f in best_features]}, with accuracy {round(max_accuracy / X.shape[0]* 100, 2)}%')
    return best_features, accuracies, times


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

def nearest_neighbour_classify(X_train, y_train, X_test):
    y_pred = []
    for test_instance in X_test:
        distances = [euclidean_distance(test_instance, train_instance) for train_instance in X_train]
        nearest_neighbour_index = np.argmin(distances)
        y_pred.append(y_train[nearest_neighbour_index])
    return y_pred

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

                y_pred = nearest_neighbour_classify(X_train_fold, y_train_fold, X_val_fold)
                accuracy = np.mean(y_pred == y_val_fold)

                accuracy_list.append(accuracy)
            else:
                for train_index, val_index in kf:
                    X_train_fold, X_val_fold = X_train[train_index][:, feature_subset], X_train[val_index][:, feature_subset]
                    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                    y_pred = nearest_neighbour_classify(X_train_fold, y_train_fold, X_val_fold)
                    accuracy = np.mean(y_pred == y_val_fold)

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

    y_pred_test = nearest_neighbour_classify(X_train[:, best_subset], y_train, X_test[:, best_subset])
    test_accuracy = np.mean(y_pred_test == y_test)

    print(f'Final result: The best feature set is {[f + 1 for f in best_subset]}, with accuracy {round(test_accuracy * 100, 2)}% on test set')
    return best_subset, test_accuracy, accuracies, times






