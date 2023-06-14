import time
from copy import deepcopy

import data_utils as du
import numpy as np
import self_knn
from sklearn.model_selection import KFold
import random

def k_fold_cross_validation(data, n_splits, random_state):
    n_samples = len(data)
    fold_size = n_samples // n_splits
    remainder = n_samples % n_splits
    
    indices = list(range(n_samples))
    random.seed(random_state)
    random.shuffle(indices)
    
    splits = []
    start = 0
    for i in range(n_splits):
        end = start + fold_size
        if remainder > 0:
            end += 1
            remainder -= 1
        
        test_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]
        
        split = [train_indices,test_indices]
        splits.append(split)
        
        start = end
    
    return splits

def forward_selection(X, y, n_splits=10, random_state=42):
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

    kf = k_fold_cross_validation(X, n_splits, random_state)
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


def backward_elimination(X, y, test_size=0.3, n_splits=5, class_num=2, random_state=42):
    print('Beginning search.')
    X_train, X_test, y_train, y_test = du.self_train_test_split(X, y, test_size=test_size, random_state=random_state)

    idx_list = list(range(X_train.shape[1]))
    max_accuracy = 0
    best_features = idx_list
    feature_max_accuracy = [0] * len(idx_list)
    kf = k_fold_cross_validation(X_train, n_splits, random_state)

    # Initialize lists to store accuracies and timings
    accuracies = []
    times = []

    q = [idx_list]
    while len(q) > 0:
        cur_max_accuracy = 0
        cur_best_features = None
        start_time = time.time()

        for feature_subset in q:
            accuracy_list = []
            for split in kf:
                train_index = split[0]
                val_index = split[1]
                X_train_fold, X_val_fold = X_train[train_index][:, feature_subset], X_train[val_index][:,
                                                                                    feature_subset]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                knn = self_knn.MyKNNClassifier(n_neighbors=class_num)
                knn.fit(X_train_fold, y_train_fold)
                accuracy = knn.get_score(X_val_fold, y_val_fold)
                accuracy_list.append(accuracy)

            avg_accuracy = np.mean(accuracy_list)
            if avg_accuracy > cur_max_accuracy:
                cur_max_accuracy = avg_accuracy
                cur_best_features = feature_subset
                for f in feature_subset:
                    feature_max_accuracy[f] = max(feature_max_accuracy[f], cur_max_accuracy)

        end_time = time.time()

        # Record accuracy and time for this iteration
        accuracies.append(cur_max_accuracy)
        times.append(end_time - start_time)

        if cur_max_accuracy > max_accuracy:
            max_accuracy = cur_max_accuracy
            best_features = cur_best_features
            print(
                f'Current best is using features {[f + 1 for f in best_features]} with accuracy {round(max_accuracy * 100, 2)}%')
        else:
            break

        q = []
        for i in range(len(best_features)):
            if cur_max_accuracy >= feature_max_accuracy[best_features[i]]:
                subset = deepcopy(best_features)
                subset.pop(i)
                q.append(subset)

    # If accuracy is less than 60%, call the aggressive backward_elimination function
    accuracy_threshold = 0.8
    if max_accuracy < accuracy_threshold:
        print("Accuracy is less than " + str(accuracy_threshold) + "%, calling aggressive backward elimination.")
        best_features_agg, max_accuracy_agg, agg_accuracies, agg_times = backward_elimination_aggressive(X_train,
                                                                                                         y_train, kf,
                                                                                                         best_features,
                                                                                                         class_num)
        # append aggressive accuracies and times
        accuracies.extend(agg_accuracies)
        times.extend(agg_times)

        if max_accuracy < max_accuracy_agg:
            print("Aggressive backward elimination found a better feature set.")
            best_features = best_features_agg
        else:
            print("Aggressive backward elimination did not find a better feature set.")

    knn = self_knn.MyKNNClassifier(n_neighbors=class_num)
    knn.fit(X_train[:, best_features], y_train)
    test_accuracy = knn.get_score(X_test[:, best_features], y_test)

    print(
        f'Final result: The best feature set is {[f + 1 for f in best_features]}, with accuracy {round(test_accuracy * 100, 2)}% on test set')

    return best_features, test_accuracy, accuracies, times


def backward_elimination_aggressive(X_train, y_train, kf, best_features, class_num):
    print("Beginning aggressive search.")
    max_accuracy = 0
    best_subset = best_features

    # Initialize lists to store accuracies and timings
    accuracies = []
    times = []

    while len(best_features) > 1:
        cur_max_accuracy = 0
        cur_best_features = None

        # Generate next level feature subsets
        subsets = []
        for i in range(len(best_features)):
            subset = deepcopy(best_features)
            subset.pop(i)
            subsets.append(subset)

        start_time = time.time()

        for feature_subset in subsets:
            accuracy_list = []
            for train_index, val_index in kf:  # change this line
                X_train_fold, X_val_fold = X_train[train_index][:, feature_subset], X_train[val_index][:,
                                                                                    feature_subset]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                knn = self_knn.MyKNNClassifier(n_neighbors=class_num)
                knn.fit(X_train_fold, y_train_fold)
                accuracy = knn.get_score(X_val_fold, y_val_fold)
                accuracy_list.append(accuracy)

            avg_accuracy = np.mean(accuracy_list)
            if avg_accuracy > cur_max_accuracy:
                cur_max_accuracy = avg_accuracy
                cur_best_features = feature_subset

        end_time = time.time()

        # Record accuracy and time for this iteration
        accuracies.append(cur_max_accuracy)
        times.append(end_time - start_time)

        # Update best_features regardless of whether accuracy has improved
        best_features = cur_best_features
        print(
            f'Current subset is using features {[f + 1 for f in best_features]} with accuracy {round(cur_max_accuracy * 100, 2)}%')

        # Only update max_accuracy and best_subset if the accuracy has improved
        if cur_max_accuracy > max_accuracy:
            max_accuracy = cur_max_accuracy
            best_subset = cur_best_features

    print(
        f'Best subset found is using features {[f + 1 for f in best_subset]} with accuracy {round(max_accuracy * 100, 2)}%')

    return best_subset, max_accuracy, accuracies, times







