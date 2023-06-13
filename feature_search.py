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

def forward_selection(X, y, n_splits=5, random_state=42):
    #modify the code to use the KNN classifier and score to analysis the feature.
    print('Beginning search.')

    # Initialize with all features
    q = []
    for i in range(X.shape[1]):
        q.append([i])
    
    max_accuracy = 0
    best_features = q[0]
    kf = k_fold_cross_validation(X, n_splits, random_state)
    # kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    
    while len(q[0]) < X.shape[1]:
        cur_max_accuracy = 0
        cur_best_features = None
        # print("q: " , q)
        for feature_subset in q:
            accuracies = []
            # for train_index, test_index in kf.split(X):
            for split in kf:
                train_index = split[0]
                test_index = split[1]
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
    accuracy_threshold = 0.6
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
            for train_index, val_index in kf.split(X_train):
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







