import random

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt


def load_data(filename, header=None, delim_whitespace=True):
    df = pd.read_csv(filename, header=header, delim_whitespace=delim_whitespace)
    # Check if the classes (in the first column) are already integers
    if df.iloc[:, 0].dtype != 'int64' and df.iloc[:, 0].dtype != 'float64':
        print("Class names are not integers. Mapping them to integers.")
        # Create a dictionary to map original class names to integers
        class_map = {class_name: i for i, class_name in enumerate(df.iloc[:, 0].unique())}
        # Apply the mapping to the first column of the dataframe
        df.iloc[:, 0] = df.iloc[:, 0].map(class_map)
        print("Class mapping:")
        for class_name, mapped_value in class_map.items():
            print(f"{class_name} -> {mapped_value}")
    else:
        print("Class names are already integers. No mapping needed.")

    return df


def split_data(df):
    # May need adjust in the feature selection part
    X = df.iloc[:, 1:].values  # take all columns except the first as features
    Y = df.iloc[:, 0].values  # take the first column as the target class
    num_class = df.iloc[:, 0].nunique()  # number of classes
    # Print the number of instances, the number of classes, and the number of features
    print("***************")
    print(f"The dataset has {df.shape[0]} instances.")
    print(f"The dataset has {df.iloc[:, 0].nunique()} classes.")
    print(f"The dataset contains {df.shape[1] - 1} features.")
    print("***************")
    return X, Y, num_class

def normalize_data(X):
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

def self_train_test_split(X, Y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)

    # Create an array of indices and shuffle them
    indices = np.random.permutation(X.shape[0])

    # Calculate the number of training examples
    train_size = int(X.shape[0] * (1 - test_size))

    # Split the indices for the train and test sets
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    # Use the indices to create the train and test sets
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = Y[train_indices], Y[test_indices]

    return X_train, X_test, y_train, y_test


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

        split = [train_indices, test_indices]
        splits.append(split)

        start = end

    return splits


def data_preprocess(X, y):
    # check if there is any missing value, if yes, use mean to fill it
    if np.isnan(X).any():
        print("There is missing value in the dataset, use mean to fill it")
        X = np.nan_to_num(X)

    # Normalize skewed features
    for i in range(X.shape[1]):
        if abs(skew(X[:, i])) > 1:
            print(f"Feature {i + 1} is skewed, performing standard scaling")
            X[:, i] = StandardScaler().fit_transform(X[:, i].reshape(-1, 1)).reshape(-1)

    return X, y


def analyze_and_save_results(iterations, accuracies, times, csv_filename='results.csv', directory='results_data'):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate and save the accuracy plot
    plt.figure(figsize=(6, 6))
    plt.plot(accuracies, marker='o')
    plt.title('Accuracy per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f"{directory}/accuracy_per_iteration.png")
    plt.show()

    # Generate and save the latency plot
    plt.figure(figsize=(6, 6))
    plt.plot(times, marker='o')
    plt.title('Latency per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Latency (seconds)')
    plt.grid(True)
    plt.savefig(f"{directory}/latency_per_iteration.png")
    plt.show()

    # Save the results to a CSV file
    df = pd.DataFrame({
        'Iteration': iterations,
        'Accuracy': accuracies,
        'Latency': times
    })
    df.to_csv(os.path.join(directory, csv_filename), index=False)

