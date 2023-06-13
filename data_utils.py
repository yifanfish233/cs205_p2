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
    num_class = df.iloc[:, 0].nunique() # number of classes
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

def print_for_analysis(accuracies, times, folder_name):
    # Make directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plt.figure(figsize=(6, 6))
    plt.plot(accuracies, marker='o')
    plt.title('Accuracy per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f"{folder_name}/accuracy_per_iteration.png") # Save the figure
    plt.show() # Display the figure

    plt.figure(figsize=(6, 6))
    plt.plot(times, marker='o')
    plt.title('Latency per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Latency (seconds)')
    plt.grid(True)
    plt.savefig(f"{folder_name}/latency_per_iteration.png") # Save the figure
    plt.show() # Display the figure

