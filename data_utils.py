import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(filename):
    df = pd.read_csv(filename, header=None, delim_whitespace=True)
    return df

def split_data(df,test_size=0.2, random_state=None):
    # May need adjust in the feature selection part
    X = df.iloc[:, 1:].values  # take all columns except the first as features
    y = df.iloc[:, 0].values  # take the first column as the target class

    # Print the number of instances, the number of classes, and the number of features
    print("***************")
    print(f"The dataset has {df.shape[0]} instances.")
    print(f"The dataset has {df.iloc[:, 0].nunique()} classes.")
    print(f"The dataset contains {df.shape[1] - 1} features.")
    print("***************")

    X_train, X_test, y_train, y_test = self_train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def normalize_data(X):
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

def self_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)

    # Create an array of indices and shuffle them
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Calculate the number of training examples
    train_size = int(X.shape[0] * (1 - test_size))

    # Split the indices for the train and test sets
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    # Use the indices to create the train and test sets
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
