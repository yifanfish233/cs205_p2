from sklearn.neighbors import KDTree
import numpy as np

class MyKNNClassifier:
    def __init__(self, n_neighbors=2):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.kdtree = KDTree(X_train)
        self.y_train = np.array(y_train, dtype=int)  # ensure labels are integers
        self.y_train = self.y_train - min(0, self.y_train.min())  # ensure labels are non-negative

    def predict(self, X_test):
        distances, indices = self.kdtree.query(X_test, self.n_neighbors)
        y_pred = self.y_train[indices]  # find the labels of the nearest neighbors
        if self.n_neighbors == 1:
            return y_pred  # if only one neighbor, return its label
        else:
            # if multiple neighbors, return the label that appears most often
            return np.array([np.argmax(np.bincount(labels.astype('int'))) for labels in y_pred]).astype(int)
