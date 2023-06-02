from sklearn.neighbors import KDTree
import numpy as np

class MyKNNClassifier:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.kdtree = KDTree(X_train)
        self.y_train = np.array(y_train, dtype=int)  # ensure labels are integers

    def predict(self, X_test):
        distances, indices = self.kdtree.query(X_test, self.n_neighbors)
        y_pred = self.y_train[indices]  # find nearest neighbors
        return np.array([np.argmax(np.bincount(y)) for y in y_pred])  # return most common label
