import numpy as np
class MyKNNClassifier:
    def __init__(self, n_neighbors=2):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = np.array(y_train, dtype=int)  # ensure labels are integers
        self.y_train = self.y_train - min(0, self.y_train.min())  # ensure labels are non-negative

    def predict(self, X_test):
        y_pred = []
        for test_instance in X_test:
            distances = np.sqrt(np.sum((self.X_train - test_instance) ** 2, axis=1))
            nearest_indices = distances.argsort()[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]

            if self.n_neighbors == 1:
                y_pred.append(nearest_labels[0])  # if only one neighbor, return its label
            else:
                y_pred.append(np.argmax(np.bincount(nearest_labels.astype('int'))))  # vote by majority

        return np.array(y_pred, dtype=int)

    def get_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        correct = sum(y_pred == y_test)
        accuracy = correct / len(y_test)
        return accuracy