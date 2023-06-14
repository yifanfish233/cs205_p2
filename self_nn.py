import numpy as np
class MyNNClassifier:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = np.array(y_train, dtype=int)  # ensure labels are integers
        self.y_train = self.y_train - min(0, self.y_train.min())  # ensure labels are non-negative

    def predict(self, X_test):
        y_pred = []
        for test_instance in X_test:
            # Using the Euclidean distance
            distances = np.sqrt(np.sum((self.X_train - test_instance) ** 2, axis=1))
            nearest_index = distances.argmin()  # find the index of the nearest neighbor
            y_pred.append(self.y_train[nearest_index])  # use the label of the nearest neighbor as prediction

        return np.array(y_pred, dtype=int)

    def get_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        correct = sum(y_pred == y_test)
        accuracy = correct / len(y_test)
        return accuracy