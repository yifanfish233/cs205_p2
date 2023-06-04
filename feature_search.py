import data_utils as du
import numpy as np
import self_knn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
def forward_selection(X, y, n_splits=5, random_state=42):
    #modify the code to use the KNN classifier as the base classifier.
    pass


def backward_elimination(X, y, n_splits=5, random_state=42):
    """
    Perform backward elimination feature selection using KNN classifier.

    Args:
        X (array-like): Input feature matrix.
        y (array-like): Target variable.
        n_splits (int): Number of splits for cross-validation. Default is 5.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        best_features (list): List of best selected feature indices.
        best_accuracy (float): Best achieved accuracy.

    """

    # Scale the features to a range [0,1] for better performance of KNN
    X = du.normalize_data(X)

    # Initialize KNN classifier
    knn = self_knn.MyKNNClassifier(n_neighbors=2)

    # Initialize with all features
    feature_subset = list(range(X.shape[1]))
    best_accuracy = 0
    best_features = feature_subset

    # Perform cross-validation
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        while len(feature_subset) > 1:
            accuracies = []

            for i in feature_subset:
                # Remove one feature at a time and evaluate accuracy
                features_to_try = list(set(feature_subset) - set([i]))
                knn.fit(X_train[:, features_to_try], y_train)
                y_pred = knn.predict(X_test[:, features_to_try])
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)

            # Use 1-based indexing for printing
            print(
                f'Using features {[f + 1 for f in features_to_try]}, the accuracy is {round(np.mean(accuracies) * 100, 2)}%')

            # Check if we found a new best
            max_accuracy = np.mean(accuracies)
            if max_accuracy > best_accuracy:
                # Update our best accuracy and best features
                best_accuracy = max_accuracy
                best_features = feature_subset
                # Remove the feature that gave us the best result
                feature_subset.remove(feature_subset[np.argmax(accuracies)])

                # Use 1-based indexing for printing
                print(
                    f'Current best is using features {[f + 1 for f in best_features]} with accuracy {round(best_accuracy * 100, 2)}%')
            else:
                # If accuracy didn't improve, stop the loop
                break

    # Use 1-based indexing for printing
    print(
        f'Final result: The best feature set is {[f + 1 for f in best_features]}, with accuracy {round(best_accuracy * 100, 2)}%')
    return best_features, best_accuracy





