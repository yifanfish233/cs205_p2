from sklearn.metrics import accuracy_score
import data_utils as du
from sklearn.model_selection import train_test_split
import self_knn


def backward_elimination(X, y, test_size=0.2, random_state=42):
    # Scale the features to a range [0,1] for better performance of KNN
    X = du.normalize_data(X)
    X_train, X_test, y_train, y_test = du.self_train_test_split(X, y, test_size=test_size, random_state=random_state)

    knn = self_knn.MyKNNClassifier(n_neighbors=2)

    # Initialize with all features
    feature_subset = list(range(X.shape[1]))
    best_accuracy = 0
    best_features = feature_subset

    while len(feature_subset) > 0:
        accuracies = []
        for i in feature_subset:
            features_to_try = list(set(feature_subset) - set([i]))
            knn.fit(X_train[:, features_to_try], y_train)
            y_pred = knn.predict(X_test[:, features_to_try])
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            # Use 1-based indexing for printing
            print(f'Using features {[f+1 for f in features_to_try]}, the accuracy is {accuracy * 100}%')

        # Check if we found a new best
        max_accuracy = max(accuracies)
        if max_accuracy > best_accuracy:
            # Update our best accuracy and best features
            best_accuracy = max_accuracy
            best_features = feature_subset
            # remove the feature that gave us our best result
            feature_subset.remove(feature_subset[accuracies.index(max_accuracy)])

            # Use 1-based indexing for printing
            print(f'Current best is using features {[f+1 for f in best_features]} with accuracy {best_accuracy * 100}%')

        else:
            # If accuracy didn't improve, stop the loop
            break

    # Use 1-based indexing for printing
    print(f'Final result: The best feature set is {[f+1 for f in best_features]}, with accuracy {best_accuracy * 100}%')
    return best_features, best_accuracy



