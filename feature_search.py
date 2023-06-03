from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def forward_selection(data, target):
    best_features = []
    remaining_features = list(data.columns)
    best_new_score = 0  # Initialize with 0

    while remaining_features:
        new_scores = []
        for feature in remaining_features:
            model = LogisticRegression()
            use_features = best_features + [feature]
            X_train, X_test, y_train, y_test = train_test_split(data[use_features], target, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            new_scores.append((accuracy, feature))

            # Print the accuracy for each feature
            print(f"Using feature {feature}, accuracy is: {accuracy * 100}%")

        new_scores.sort(reverse=True)  # sort by accuracy
        current_score, best_feature = new_scores[0]
        if current_score > best_new_score:
            remaining_features.remove(best_feature)
            best_features.append(best_feature)
            best_new_score = current_score

            # Print the best result of this iteration
            print(f"This iteration, best result is using feature {best_feature}, accuracy is {current_score * 100}%")
        else:
            break

    return best_features, best_new_score


def backward_elimination(data, target):
    features = list(data.columns)
    best_score = 0

    while features:
        new_scores = []
        for feature in features:
            use_features = list(set(features) - set([feature]))
            model = LogisticRegression()
            X_train, X_test, y_train, y_test = train_test_split(data[use_features], target, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            new_scores.append((accuracy, feature))

            # Print the accuracy for each feature
            print(f"Without feature {feature}, accuracy is: {accuracy * 100}%")

        new_scores.sort(reverse=True)  # sort by accuracy
        best_new_score, worst_feature = new_scores[0]
        if best_new_score > best_score:
            features.remove(worst_feature)
            best_score = best_new_score

            # Print the best result of this iteration
            print(f"This iteration, best result is without feature {worst_feature}, accuracy is {best_new_score * 100}%")
        else:
            break

    return features, best_score


