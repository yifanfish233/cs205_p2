import data_utils as du
from sklearn.metrics import accuracy_score

from self_knn import MyKNNClassifier


def main():
    df = du.load_data('CS170_small_Data__15.txt')
    X_train, X_test, y_train, y_test = du.split_data(df)

    classifier = MyKNNClassifier(n_neighbors=2)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'The accuracy of the classifier on the test set is: {accuracy * 100}%')


if __name__ == "__main__":
    main()
