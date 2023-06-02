import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_data(filename):
    df = pd.read_csv(filename, header=None, delim_whitespace=True)
    return df

def split_data(df):
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def knn_classification(X_train, X_test, y_train):
    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test)

def main():
    df = load_data('CS170_small_Data__15.txt')
    X_train, X_test, y_train, y_test = split_data(df)
    y_pred = knn_classification(X_train, X_test, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'The accuracy of the classifier on the test set is: {accuracy*100}%')

if __name__ == "__main__":
    main()