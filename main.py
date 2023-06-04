import data_utils as du
from feature_search import backward_elimination

def main():

    # Load the data
    df = du.load_data('CS170_small_Data__15.txt')

    # Split data into training and testing set
    X_train, X_test, y_train, y_test = du.split_data(df, test_size=0.2, random_state=42)

    print("1. Forward Selection")
    print("2. Backward Elimination")
    selection_method = input("Please choose a feature selection method: ")

    if selection_method == "1":
        pass
    elif selection_method == "2":
        selected_features, best_accuracy = backward_elimination(X_train, y_train)
        print(f"The best feature set is: {selected_features}, with accuracy: {best_accuracy * 100}%")
    else:
        print("Invalid selection method!")

if __name__ == "__main__":
    main()
