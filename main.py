import data_utils as du
from feature_search import forward_selection, backward_elimination


def main():
    df = du.load_data('CS170_small_Data__15.txt')
    # Split the data into features and target variable
    X = df.drop(0, axis=1)
    y = df[0]

    print("1. Forward Selection")
    print("2. Backward Elimination")
    selection_method = input("Please choose a feature selection method: ")

    if selection_method == "1":
        selected_features, best_accuracy = forward_selection(X, y)
    elif selection_method == "2":
        selected_features, best_accuracy = backward_elimination(X, y)
    else:
        print("Invalid selection method!")
        return

    print(f"The best feature set is: {selected_features}, with accuracy: {best_accuracy * 100}%")


if __name__ == "__main__":
    main()
