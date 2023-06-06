import data_utils as du
from feature_search import forward_selection, backward_elimination

def main():

    print("1. Small dataset")
    print("2. Medium dataset")
    print("3. Large dataset")
    dataset_size = input("Please choose a dataset size: ")

    if dataset_size == "1":
        dataset = 'CS170_small_Data__15.txt'
    elif dataset_size == "2":
        dataset = 'CS170_large_Data__18.txt'
    elif dataset_size == "3":
        dataset = 'CS170_XXXlarge_Data__6.txt'
    else:
        print("Invalid dataset size!")
        return

    # Load the data
    df = du.load_data(dataset)

    # Split data into training and testing set
    # X_train, X_test, y_train, y_test = du.split_data(df, test_size=0.2, random_state=42)
    X, Y = du.split_data(df)

    print("1. Forward Selection")
    print("2. Backward Elimination")
    selection_method = input("Please choose a feature selection method: ")

    if selection_method == "1":
        selected_features, best_accuracy = forward_selection(X, Y)
    elif selection_method == "2":
        selected_features, best_accuracy = backward_elimination(X, Y)
    else:
        print("Invalid selection method!")

if __name__ == "__main__":
    main()
