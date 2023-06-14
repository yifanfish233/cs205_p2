import time
import data_utils as du
from feature_search import forward_selection, backward_elimination


def main():
    print("1. Small dataset")
    print("2. Medium dataset")
    print("3. Large dataset")
    print("4. abalone dataset")
    dataset_size = input("Please choose a dataset size: ")

    if dataset_size == "1":
        # dataset = 'CS170_small_Data__15.txt'
        dataset = 'CS170_small_Data__33.txt'
        df = du.load_data(dataset)
    elif dataset_size == "2":
        # dataset = 'CS170_large_Data__18.txt'
        dataset = 'CS170_large_Data__32.txt'
        df = du.load_data(dataset)
    elif dataset_size == "3":
        dataset = 'CS170_XXXlarge_Data__6.txt'
        df = du.load_data(dataset)
    elif dataset_size == "4":
        df = du.load_data("abalone.data", header=None, delim_whitespace=False)
    else:
        print("Invalid dataset size!")
        return

    X, Y, num_class = du.split_data(df)

    print("1. Forward Selection")
    print("2. Backward Elimination")
    selection_method = input("Please choose a feature selection method: ")

    if selection_method == "1":
        start_time = time.time()
        forward_selection(X, Y)
        end_time = time.time()
        print("Time cost forward (second(s)): ", round(end_time - start_time, 3))
    elif selection_method == "2":
        X, Y = du.data_preprocess(X, Y)
        start_time = time.time()
        best_features, test_accuracy, accuracies, times = backward_elimination(X, Y, threshold=0.70,
                                                                               speed_priority=True)
        end_time = time.time()
        print("Time cost backward (second(s)): ", round(end_time - start_time, 3))
        du.print_for_analysis(accuracies, times, "backward_elimination")
        iterations = list(range(1, len(accuracies) + 1))
        du.save_results_to_csv(iterations, accuracies, times)
    else:
        print("Invalid selection method!")


if __name__ == "__main__":
    main()
