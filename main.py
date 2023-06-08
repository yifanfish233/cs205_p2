import data_utils as du
from feature_search import forward_selection, backward_elimination
from sklearn.datasets import load_digits

def main():

    print("1. Small dataset")
    print("2. Medium dataset")
    print("3. Large dataset")
    print("4. abalone dataset")
    dataset_size = input("Please choose a dataset size: ")

    if dataset_size == "1":
        dataset = 'CS170_small_Data__15.txt'
        df = du.load_data(dataset)
    elif dataset_size == "2":
        dataset = 'CS170_large_Data__18.txt'
        df = du.load_data(dataset)
    elif dataset_size == "3":
        dataset = 'CS170_XXXlarge_Data__6.txt'
        df = du.load_data(dataset)
    elif dataset_size == "4":
        df= du.load_data("abalone.data",header=None,delim_whitespace=False)
    else:
        print("Invalid dataset size!")
        return

    X, Y,num_class = du.split_data(df)

    print("1. Forward Selection")
    print("2. Backward Elimination")
    selection_method = input("Please choose a feature selection method: ")

    if selection_method == "1":
        forward_selection(X, Y)
    elif selection_method == "2":
        X, Y = du.data_preprocess(X, Y)
        backward_elimination(X, Y, class_num=num_class)
    else:
        print("Invalid selection method!")

if __name__ == "__main__":
    main()
