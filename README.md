# Nearest Neighbor and Feature Selection

This project involves the implementation of the Nearest Neighbor (NN) classification algorithm and two feature selection methods: Forward Selection and Backward Elimination.

## Project Description

The Nearest Neighbor algorithm is a simple, yet powerful classification algorithm, but it's sensitive to irrelevant features. To overcome this, we implement two feature selection methods:

- Forward Selection: Begins with an empty set of features and iteratively adds the feature that most improves the performance of the Nearest Neighbor algorithm.

- Backward Elimination: Begins with the full set of features and iteratively removes the feature that least degrades the performance of the Nearest Neighbor algorithm.

The project includes processing three synthetic datasets of increasing sizes and a real-world classification dataset.



## Project Structure
- main.py: The main script to run the project.
- nn.py: Contains the implementation of the Nearest Neighbor algorithm.
- feature_selection.py: Contains the implementation of the Forward Selection and Backward Elimination methods.
- utils.py: Contains utility functions for data processing.

## Installation

```bash
git clone git@github.com:yifanfish233/cs205_p2.git
```

## Development
This project was developed by Yifan Yu and Kang-Yi Shih as part of CS 205 course project 2. 
The project is written in Python and uses NumPy for numerical computations.